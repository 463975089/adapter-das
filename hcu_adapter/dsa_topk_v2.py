"""DSA top-k v2 fusion enablement for DeepSeek-V4.1 decode (HIP arms).

2026-10-22 trace attribution (nmz26, torch-profiler decode20): the indexer
top-k of every decode step ran torch.topk -- 57 mbtopk-family kernels baked
into the verify graph, of which 4 groups x ~920us came from the four
ratio-1/2 index-source layers' FullTopKIndexer._torch_decode (upstream keeps
a torch topk/sort/gather/mask branch for non-SM90) and the rest from the C4
dispatch, whose v2 gate (SGLANG_OPT_USE_TOPK_V2) the upstream model_hook
force-disables for DSA-family archs when not explicitly set. Upstream ships
its own fused JIT kernel (topk_v2) for both surfaces; these seams make it
usable on DTK gfx938:

1. stride seam -- the v2 kernel requires a 4-multiple score row stride
   (16-byte vectorized load, topk_v2.cuh:1044); production lmax
   (max_compressed_seq_len) is not aligned. When unaligned, pad the tail
   columns with -inf (they can never win selection; lens/plan unchanged).
   No-op when already aligned. Applies to every caller of
   ``topk_transform_paged_v2``.
2. v41 full-topk arm -- ``select_decode``'s DecodeScores branch routes
   through the fused kernel using the per-row slot table as a page_size=1
   transform, replacing topk+sort+gather+mask. Parity vs the torch branch:
   5/5 random seeds row-exact + len=0 all -1 edge, nmz26 2026-10-22
   (contract: top-min(k,len) of the first lens[b] scores, slots-table
   mapped, -1 padded, ascending).

The C4-side flip is NOT automated: it stays an explicit recipe env
(``SGLANG_OPT_USE_TOPK_V2=1``, exempt from the hook's "not is_set" demotion
per model_hook's own honoring rule). Gates: ``HCU_TOPK_V2=0`` disables both
seams; ``HCU_V41_TOPK_V2=0`` disables only the v41 arm.
"""

import logging
import sys

import torch

LOG = logging.getLogger("hcu.adapter")

_TOPK_MOD = "sglang.kernels.ops.attention.dsv4.topk"
_SCORING_MOD = "sglang.srt.layers.attention.dsv4.v41_indexer.scoring"


def _wrap_paged_v2(orig):
    def topk_transform_paged_v2(
        scores, seq_lens, page_tables, out_page_indices, page_size, metadata,
        out_raw_indices=None,
    ):
        if scores.dim() == 2 and scores.stride(0) % 4 != 0:
            scores = torch.nn.functional.pad(
                scores, (0, 4 - scores.stride(0) % 4), value=float("-inf")
            )
        return orig(
            scores, seq_lens, page_tables, out_page_indices, page_size,
            metadata, out_raw_indices,
        )

    topk_transform_paged_v2._hcu_topk_v2 = True
    return topk_transform_paged_v2


def _wrap_select_decode(orig, v41_enabled):
    def select_decode(inputs, d, topk):
        if not v41_enabled:
            return orig(inputs, d, topk)
        try:
            from sglang.kernels.ops.attention.dsv4.topk import (
                plan_topk_v2,
                topk_transform_paged_v2,
            )
            from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
                PagedDecodeScores,
            )
        except Exception:
            return orig(inputs, d, topk)
        k = min(topk, d.lmax)
        if isinstance(d, PagedDecodeScores) or k <= 0 or d.bs <= 0:
            return orig(inputs, d, topk)
        if d.scores.dtype != torch.float32 or d.scores.dim() != 2:
            return orig(inputs, d, topk)
        lens32 = d.lens.to(torch.int32)
        idx = torch.empty(
            (d.bs, k), dtype=torch.int32, device=d.scores.device
        )
        topk_transform_paged_v2(
            d.scores, lens32, d.slots.to(torch.int32), idx, 1,
            plan_topk_v2(lens32),
        )
        inputs.out_page_indices[: d.bs, : k] = idx

    select_decode._hcu_topk_v2 = True
    return select_decode


def _rebind(name, orig, wrapped):
    for mod in list(sys.modules.values()):
        if getattr(mod, name, None) is orig:
            setattr(mod, name, wrapped)


def install() -> None:
    from . import hcu_env as _env

    if not _env.enabled("HCU_TOPK_V2"):
        return

    installed = []
    try:
        import importlib

        m = importlib.import_module(_TOPK_MOD)
        orig = getattr(m, "topk_transform_paged_v2", None)
        if orig is not None and not getattr(orig, "_hcu_topk_v2", False):
            wrapped = _wrap_paged_v2(orig)
            setattr(m, "topk_transform_paged_v2", wrapped)
            _rebind("topk_transform_paged_v2", orig, wrapped)
            installed.append("stride-pad seam")
    except Exception:
        LOG.warning("HCU dsa topk v2: stride seam not installed", exc_info=True)

    try:
        import importlib

        s = importlib.import_module(_SCORING_MOD)
        orig = getattr(s, "select_decode", None)
        if orig is not None and not getattr(orig, "_hcu_topk_v2", False):
            wrapped = _wrap_select_decode(orig, _env.enabled("HCU_V41_TOPK_V2"))
            setattr(s, "select_decode", wrapped)
            _rebind("select_decode", orig, wrapped)
            installed.append("v41 fused decode arm")
    except Exception:
        LOG.warning("HCU dsa topk v2: v41 arm not installed", exc_info=True)

    if installed:
        LOG.warning(
            "HCU dsa topk v2 fusion installed (%s); HCU_TOPK_V2=0 / "
            "HCU_V41_TOPK_V2=0 disable; C4 flip needs explicit "
            "SGLANG_OPT_USE_TOPK_V2=1",
            ", ".join(installed),
        )
