# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""torch paged-MQA logits for the v0.5.21 DSA indexer (is_cuda semantic branch).

Upstream call site (dsa_indexer.py, ``deep_gemm.fp8_paged_mqa_logits`` arm):
    q      (N, H, 128) fp8e4m3 — per-head q scales are pre-folded by the
           caller into ``weights`` (N_CTX heads x 1 fp32); the per-dim
           quant is hadamard-flattened so a straight dot is the contract.
    kv     fused cache viewed (P, block_kv=64, 1, 132): each PAGE is k in
           the 16-tile interleave order [0, PS*128) + PS fp32 scales at
           the tail [PS*128, PS*132) (kpool writer family / SetKAndS
           under the forced preshuffle gate; row-major arm behind
           HCU_DSA_PRESHUFFLE=0 — see the unpack comment below).
    ctx    int32 context lens, 2D (N, 1) after the (N_total,1) normalisation.
    bt     int32 page table (N, L) — page indices; token t = page*64 + slot.
    sched  get_paged_mqa_logits_metadata(ctx, 64, num_sms) opaquely — not
           interpreted by the kernel contract, only scheduling.
    out    fp32 logits (N, max_len); clean_logits=False leaves stale values
           outside each row's context (caller reads only valid windows and
           later top-k masks), True zeroes them.

The is_hip arm routes to aiter, which HCU never takes under the is_cuda
platform semantic; this emulation keeps identical numerics paths to the
proven HCU emu (fp8->bf16 exact dequant, bf16 GEMM widening in fp32, fp32
per-token scale fold), tiling over (rows, tokens) to stay capture-safe.
"""

import os

import torch

from .kernels.lightop_usage import _count

_HEAD_PAD = 132

_LIGHTOP_PAGED = []  # [fn-or-None] lightop.gemmopt.paged_mqa_logits probe
_WARNED = set()


def _lightop_paged_fn():
    if not _LIGHTOP_PAGED:
        try:
            from lightop.gemmopt import paged_mqa_logits as fn
        except Exception:
            fn = None
        _LIGHTOP_PAGED.append(fn)
    return _LIGHTOP_PAGED[0]


def _warn_once(key, msg):
    if key in _WARNED:
        return
    _WARNED.add(key)
    import logging

    logging.getLogger("hcu.adapter").warning(msg)


def _fp8_view_dtype():
    try:
        # 替代档: 无（fp8 dtype probe helper；包面未导出 is_fp8_fnuz）
        from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

        return torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
    except Exception:
        return torch.float8_e4m3fn


def _as_view(t, dtype):
    if t.dtype in (torch.uint8, torch.int8):
        return t.contiguous().view(dtype)
    return t


def fp8_paged_mqa_logits(
    q_fp8,
    kv_cache_fp8,
    weights,
    context_lens,
    block_table,
    schedule_metadata=None,
    max_context_len=None,
    clean_logits=False,
    **_ignored,
):
    """Torch paged-MQA logits, tile loop (reference) or full-window vec form.

    Contract & numerics: see module docstring. ``HCU_MQA_LOGITS_IMPL=vec``
    (rows <= HCU_MQA_VEC_MAXROWS) does the whole window in one gather+bmm
    pass: the head-weight contraction runs as a second bmm (fp32-accumulated)
    instead of materialising the (N, H, T) fp32 tensor -- a whole-window
    .float() there hits the DTK elementwise cliff (~0.8 GB/s) and made the
    naive vec form 3.5x slower than loop in serving. Parity vs loop: rel
    ~2e-3 (w rounded to bf16), top-k stable.
    """
    if os.environ.get("HCU_DUMP_TOPK") == "1" and not getattr(fp8_paged_mqa_logits, "_hcu_p", False):
        fp8_paged_mqa_logits._hcu_probe_done = True
    if os.environ.get("HCU_DUMP_TOPK") == "1":
        try:
            qs = float(q_fp8.float().abs().sum()) if hasattr(q_fp8, "float") else -1
            ks = float(kv_cache_fp8.float().abs().sum()) if hasattr(kv_cache_fp8, "float") else -1
            print(f"[HCU mqain] q_abs={qs:.4f} kv_abs={ks:.4f} qshape={tuple(q_fp8.shape)} kvshape={tuple(kv_cache_fp8.shape)}", flush=True)
        except Exception as e:
            print("[HCU mqain] err", e, flush=True)
    if max_context_len is None:
        max_context_len = int(block_table.shape[1]) * 64
    N = q_fp8.shape[0] if q_fp8.dim() == 3 else q_fp8.shape[0] * q_fp8.shape[1]
    if q_fp8.dim() == 4:
        q_flat = q_fp8.reshape(-1, q_fp8.shape[-2], q_fp8.shape[-1])
        H, D = q_flat.shape[-2], q_fp8.shape[-1]
        if q_fp8.dtype in (torch.uint8, torch.int8):
            q_flat = q_flat.contiguous().view(*q_flat.shape[:-1], D)
        # (B, next_n, H, D) -> rows are B*next_n; decode passes next_n=1 views
        q_b = _as_view(q_flat, _fp8_view_dtype()).to(torch.bfloat16).contiguous()
        N, H, D = q_b.shape
    else:
        q_b = _as_view(q_fp8, _fp8_view_dtype()).to(torch.bfloat16).contiguous()
        N, H, D = q_b.shape

    if kv_cache_fp8.dim() == 2:
        kv_cache_fp8 = kv_cache_fp8.reshape(-1, 64, 1, kv_cache_fp8.shape[-1])
    PS = kv_cache_fp8.shape[1] if kv_cache_fp8.dim() == 4 else 1
    # Pool layout (2026-10-18 R-02 audit, final): every page is k in the
    # 16-tile INTERLEAVE order + PS fp32 scales at the tail — written by
    # the kpool writer family (kpool_softmax_rotate_write_cache et al.,
    # stock->kpool_write_emu) and SetKAndS under the preshuffle gate the
    # adapter forces on; the prefill gather reads the same order
    # (_kpool_cache_k_offsets), so prefill is consistent. fused_store
    # (row-major) never fires for kpool>1 models (IndexerKPool has its
    # own store flow). The original row-level 132-byte split misread BOTH
    # layouts (k shifted 4*s bytes per slot, scales read off k bytes) —
    # decode DSA top-k was deterministically wrong in production; this
    # page-level + interleave-aware unpack is the fix. HCU_DSA_PRESHUFFLE=0
    # (legacy / kpool=1 fused_store world) selects the row-major arm;
    # PS=1 degenerates to the same bytes either way.
    pool2d = kv_cache_fp8.reshape(kv_cache_fp8.shape[0], -1).contiguous()
    if pool2d.shape[1] == PS * 132:
        kreg = pool2d[:, : PS * 128]
        preshuffle = os.environ.get("HCU_DSA_PRESHUFFLE", "1") != "0"
        tile = 16
        try:
            # 替代档: 无（私有 PRESHUFFLE tile 常量；仅模块体可见）
            from sglang.kernels.ops.attention.dsa.kpool_fp8_index import (
                INDEXER_K_CACHE_PRESHUFFLE_TILE as _tile,
            )

            tile = int(_tile)
        except Exception:
            pass
        if preshuffle and tile and PS >= tile and PS % tile == 0:
            # k[slot, d] -> (slot//t)*(t*128) + (d//t)*(t*t) + (slot%t)*t
            # + d%t within the page k region: view (st, dt, si, di) and
            # permute to logical (slot, d).
            P = kreg.shape[0]
            kreg = (
                kreg.contiguous()
                .view(P, PS // tile, 128 // tile, tile, tile)
                .permute(0, 1, 3, 2, 4)
                .reshape(P * PS, 128)
            )
        else:
            kreg = kreg.reshape(-1, 128)
        k_b = _as_view(kreg, _fp8_view_dtype()).to(torch.bfloat16).contiguous()
        k_sf = (
            pool2d[:, PS * 128 :]
            .contiguous()
            .view(torch.float32)
            .reshape(-1)
        )
    else:
        # bf16/fp16 packed cache (P, S, 1, D): no per-token scale,
        # reshape any trailing head-dim padding back to tokens x D.
        kb16 = kv_cache_fp8.contiguous()
        k_b = kb16.to(torch.bfloat16).reshape(-1, kb16.shape[-1])
        k_sf = torch.ones(k_b.shape[0], device=kb16.device)

    if weights.dim() == 3:
        weights = weights.reshape(weights.shape[0], -1)
    w = weights.reshape(q_b.shape[0], H).to(torch.float32)
    ctx = context_lens.to(torch.int64).reshape(-1)
    bt = block_table.to(torch.int64).contiguous()

    device = q_b.device

    # impl selection (registry-gated): "vec" collapses the tile loop into a
    # single full-window pass for decode-shaped rows. The double loop below
    # tiles (64, 2048); the CAPTURED decode graph runs it with the full
    # max-context window (e.g. 64K -> 32 inner iterations x ~12 aten ops x
    # 45 layers ~= 17k of the ~24k graph nodes = the submission bottleneck).
    # Large-N (prefill) keeps the loop: full-window gather there would need
    # N*T*128*2B transient memory (8K tokens x 64K window = impossible).
    impl = "loop"
    try:
        from .hcu_env import resolve

        impl, _src = resolve("HCU_MQA_LOGITS_IMPL")
    except Exception:
        pass
    if impl == "lightop":
        # Explicit tier only: the catalog status is planned (gate pending),
        # and a mis-computing kernel that does not raise would silently
        # corrupt the auto ladder, so auto never selects it. An explicit
        # =lightop request is honoured and falls through on failure.
        fn = _lightop_paged_fn()
        if fn is not None:
            try:
                _count("mqa.paged.lightop")
                return fn(
                    q_fp8,
                    kv_cache_fp8,
                    weights,
                    context_lens,
                    block_table,
                    schedule_metadata,
                    int(max_context_len),
                    clean_logits,
                    False,
                )
            except Exception:
                if impl == "lightop":
                    _warn_once(
                        "mqa-lightop",
                        "HCU_MQA_LOGITS_IMPL=lightop failed; falling through "
                        "the auto ladder",
                    )
                    import traceback

                    traceback.print_exc()
        elif impl == "lightop":
            _warn_once(
                "mqa-lightop-missing",
                "HCU_MQA_LOGITS_IMPL=lightop but lightop.gemmopt is not "
                "importable; falling through the auto ladder",
            )
    if impl == "vendor":
        # R-02: DTK deepgemm wheel native path, A/B tier (parity gate
        # tests/kernels/test_mqa_vendor_gfx938.py; layout unvalidated).
        from .mqa_vendor import deepgemm_paged_mqa_logits as _dg

        return _dg(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_table,
            schedule_metadata=schedule_metadata,
            max_context_len=max_context_len,
            clean_logits=clean_logits,
        )
    if impl in ("auto", "triton"):
        # R-02 homegrown tier: reads the REAL SetKAndS pool layout (16-tile
        # interleave + tail scales) that this emu's flat (T,132) unpack
        # misreads — correctness fix + single-launch decode graph. Parity
        # gate tests/kernels/test_mqa_triton_gfx938.py.
        from .mqa_logits_triton import triton_paged_mqa_logits as _tri

        try:
            return _tri(
                q_fp8,
                kv_cache_fp8,
                weights,
                context_lens,
                block_table,
                schedule_metadata=schedule_metadata,
                max_context_len=max_context_len,
                clean_logits=clean_logits,
            )
        except Exception:
            if impl == "triton":
                raise
            _warn_once(
                "mqa-triton",
                "MQA auto: triton arm refused this face; falling back to "
                "the torch loop",
            )
    vec_max = int(os.environ.get("HCU_MQA_VEC_MAXROWS", "32"))
    if impl == "vec" and N <= vec_max:
        T_all = min(int(max_context_len), bt.shape[1] * PS)
        n_pages = min((T_all + PS - 1) // PS, bt.shape[1])
        pages = bt[:, :n_pages]  # (N, P)
        toks = (pages.unsqueeze(2) * PS + torch.arange(PS, device=device))
        toks = toks.reshape(N, -1)[:, :T_all]  # (N, T)
        kg = k_b[toks]  # (N, T, D)
        sc = k_sf[toks]  # (N, T)
        # both bmms put the SMALL operand on the possibly-copied side:
        # sT = kg @ q^T and row = sT @ w^T never materialise a transposed
        # (N, T, D) buffer. The H-contraction runs inside the second bmm
        # (fp32-accumulated) -- a whole-window .float() of the (N, H, T)
        # product hits the DTK elementwise cliff (~0.8 GB/s) and made the
        # naive vec form 3.5x slower than loop in serving.
        sT = torch.matmul(kg, q_b.transpose(1, 2))  # (N, T, H) bf16
        row = torch.matmul(sT, w.unsqueeze(2).to(torch.bfloat16)).squeeze(2)  # (N, T)
        row = row.float() * sc
        valid = torch.arange(T_all, device=device).unsqueeze(0) < ctx.unsqueeze(1)
        if T_all >= int(max_context_len):
            return torch.where(valid, row, 0.0)
        # caller window exceeds page-table coverage: keep output width
        out = torch.zeros((N, int(max_context_len)), dtype=torch.float32, device=device)
        out[:, :T_all] = torch.where(valid, row, 0.0)
        return out

    logits = torch.zeros((N, int(max_context_len)), dtype=torch.float32, device=device)

    qt = int(os.environ.get("HCU_DSA_MQA_QTILE", "64"))
    tt = int(os.environ.get("HCU_DSA_MQA_TTILE", "2048"))
    ar = torch.arange(tt, device=device)
    for i0 in range(0, N, qt):
        i1 = min(N, i0 + qt)
        qb = q_b[i0:i1]  # (Q, H, D)
        bt_rows = bt[i0:i1]
        seq_rows = ctx[i0:i1]
        for t0 in range(0, int(max_context_len), tt):
            t1 = min(int(max_context_len), t0 + tt)
            pages = bt_rows[:, t0 // PS : (t1 + PS - 1) // PS]  # (Q, P)
            toks = pages.unsqueeze(2) * PS + torch.arange(PS, device=device)
            toks = toks.reshape(pages.shape[0], -1)[:, : t1 - t0]  # (Q, T)
            kg = k_b[toks]  # (Q, T, D)
            sc = k_sf[toks]  # (Q, T)
            s = torch.matmul(qb, kg.transpose(1, 2)).float()  # (Q, H, T)
            row = (s * w[i0:i1].unsqueeze(-1)).sum(dim=1) * sc  # (Q, T)
            valid = ar[: t1 - t0].unsqueeze(0) + t0 < seq_rows.unsqueeze(1)
            logits[i0:i1, t0:t1] = torch.where(valid, row, 0.0)
    return logits


def get_num_sms():
    try:
        return torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    except Exception:
        return 120


_LIGHTOP_META = []


def _lightop_metadata_fn():
    if not _LIGHTOP_META:
        try:
            from lightop.gemmopt import get_paged_mqa_logits_metadata as fn
        except Exception:
            fn = None
        _LIGHTOP_META.append(fn)
    return _LIGHTOP_META[0]


def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms):
    """Scheduling blob for ``fp8_paged_mqa_logits``.

    Contract (DeepGEMM): an int32 [*, 2] split-KV plan sized by num_sms that
    only the paged-MQA kernel reads back. The torch/triton tiers here ignore it, but
    any lightop ``paged_mqa_logits`` consumer asserts a genuine plan, so the
    lightop planner is the producer whenever it is installed; the zero blob
    is kept only for boxes without lightop.
    """
    rows = context_lens.reshape(-1)
    planner = _lightop_metadata_fn()
    if planner is not None:
        _count("mqa.paged_metadata.native")
        return planner(rows.to(torch.int32).contiguous(), block_kv, num_sms)
    meta = torch.zeros(
        (rows.numel() + 1, 2), dtype=torch.int32, device=rows.device
    )
    return meta


fp8_paged_mqa_logits.__name__ = "fp8_paged_mqa_logits"
