"""MHC fused-stats HIP needle (needle-ized from the 2026-10-22 manual tree
patches, four-gates-verified on nmz26 that night; tree pins kept the win but
must not stay manual divergences).

Two flips, both module-global rebinds (no source rewrite needed):

1. ``sglang.srt.models.deepseek_v4_mhc._is_gfx95_supported = True`` — makes
   ``use_fused_stats`` true on HIP so the MHC pre boundary takes the fused
   stats path (hc_mix_stats_sinkhorn -> triton partial + fused reduce)
   instead of the ~10-torch-kernel chain per layer (manual patch had added
   ``or torch.version.hip is not None``; alias-True is equivalent because the
   alias is used nowhere else).
2. ``sglang.kernels.ops.layernorm.mhc``: under the HCU cuda-semantics seam
   the module-level ``is_hip()`` read False at import, so the HIP import
   block (``hc_mix_reduce_sinkhorn_vec``) never ran. Rebind ``_is_hip`` and
   backfill the missing import; the call site resolves it by module-global
   lookup at runtime.

Gate: ``HCU_MHC_STATS`` (default on; ``=0`` restores the torch chain).
"""

from __future__ import annotations

import logging
import os

import torch

LOG = logging.getLogger("hcu.adapter")


def install() -> None:
    if os.environ.get("HCU_MHC_STATS") == "0":
        return
    if torch.version.hip is None:
        return  # hardware truth: only meaningful on an HIP runtime

    import sglang.srt.models.deepseek_v4_mhc as _model_mhc

    if not getattr(_model_mhc, "_is_gfx95_supported", False):
        _model_mhc._is_gfx95_supported = True

    import sglang.kernels.ops.layernorm.mhc as _kern_mhc

    flipped = False
    if not getattr(_kern_mhc, "_is_hip", False):
        _kern_mhc._is_hip = True
        flipped = True
    # Two more constants froze at import from the same seam-read `_is_hip`
    # (nmz18 2026-10-23: arm-B boot died in triton compile --
    # "input_precision must be one of ('ieee', 'tf32'). Got tf32x3").
    # Launch sites read these module globals at call time, so rebinding is
    # enough; "ieee" is the fp32 MFMA path the HIP arm is designed for.
    if getattr(_kern_mhc, "_HC_MIX_DOT_PRECISION", "ieee") != "ieee":
        _kern_mhc._HC_MIX_DOT_PRECISION = "ieee"
        flipped = True
    if getattr(_kern_mhc, "_HC_MIX_NUM_STAGES", 2) != 2:
        _kern_mhc._HC_MIX_NUM_STAGES = 2
        flipped = True
    if not hasattr(_kern_mhc, "hc_mix_reduce_sinkhorn_vec"):
        from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
            hc_mix_reduce_sinkhorn_vec,
        )

        _kern_mhc.hc_mix_reduce_sinkhorn_vec = hc_mix_reduce_sinkhorn_vec
        flipped = True
    if flipped:
        LOG.warning(
            "HCU adapter: MHC fused-stats HIP needle installed "
            "(use_fused_stats + mhc_boundary_hip import; HCU_MHC_STATS=0"
            " disables)"
        )
