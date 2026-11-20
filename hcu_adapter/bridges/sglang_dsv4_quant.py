# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""DSV4 quant/rope family bridges (silu-fused-quant, MHC stats, sparse rope/norm, q/k norm+rope)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging


def install(_he=None) -> None:
    # DSV4 MoE silu-clamp x per-token-group-128 fp8 quant in ONE triton kernel
    # (replaces the swiglu-clamp bf16 write + invoke-internal re-quant pair;
    # upstream's fused JIT cuh family is numerically broken on gfx938 --
    # deterministic ~3.5% wrong elements, probes 2026-10-23). Must run AFTER
    # the swiglu-clamp rebind above (the fallback branch depends on it).
    # HCU_SILU_FUSED_QUANT=0 restores the pair.
    try:
        from hcu_adapter.features.dsv4 import dsv4_silu_fused_quant

        dsv4_silu_fused_quant.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: silu fused-quant needle not installed", exc_info=True
        )

    # MHC pre-boundary fused stats on HIP (needle-ized from the 2026-10-22
    # manual tree patches): use_fused_stats flip + mhc_boundary_hip import
    # backfill under the cuda-semantics seam. HCU_MHC_STATS=0 disables.
    try:
        from hcu_adapter.features.dsv4 import dsv4_mhc_stats

        dsv4_mhc_stats.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: MHC fused-stats needle not installed", exc_info=True
        )

    # DSV4.1 sparse indexer rope/norm triton gates: dsv41_sparse gates its
    # two pure-triton kernels (rope_tail_fake_quant_fp4, rmsnorm_fp32) on
    # `torch.version.cuda is not None or is_gfx95_supported()` — both False
    # on DTK (cuda=None, gfx938), forcing the torch complex-rope chain and
    # torch rsqrt norm every step. Bitwise-verified on gfx938
    # (tmp_fetch/parity_sparse_triton.py). HCU_DSV41_SPARSE_TRITON=0
    # restores the torch fallbacks.
    try:
        from hcu_adapter.features.dsv4 import dsv4_sparse_triton_gates

        dsv4_sparse_triton_gates.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: sparse triton-gate needle not installed",
            exc_info=True,
        )

    # DSV4/GLM-5.2-DSA fused q/k norm+rope: upstream elementwise module
    # routes on module-level `_is_hip`/`_is_xpu` constants frozen at import.
    # Under the is_cuda seam both read False -> the CUDA JIT arm
    # (main_norm_rope.cuh) compiles for gfx938 and fails (__shfl_xor_sync,
    # __syncwarp, HIP vector decompose). The HIP arms are exactly what HCU
    # wants: plain rope falls to the upstream triton kernel; q/k norm_rope
    # call torch.ops.sgl_kernel.dsv4_* which the DTK AOT wheel ships as real
    # kernels. Flip the constants on the module (import-time rebind; the
    # loader may not even have imported it yet, so also pre-import to be
    # sure). The FP4-indexer arm keeps its loud CUDA-only raise (we don't
    # enable enable_deepseek_v4_fp4_indexer).
    try:
        # 替代档: sgl_kernel dsv4_*（DTK AOT；本处仅 rebind 模块常量）
        import sglang.kernels.ops.attention.dsv4.elementwise as _dsew
        from sglang.srt.utils import common as _seam_common  # noqa: F401

        _dsew._is_hip = True
        _dsew._is_xpu = False
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: DSV4 norm_rope elementwise routed to HIP arms"
            " (triton rope + dsv4_* AOT kernels)"
        )
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: dsv4 elementwise bridge not installed",
            exc_info=True,
        )
