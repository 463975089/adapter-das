# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""vendor/triton routes (libdevice redirect, PDL strip, KDA router, fa3 vendor, GDN aiter, layernorm face)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging


def install(_he=None) -> None:
    # triton extra.cuda libdevice -> hip/OCML redirect (is_cuda semantics
    # picks the cuda libdevice package; hcu backend rejects __nv_* LLVM).
    try:
        from hcu_adapter.infra import triton_libdev

        triton_libdev.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: triton libdevice redirect not installed", exc_info=True
        )
    # strip triton-emitted griddepcontrol.* asm (capability>=9 enables PDL
    # upstream; gfx936 assembler rejects the instructions, and removing
    # them == PDL-off semantics).
    try:
        from hcu_adapter.infra import triton_pdl

        triton_pdl.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "hcu adapter: triton pdl stripper not installed", exc_info=True
        )
    # KDA prefill + causal_conv1d router (2026-10-16): vendored sglang-das
    # hcu triton kernels (hcu_adapter/fla_hcu/) — parity-gated
    # replacements for the gfx938-broken stock kernels (chunk_kda wrong Aqk;
    # conv1d update noise / fn dead initial-state arm). Kernel-level wins:
    # prefill 473-1009x, conv fn 16.4x, update 4.3x, cos >= 0.99999 vs the
    # emus — but the SERVING gate turned red (Berlin anchor flips
    # deterministically under =fork across clean boots while every standalone
    # gate stays green, incl. serving shapes and strided state views), so emu
    # keeps the defaults and fork is opt-in only until a serving-side dump
    # compare finds the divergence. HCU_KDA_PREFILL_IMPL / HCU_CONV1D_IMPL =
    # emu (default) | fork | stock.
    try:
        from hcu_adapter.features.gdn import fla_hcu_router

        fla_hcu_router.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: fla-hcu router not installed", exc_info=True
        )
    # fa3 attention vendor routing (DTK flash_attn wheel; emu fallback).
    # Needs sgl_kernel.flash_attn importable (below pulls it in).
    try:
        from hcu_adapter.bridges import sglang_fa_vendor as fa_vendor

        fa_vendor.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: fa vendor router not installed", exc_info=True
        )
    # GDN / gated-delta-rule linear attention aiter routing (opt-in via
    # HCU_LINEAR_ATTN_GDN=aiter; default keeps upstream triton fla).
    # Requires sglang importable (imports the fla modules it rebinds).
    try:
        from hcu_adapter.features.gdn import gdn_aiter

        gdn_aiter.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: gdn aiter router not installed", exc_info=True
        )
    try:
        from hcu_adapter.features.gdn import mamba_state_scatter

        mamba_state_scatter.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: Mamba state scatter grid route not installed",
            exc_info=True,
        )
    try:
        from hcu_adapter.features.gdn import gdn_cutedsl_fallback

        gdn_cutedsl_fallback.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: gdn CuTeDSL fallback not installed", exc_info=True
        )
    try:
        # Unconditional rebind: torch's lazy op namespace can pass an
        # import-time probe and still miss the native symbol at call time.
        from hcu_adapter.bridges import vllm as _vllm_bridge

        _vllm_bridge.install_rms_norm_bridge()
    except Exception:  # pragma: no cover - vllm absent: sglang falls back natively
        pass

    # v0.5.22's layernorm face otherwise compiles CUDA-source JIT kernels.
    try:
        from hcu_adapter.bridges import sglang_layernorm as _sgl_layernorm

        _sgl_layernorm.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: sglang layernorm route not installed", exc_info=True
        )

    try:
        import sgl_kernel as _sk

        if not hasattr(_sk, "apply_token_bitmask_inplace_cuda"):
            # sglang's xgrammar shim takes the HIP branch and imports this from
            # sgl_kernel; bridge it to the in-tree triton op (CUDA branch).
            # 替代档: 本文件 remap 到 triton（sgl_kernel 无 HIP bitmask 符号）
            from sglang.kernels.ops.grammar.bitmask_ops import (
                apply_token_bitmask_inplace_triton as _bitmask_triton,
            )

            _sk.apply_token_bitmask_inplace_cuda = _bitmask_triton
    except Exception:
        pass
