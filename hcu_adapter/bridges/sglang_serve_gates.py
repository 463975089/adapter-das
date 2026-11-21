# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""serve-time gates (fa4/flashinfer registry, breakable prefill graph, QKV-A eligibility, debug probes, model env, MoE tile, MLA bmm, R-09 aiter quant/AR-norm)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
import os


def install(_he=None) -> None:
    # CUDA-only attention backends (fa4 = CUTLASS cute DSL SM90a;
    # flashinfer = is_flashinfer_available() False yet unguarded names):
    # fail at the backend-registry seam (constructor-time ImportError) so
    # upstream attention-kit tests skip "backend not available" instead of
    # dying mid-construction/mid-forward.
    try:
        from hcu_adapter import attn_backend_gate

        attn_backend_gate.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: attention-backend registry gate not installed",
            exc_info=True,
        )
    # Breakable prefill CUDA graph (sglang main): its runtime probes
    # stream-capture status through cuda-python, which needs a CUDA
    # driver HCU does not have. #149 (2026-11-19): prefill_bcg_gate now
    # rebinds the probe to the HIP runtime stream query, so breakable
    # prefill graphs are PRESERVED instead of routed to the eager
    # runner; decode FULL graphs were never affected.
    try:
        from hcu_adapter import prefill_bcg_gate

        prefill_bcg_gate.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: prefill BCG gate not installed",
            exc_info=True,
        )
    # DeepSeek-V3 fused QKV-A GEMM min-latency path (sglang main): its
    # eligibility predicate sees the seam's CUDA answer plus sm 93 and
    # routes decode through a TensorRT-LLM Hopper kernel whose body is
    # __CUDA_ARCH__ >= 900 only and which asks for ~192 KB dynamic LDS
    # against a 64 KB/block cap -- graph capture dies in
    # set_max_dynamic_smem (invalid argument). Eligibility -> False on
    # HIP, each layer keeps its own linear.
    try:
        from hcu_adapter.features.dsa import fused_a_gemm_gate

        fused_a_gemm_gate.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: fused A-GEMM eligibility gate not installed",
            exc_info=True,
        )
    # Debug tooling (default OFF): in-graph side-buffer probe for the
    # graph-vs-emu corruption hunts — see hcu_adapter/probe_side.py. Only
    # loads when HCU_SIDE_PROBE is set (replay|eager); never in production.
    if os.environ.get("HCU_SIDE_PROBE"):
        try:
            from hcu_adapter.infra import probe_side

            probe_side.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: side-buffer probe not installed", exc_info=True
            )
    # Debug tooling (default OFF): request-boundary state snapshot probe for
    # the R-01/R-06 cross-request leakage hunt — see
    # hcu_adapter/state_snapshot.py. Only loads when HCU_STATE_SNAPSHOT is set.
    if os.environ.get("HCU_STATE_SNAPSHOT"):
        try:
            from hcu_adapter.infra import state_snapshot

            state_snapshot.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: state snapshot probe not installed", exc_info=True
            )
    # Model-aware env defaults (R-08 B-tier, default on via HCU_MODEL_ENV):
    # DSA models get their two fusion escape hatches, fp16 GDN bottoms get
    # the conv-dtype fix — derived from model_config, setdefault only.
    try:
        from hcu_adapter.infra import model_env

        model_env.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: model-env defaults not installed", exc_info=True
        )
    # fused-MoE triton tile guard: HCU 64KB LDS vs the ~72KB NV default
    # config — clamp oversized tiles at the durable resolution seam.
    try:
        from hcu_adapter.infra import moe_config_guard

        moe_config_guard.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: moe config smem guard not installed", exc_info=True
        )
    # MLA-absorb fp8 bmm: upstream gemm.bmm_fp8 spec resolves to
    # fp8_utils:bmm_fp8 which only exists behind flashinfer import. Supply a
    # torch-native dequant-bmm (scales applied in the epilogue role).
    try:
        from hcu_adapter import bmm_fp8_bridge

        bmm_fp8_bridge.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: bmm_fp8 bridge not installed", exc_info=True
        )
    # R-09(2/3): aiter per-token fp8 quant vendor tier -- rowwise wrapper
    # NameError fix (HIP has no sgl_per_token_quant_fp8), group plain arm
    # 15.1x, scaled_fp8_quant natives via upstream's own aiter calls, and
    # the whole-row fused silu+quant arm (HCU_QUANT_IMPL, auto default).
    try:
        from hcu_adapter.features.aiter import quant_aiter

        quant_aiter.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: aiter fp8-quant tier not installed", exc_info=True
        )
    # R-09(1): aiter fused AR+RMSNorm -- upstream layernorm carries the
    # whole consumer chain dormant behind _use_aiter; re-point the
    # flashinfer entry its else-branch already calls at the aiter
    # GraphSafe fused kernels (HCU_AR_NORM_IMPL, auto default; full-TP-
    # group-only, per-call decline to stock).
    try:
        from hcu_adapter.features.aiter import ar_norm_aiter

        ar_norm_aiter.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: aiter fused AR+RMSNorm tier not installed",
            exc_info=True
        )
    # aiter JIT cache observability: the vendor wheel ships every op
    # prebuilt (fresh import ~seconds, zero compiles); surface the wheel
    # version and any runtime-built artifacts so they get harvested or the
    # wheel version pinned (see the module docstring for the harvest rule).
    try:
        from hcu_adapter.features.aiter import aiter_jit_cache

        aiter_jit_cache.report()
    except Exception:
        logging.getLogger("hcu.adapter").debug(
            "HCU adapter: aiter jit cache report skipped", exc_info=True
        )
