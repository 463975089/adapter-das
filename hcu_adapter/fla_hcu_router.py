# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Route KDA prefill / conv1d to the vendored sglang-das hcu triton kernels.

Parity vs the adapter emus (gfx938, T=2048/8192 serving shapes, gate:
``tmp_fetch/parity_flahcu.py`` / ``parity2_flahcu.py``):
- chunk_kda (prefill): out & final-state cos 0.99999, 473-1009x faster
  than the sequential emu (0.88ms vs 414ms @T=2048/H16/K128).
- causal_conv1d_fn: cos 0.999996, conv-state bit-exact, 16.4x.
- causal_conv1d_update: cos 0.999997, 4.3x.
- KDA decode: fork hcu recurrent == stock v0.5.21 (cos 0.9999, 1.0x) —
  decode is NOT rerouted; the stock kernel stays.

beta convention (validated against serving): the v0.5.21 kda_backend
hands extend() an ACTIVATED beta for GLM (gate_was_flat=False ->
beta_is_raw=False); the fork chunk_kda expects it activated too (its
intra kernels apply no sigmoid), so beta passes straight through unless
the caller explicitly sets beta_is_raw=True.

Env:
- HCU_KDA_PREFILL_IMPL = fork (hip default since 2026-10-18) | emu | stock
- HCU_CONV1D_IMPL      = fork (hip default since 2026-10-18) | emu | stock

Serving-gate history: the 2026-10-16 "fork flips Berlin deterministically"
verdict was RE-ATTRIBUTED 2026-10-18 — the flip carrier was the adapter's
custom triton AR (ar_ext.cu cross-block store-ordering bug; 160-arm A/B:
custom 60/100 vs non-custom 1/60), and the polluted-GPU-era fork flip data
was voided. Clean-route re-verification 2026-10-18 (post host GPU reset,
zero-env tp4 GLM-5.3 GPU4-7, aiter AR): bothfork arm anchors 4/4 +
marathon 0/20 + long-ctx 4972 tok correct; conv1d-only arm 0/20 + Tokyo +
long-ctx correct. Fork is now the hip default; =emu remains the opt-out.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("hcu.adapter")


def _install_kda_prefill(impl: str) -> None:
    if impl == "emu":
        from hcu_adapter import kda_prefill_emu

        kda_prefill_emu.install_patch()
        return
    if impl != "fork":
        return  # stock
    from hcu_adapter.fla_hcu.kda import chunk_kda
    from sglang.srt.layers.attention.linear.kernels.kda_triton import (
        TritonKDAKernel,
    )

    if getattr(TritonKDAKernel, "_hcu_fork_patched", False):
        return

    def extend(self, *args, **kwargs):
        if args:  # positional (q, k, v, g, beta) + keyword-only config
            q, k, v, g, beta = args[:5]
            kwargs = {**kwargs}
            kwargs.setdefault("q", q)
            kwargs.setdefault("k", k)
            kwargs.setdefault("v", v)
            kwargs.setdefault("g", g)
            kwargs.setdefault("beta", beta)
        beta = kwargs["beta"]
        if kwargs.get("beta_is_raw", False):
            beta = beta.float().sigmoid().to(beta.dtype)
        return chunk_kda(
            q=kwargs["q"],
            k=kwargs["k"],
            v=kwargs["v"],
            g=kwargs["g"],
            beta=beta,
            initial_state=kwargs["ssm_states"],
            initial_state_indices=kwargs["cache_indices"],
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=kwargs["query_start_loc"],
            A_log=kwargs.get("A_log"),
            dt_bias=kwargs.get("dt_bias"),
            lower_bound=kwargs.get("lower_bound"),
            return_intermediate_state=bool(kwargs.get("return_intermediate_states")),
        )

    TritonKDAKernel.extend = extend
    TritonKDAKernel._hcu_fork_patched = True
    logger.info("[HCU fla-hcu] KDA prefill -> fork chunk_kda triton")


def _install_conv1d(impl: str) -> None:
    if impl == "emu":
        from hcu_adapter import conv1d_fn_emu, conv1d_update_emu

        conv1d_update_emu.install_patch()
        conv1d_fn_emu.install_patch()
        return
    if impl != "fork":
        return  # stock (red gates kernels/test_conv1d_* on gfx938)
    from sglang.kernels.ops.mamba import causal_conv1d_triton as m
    from hcu_adapter.fla_hcu import conv1d_triton as fork

    m.causal_conv1d_fn = fork.causal_conv1d_fn
    m.causal_conv1d_update = fork.causal_conv1d_update
    logger.info("[HCU fla-hcu] causal_conv1d fn+update -> fork triton")


def install() -> None:
    hip = torch.version.hip is not None
    kda = os.environ.get("HCU_KDA_PREFILL_IMPL", "fork" if hip else "stock")
    conv = os.environ.get("HCU_CONV1D_IMPL", "fork" if hip else "stock")
    _install_kda_prefill(kda)
    _install_conv1d(conv)
    if kda == "fork" or conv == "fork":
        logger.info(
            "[HCU fla-hcu] fork kernels active (kda=%s, conv=%s; default "
            "since 2026-10-18, =emu opts out)",
            kda,
            conv,
        )
