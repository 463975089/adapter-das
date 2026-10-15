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
- HCU_KDA_PREFILL_IMPL = fork | emu (hip default) | stock
- HCU_CONV1D_IMPL      = fork | emu (hip default) | stock

2026-10-16 serving gate note: with both knobs =fork the server flips the
Berlin anchor (first token " a..." instead of 19808) deterministically
across clean boots, while every standalone parity gate above stays
green — including serving shapes (T=5 varlen) and strided conv-state
views. Bisect: emu boots are green, fork boots red, so the divergence
is fork-in-serving only; root cause needs a serving-side dump compare
(reuse the HCU_DUMP_LAYERS + replay_kda_l0 machinery). Until then fork
is opt-in only and emu keeps the correctness defaults.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("hcu.adapter")


def _install_kda_prefill(impl: str) -> None:
    if impl == "emu":
        from sgl_kernel._adapters import kda_prefill_emu

        kda_prefill_emu.install_patch()
        return
    if impl != "fork":
        return  # stock
    from sgl_kernel._adapters.fla_hcu.kda import chunk_kda
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
        from sgl_kernel._adapters import conv1d_fn_emu, conv1d_update_emu

        conv1d_update_emu.install_patch()
        conv1d_fn_emu.install_patch()
        return
    if impl != "fork":
        return  # stock (red gates kernels/test_conv1d_* on gfx938)
    from sglang.kernels.ops.mamba import causal_conv1d_triton as m
    from sgl_kernel._adapters.fla_hcu import conv1d_triton as fork

    m.causal_conv1d_fn = fork.causal_conv1d_fn
    m.causal_conv1d_update = fork.causal_conv1d_update
    logger.info("[HCU fla-hcu] causal_conv1d fn+update -> fork triton")


def install() -> None:
    hip = torch.version.hip is not None
    kda = os.environ.get("HCU_KDA_PREFILL_IMPL", "emu" if hip else "stock")
    conv = os.environ.get("HCU_CONV1D_IMPL", "emu" if hip else "stock")
    _install_kda_prefill(kda)
    _install_conv1d(conv)
    if kda == "fork" or conv == "fork":
        print(
            "[HCU fla-hcu] OPT-IN fork kernels active "
            f"(kda={kda}, conv={conv}) — serving gate RED (Berlin flip), "
            "use only for debugging",
            flush=True,
        )
