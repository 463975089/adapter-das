# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HCU compatibility route for CuTeDSL GDN MTP verify kernels.

The CuTeDSL kernels currently fail in the HCU toolchain during lowering
(address-space pointer mismatch or unsupported l1c_evict_priority copies).
Route the public face to sglang's Triton recurrent implementation while
preserving the CuTe-facing tensor layout and ReplaySSM ring contract.
"""

from __future__ import annotations

import logging

import torch

from .registry import MECH_REBIND, register

LOG = logging.getLogger("hcu.adapter")


def _triton_gated_delta_rule_mtp(original, recurrent, *args, **kwargs):
    A_log, a, dt_bias = args[:3]
    q = kwargs.get("q")
    k = kwargs.get("k")
    v = kwargs.get("v")
    b = kwargs.get("b")
    state = kwargs.get("initial_state_source")
    slots = kwargs.get("initial_state_indices")
    if any(value is None for value in (q, k, v, b, state, slots)):
        return original(*args, **kwargs)
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or b.ndim != 3:
        return original(*args, **kwargs)

    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    # Both kernels address state as [slot, HV, V, K] (offset v*K+k).
    # Preserve the pool view, including its envelope slot stride.
    cu_seqlens = torch.arange(
        0, B * T + 1, step=T, dtype=torch.int32, device=q.device
    )
    recurrent_kwargs = {
        "A_log": A_log,
        "dt_bias": dt_bias,
        "softplus_beta": kwargs.get("softplus_beta", 1.0),
        "softplus_threshold": kwargs.get("softplus_threshold", 20.0),
        "q": q.reshape(1, B * T, H, K),
        "k": k.reshape(1, B * T, H, K),
        "v": v.reshape(1, B * T, HV, V),
        "b": b.reshape(B * T, HV),
        "a": a.reshape(B * T, HV),
        "initial_state_source": state,
        "initial_state_indices": slots,
        "cu_seqlens": cu_seqlens,
        "use_qk_l2norm_in_kernel": kwargs.get("use_qk_l2norm_in_kernel", True),
        "disable_state_update": kwargs.get("disable_state_update", False),
    }
    for name in (
        "cache_ring",
        "replayssm_rawv",
        "replayssm_rawk",
        "replayssm_g",
        "replayssm_beta",
    ):
        if name in kwargs:
            recurrent_kwargs[name] = kwargs[name]
    out = recurrent(**recurrent_kwargs)
    return out.reshape(B, T, HV, V)


def install():
    if not getattr(torch.version, "hip", None):
        return
    from sglang.kernels.ops.attention import cutedsl_gdn_mtp_ring as cute
    from sglang.kernels.ops.attention.fla import fused_sigmoid_gating_recurrent as fla

    original = cute.gated_delta_rule_mtp
    if getattr(original, "_hcu_triton_fallback", False):
        return

    def routed(*args, **kwargs):
        return _triton_gated_delta_rule_mtp(
            original, fla.fused_sigmoid_gating_delta_rule_update, *args, **kwargs
        )

    routed._hcu_triton_fallback = True
    routed.__wrapped__ = original
    cute.gated_delta_rule_mtp = routed
    register(
        "linear_attn_gdn",
        routed,
        target="sglang.kernels.ops.attention.cutedsl_gdn_mtp_ring.gated_delta_rule_mtp",
        mechanism=MECH_REBIND,
        detail="HIP CuTeDSL verify face -> upstream Triton recurrent compatibility route",
    )
    LOG.warning(
        "HCU adapter: CuTeDSL GDN MTP routed to Triton recurrent compatibility path"
    )
