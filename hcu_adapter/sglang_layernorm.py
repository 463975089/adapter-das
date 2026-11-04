# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Route sglang v0.5.22 layernorm JIT faces to adapter-owned HCU kernels."""

from __future__ import annotations

import logging


def install() -> None:
    import torch

    if not getattr(torch.version, "hip", None):
        return

    from sgl_kernel.layernorm import fused_add_rmsnorm as _adapter_fused
    from sgl_kernel.layernorm import rmsnorm as _adapter_rmsnorm
    from sglang.kernels.ops.layernorm import norm as _norm

    if getattr(_norm.rmsnorm, "_hcu_adapter_route", False):
        return

    def _rmsnorm(input, weight, out=None, eps=1e-6):
        destination = input if out is None else out
        _adapter_rmsnorm(input, weight, eps=eps, out=destination)

    def _fused_add_rmsnorm(
        input, residual, weight, eps=1e-6, *, cast_x_before_out_mul=False
    ):
        if not cast_x_before_out_mul:
            _adapter_fused(input, residual, weight, eps=eps)
            return
        acc = input.float() + residual.float()
        residual.copy_(acc.to(residual.dtype))
        normed = acc * torch.rsqrt(acc.square().mean(-1, keepdim=True) + eps)
        input.copy_((normed.to(input.dtype) * weight).to(input.dtype))

    _rmsnorm._hcu_adapter_route = True
    _fused_add_rmsnorm._hcu_adapter_route = True
    _norm.rmsnorm = _rmsnorm
    _norm.fused_add_rmsnorm = _fused_add_rmsnorm

    # The bundled Aiter wheel uses return-value signatures while v0.5.22 calls
    # older out-parameter signatures. Keep the backend visible for route tests,
    # but execute it through the adapter's validated HCU implementation.
    import sglang.kernels.ops.layernorm as _ops

    def _rms_forward_aiter(self, input, weight, eps=1e-6, out=None, enable_pdl=None):
        destination = torch.empty_like(input) if out is None else out
        return _adapter_rmsnorm(input, weight, eps=eps, out=destination)

    def _fused_forward_aiter(
        self, input, residual, weight, eps=1e-6, enable_pdl=None
    ):
        return _adapter_fused(input, residual, weight, eps=eps)

    _ops.RMSNormOp.forward_aiter = _rms_forward_aiter
    _ops.FusedAddRMSNormOp.forward_aiter = _fused_forward_aiter
    logging.getLogger("hcu.adapter").info(
        "HCU adapter: sglang layernorm JIT faces routed to adapter kernels"
    )
