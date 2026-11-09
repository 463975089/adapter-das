# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm family: rmsnorm / fused_add_rmsnorm / gemma variants.

Mirrors the official sgl_kernel signatures consumed by sglang's layernorm
layers (and the future `sglang.kernels.ops.layernorm` AOT backend):

    rmsnorm(input, weight, eps=1e-6, out=None) -> Tensor
    rmsscale: input / RMS(input) * weight
    fused_add_rmsnorm(input, residual, weight, eps=1e-6) -> None   (inplace)
        step1: residual += input
        step2: input = rmsnorm(residual) * weight, written back into input
    gemma_*: same but multiplies by (1 + weight) (Gemma-style norm)

All math is accumulated in fp32; bf16/fp16/fp32 inputs supported.
"""

from __future__ import annotations

from typing import Optional

import torch

from .compat import HAS_TRITON, check_cuda_tensor, next_power_of_2

_RMS_IMPL = None  # cached backend choice: "triton" | "aiter" | "lightop" | "torch"
_FUSED_ADD_IMPL = None      # cached fused-add tier: "triton" | "lightop" | "torch"
_LIGHTOP_FUSED_ADD = None   # resolved (fused_add_rms_norm, gemma_fused_add_rmsnorm)


def _resolve_rms_impl():
    """rms_norm backend tier (config file > env > default triton).

    Applies to the plain rmsnorm() entry only; the fused-add family has
    its own tier (HCU_FUSED_ADD_RMS_IMPL)."""
    global _RMS_IMPL
    if _RMS_IMPL is None:
        _RMS_IMPL = "triton"
        try:
            from hcu_adapter import hcu_env

            impl, _src = hcu_env.resolve("HCU_RMS_NORM_IMPL")
            if impl in ("triton", "aiter", "lightop", "torch"):
                _RMS_IMPL = impl
        except Exception:
            import traceback

            traceback.print_exc()
    return _RMS_IMPL


def _resolve_fused_add_impl():
    """fused_add_rmsnorm backend tier (config file > env > default triton).

    lightop 0.6.0 ships in-place fused_add_rms_norm(x, residual, weight,
    variance_epsilon, limit=-1.0) plus the gemma_fused_add_rmsnorm variant —
    Vendor numerics require their own parity checks: the triton path below
    uses the unrounded fp32 sum for variance and the stored residual for the
    numerator. auto = triton; lightop is opt-in until it passes the contract
    + graph-replay matrix (same policy as HCU_SILU_MUL_IMPL)."""
    global _FUSED_ADD_IMPL, _LIGHTOP_FUSED_ADD
    if _FUSED_ADD_IMPL is None:
        _FUSED_ADD_IMPL = "triton"
        try:
            from hcu_adapter import hcu_env

            impl, _src = hcu_env.resolve("HCU_FUSED_ADD_RMS_IMPL")
            if impl == "lightop":
                from lightop import fused_add_rms_norm, gemma_fused_add_rmsnorm

                _LIGHTOP_FUSED_ADD = (fused_add_rms_norm, gemma_fused_add_rmsnorm)
                _FUSED_ADD_IMPL = "lightop"
            elif impl in ("triton", "torch"):
                _FUSED_ADD_IMPL = impl
        except Exception:
            # visibility beats silence: a silent vendor-probe failure reads
            # as "auto mode" and hides a real wheel/env problem.
            import traceback

            traceback.print_exc()
    return _FUSED_ADD_IMPL


def _try_lightop_fused_add(input, residual, weight, eps, gemma):
    """Vendor-tier dispatch; False = caller continues on triton/native.

    Guarded to the shapes the wheel's launcher documents; a raising vendor
    call is one-strike: traceback printed once, the tier disabled for the
    process, and THIS call falls back — an opt-in knob must never crash a
    serving process that would otherwise have run on triton."""
    global _LIGHTOP_FUSED_ADD
    if (
        _LIGHTOP_FUSED_ADD is None
        or not input.is_cuda
        or input.dtype not in (torch.bfloat16, torch.float16)
        or not (input.is_contiguous() and residual.is_contiguous())
        or input.shape != residual.shape
        or weight.dim() != 1
        or weight.shape[0] != input.shape[-1]
    ):
        return False
    fn = _LIGHTOP_FUSED_ADD[1] if gemma else _LIGHTOP_FUSED_ADD[0]
    try:
        # in-place on both, matching the sgl_kernel contract; the vendor
        # kernel is row-wise, so flatten leading dims to the canonical 2D
        fn(input.view(-1, input.shape[-1]),
           residual.view(-1, residual.shape[-1]),
           weight, eps)
        return True
    except Exception:
        import traceback

        traceback.print_exc()
        print("hcu: lightop fused_add_rmsnorm raised once — tier disabled, "
              "falling back to triton for this process")
        _LIGHTOP_FUSED_ADD = None
        return False


def _aiter_rmsnorm(input, weight, eps):
    """aiter.rmsnorm_forward bridge: (x, w, eps, training) -> (out, ..).

    Vendor kernel takes 2D rows; arbitrary leading dims are flattened and
    restored. Returns a fresh tensor (caller copies into the out buffer)."""
    import aiter

    shape = input.shape
    res = aiter.rmsnorm_forward(
        input.contiguous().view(-1, shape[-1]), weight, eps, False
    )
    out = res[0] if isinstance(res, tuple) else res
    return out.view(shape)


def _lightop_rmsnorm(input, weight, eps, out):
    """Use the public BF16 provider while preserving the caller's output buffer."""
    import lightop

    check_cuda_tensor(input, weight, out)
    if input.device != weight.device or input.device != out.device:
        raise ValueError("LightOp RMSNorm tensors must share one device")
    if not input.numel():
        return out
    x = input.contiguous().view(-1, input.shape[-1])
    y = out.contiguous().view_as(x)
    lightop.rms_norm(x, weight.contiguous(), eps, out=y)
    if y.data_ptr() != out.data_ptr():
        out.copy_(y.view_as(out))
    return out


if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _rmsnorm_kernel(
        X,
        W,
        Y,
        x_stride,
        y_stride,
        N,
        eps,
        GEMMA: tl.constexpr,  # False: y = x*rstd*w ; True: y = x*rstd*(1+w)
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        x_row = X + row * x_stride
        y_row = Y + row * y_stride
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            acc += x * x
        var = tl.sum(acc, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        d2f = 1.0 if GEMMA else 0.0
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            y = x * rstd * (w + d2f)
            tl.store(y_row + cols, y.to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def _fused_add_rmsnorm_kernel(
        X,  # inout: becomes normed output
        RES,  # inout: becomes residual + input
        W,
        stride,
        N,
        eps,
        GEMMA: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        x_row = X + row * stride
        res_row = RES + row * stride
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            r = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
            t = x + r
            acc += t * t
        var = tl.sum(acc, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        d2f = 1.0 if GEMMA else 0.0
        for off in range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < N
            # Default community fused-add uses the FP32 sum for variance but
            # the activation-dtype residual for the normalization numerator.
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            r = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
            t = x + r
            stored = t.to(RES.dtype.element_ty)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            y = stored.to(tl.float32) * rstd * (w + d2f)
            tl.store(res_row + cols, stored, mask=mask)
            tl.store(x_row + cols, y.to(X.dtype.element_ty), mask=mask)


def _rmsnorm_native(
    input: torch.Tensor, weight: torch.Tensor, eps: float, gemma: bool, out=None
):
    x = input.float()
    scale = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    w = weight.float() + (1.0 if gemma else 0.0)
    result = (scale * w).to(input.dtype if out is None else out.dtype)
    if out is None:
        return result
    out.copy_(result)
    return out


def _fused_add_rmsnorm_native(input, residual, weight, eps: float, gemma: bool):
    # Match the two-pass fused kernel: its variance consumes the fp32 sum,
    # while its numerator reloads the residual after activation-dtype storage.
    total = input.float() + residual.float()
    residual.copy_(total)
    x = residual.float()
    scale = x * torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + eps)
    w = weight.float() + (1.0 if gemma else 0.0)
    input.copy_((scale * w).to(input.dtype))


def _do_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    elif tuple(out.shape) != tuple(input.shape):
        raise ValueError(f"out shape {tuple(out.shape)} != {tuple(input.shape)}")

    impl = _resolve_rms_impl()
    if gemma and impl != "triton":
        impl = "triton"  # vendor backend covers the plain norm only
    if impl == "torch" or not input.is_cuda:
        return _rmsnorm_native(input, weight, eps, gemma, out)
    if impl == "lightop":
        # Only the BF16 plain-norm contract has been qualified with this provider.
        # Other dtypes and Gemma retain the existing Triton implementation.
        if input.dtype == weight.dtype == out.dtype == torch.bfloat16:
            return _lightop_rmsnorm(input, weight, eps, out)
        impl = "triton"
    if impl == "aiter":
        if input.dtype in (torch.bfloat16, torch.float16):
            out.copy_(_aiter_rmsnorm(input, weight, eps))
            return out
        impl = "triton"  # vendor bf16/fp16 only; fall through

    use_triton = (
        HAS_TRITON
        and input.is_cuda
        and weight.is_cuda
        and input.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and input.shape[-1] > 0
    )
    if not use_triton:
        return _rmsnorm_native(input, weight, eps, gemma, out)

    check_cuda_tensor(input, weight, out)
    x_2d = input.contiguous().view(-1, input.shape[-1])
    y_2d = out.contiguous().view(-1, out.shape[-1])
    had_noncontig_out = y_2d.data_ptr() != out.data_ptr()
    n = x_2d.shape[-1]
    block = min(next_power_of_2(n), 8192)
    num_warps = 16 if block >= 4096 else (8 if block >= 1024 else 4)
    _rmsnorm_kernel[(x_2d.shape[0],)](
        x_2d,
        weight,
        y_2d,
        x_2d.stride(0),
        y_2d.stride(0),
        n,
        eps,
        GEMMA=gemma,
        BLOCK=block,
        num_warps=num_warps,
    )
    if had_noncontig_out:
        out.copy_(y_2d.view(out.shape))
    return out


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """input / RMS(input) * weight.enable_pdl is accepted for API parity and
    ignored (no PDL on HCU)."""
    return _do_rmsnorm(input, weight, eps, False, out)


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Gemma-style: input / RMS(input) * (1 + weight)."""
    return _do_rmsnorm(input, weight, eps, True, out)


def _do_fused_add(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
) -> None:
    impl = _resolve_fused_add_impl()
    if impl == "lightop" and _try_lightop_fused_add(input, residual, weight, eps, gemma):
        return
    use_triton = (
        impl != "torch"
        and HAS_TRITON
        and input.is_cuda
        and residual.is_cuda
        and input.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and input.is_contiguous()
        and residual.is_contiguous()
        and input.shape == residual.shape
    )
    if not use_triton:
        _fused_add_rmsnorm_native(input, residual, weight, eps, gemma)
        return

    check_cuda_tensor(input, residual, weight)
    x_2d = input.view(-1, input.shape[-1])
    res_2d = residual.view(-1, residual.shape[-1])
    n = x_2d.shape[-1]
    block = min(next_power_of_2(n), 8192)
    num_warps = 16 if block >= 4096 else (8 if block >= 1024 else 4)
    _fused_add_rmsnorm_kernel[(x_2d.shape[0],)](
        x_2d,
        res_2d,
        weight,
        x_2d.stride(0),
        n,
        eps,
        GEMMA=gemma,
        BLOCK=block,
        num_warps=num_warps,
    )


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    """residual += input; input = rmsnorm(residual) * weight (both in-place)."""
    _do_fused_add(input, residual, weight, eps, False)


def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    """residual += input; input = rmsnorm(residual) * (1 + weight)."""
    _do_fused_add(input, residual, weight, eps, True)
