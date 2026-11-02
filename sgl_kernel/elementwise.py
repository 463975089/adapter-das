"""Gated activation kernels: silu_and_mul / gelu_and_mul / gelu_tanh_and_mul.

Mirrors the official sgl_kernel API so `sglang.kernels.ops.activation`
`forward_aot` (`getattr(sgl_kernel, kernel_attr)(input, out)`) resolves here.

Semantics (out optional, allocated when omitted):
    input  (..., 2d) -> out (..., d)
    out[...] = act(input[..., :d]) * input[..., d:]
gelu_quick (HIP-only export upstream):
    out = input * sigmoid(1.702 * input), same shape as input.
"""

from __future__ import annotations

from typing import Optional

import torch

from .compat import HAS_ERF, HAS_TRITON, check_cuda_tensor, next_power_of_2

_LIGHTOP_IMPL = None  # resolved once: None=probe-pending, (~)=unavailable
_IMPL_CHOICE = None   # cached backend choice: "lightop" | "torch" | None


def _resolve_impl():
    """Backend choice for gated activations (config file > env > default).

    Cached once -- HCU_SILU_MUL_IMPL / ops.silu_and_mul in
    hcu_adapter.yaml both land here."""
    global _IMPL_CHOICE, _LIGHTOP_IMPL
    if _IMPL_CHOICE is None:
        try:
            from hcu_adapter import hcu_env

            impl, _src = hcu_env.resolve("HCU_SILU_MUL_IMPL")
            # auto = triton (default); lightop/torch only on explicit
            # selection until lightop passes the graph-replay matrix and
            # torch stays the reference/debug tier.
            if impl == "lightop":
                from lightop import fuse_silu_and_mul

                _LIGHTOP_IMPL = fuse_silu_and_mul
                _IMPL_CHOICE = "lightop"
            elif impl == "torch":
                _IMPL_CHOICE = "torch"
            else:
                _IMPL_CHOICE = "triton"
        except Exception:
            # visibility beats silence: a silent vendor-probe failure reads
            # as "auto mode" and hides a real wheel/env problem.
            import traceback

            traceback.print_exc()
            _IMPL_CHOICE = "triton"
    return _IMPL_CHOICE

if HAS_TRITON:
    from .compat import _tl_erf

    import triton
    import triton.language as tl

    _SQRT_1_OVER_2 = 0.7071067811865476
    _INV_SQRT_2PI = 0.3989422804014327

    @triton.jit
    def _gated_act_kernel(
        X,  # *input, row-major, (M, 2D)
        Y,  # *output, row-major, (M, D)
        x_stride,
        y_stride,
        D,
        ACT: tl.constexpr,  # 0 silu, 1 gelu(erf), 2 gelu-tanh
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        x_base = X + row * x_stride
        gate = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_base + D + cols, mask=mask, other=0.0).to(tl.float32)
        if ACT == 0:
            g = gate * tl.sigmoid(gate)
        elif ACT == 1:
            g = 0.5 * gate * (1.0 + _tl_erf(gate * 0.7071067811865476))
        else:
            inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
            # 0.5*(1+tanh z) = sigmoid(2z) = 1/(1+exp(-2z)); gelu_tanh = x*sigmoid(2z)
            g = gate * (1.0 / (1.0 + tl.exp(-2.0 * inner)))
        tl.store(
            Y + row * y_stride + cols,
            (g * up).to(Y.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _gelu_quick_kernel(
        X,  # (numel,)
        Y,
        numel,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * tl.sigmoid(1.702 * x)
        tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def _silu_mul_clamp_kernel(
        X,  # *input, row-major, (M, 2D)
        Y,  # *output, row-major, (M, D)
        x_stride,
        y_stride,
        D,
        limit,
        BLOCK: tl.constexpr,
    ):
        # DSV4/GLM swiglu-clamp: gate caps its max only, up clamps both
        # sides (glm5_next.swiglu_clamped reference semantics).
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        x_base = X + row * x_stride
        gate = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_base + D + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
        g = gate * tl.sigmoid(gate)
        tl.store(
            Y + row * y_stride + cols,
            (g * up).to(Y.dtype.element_ty),
            mask=mask,
        )


def _silu_native(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _gelu_native(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x)


def _gelu_tanh_native(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def _gated_mul_impl(
    input: torch.Tensor,
    out: Optional[torch.Tensor],
    act: str,
    force_native: bool = False,
) -> torch.Tensor:
    """Shared gate path for silu_and_mul / gelu_and_mul / gelu_tanh_and_mul."""
    d = input.shape[-1] // 2
    if out is None:
        out = torch.empty(
            input.shape[:-1] + (d,), device=input.device, dtype=input.dtype
        )
    else:
        if tuple(out.shape) != tuple(input.shape[:-1] + (d,)):
            raise ValueError(
                f"out shape {tuple(out.shape)} != {tuple(input.shape[:-1] + (d,))}"
            )

    use_triton = (
        not force_native
        and HAS_TRITON
        and input.is_cuda
        and input.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and input.shape[-1] % 2 == 0
    )
    if act == "gelu" and not HAS_ERF:
        # erf unavailable in this triton build; native path is exact.
        use_triton = False
    if not use_triton:
        gate, up = input[..., :d], input[..., d:]
        act_fn = {"silu": _silu_native, "gelu": _gelu_native, "gelu_tanh": _gelu_tanh_native}[act]
        result = act_fn(gate.float()) * up.float()
        out.copy_(result.to(out.dtype))
        return out

    check_cuda_tensor(input, out)
    x_2d = input.reshape(-1, input.shape[-1])
    y_2d = out.reshape(-1, d)
    m, n = x_2d.shape[0], d
    act_id = {"silu": 0, "gelu": 1, "gelu_tanh": 2}[act]
    block = next_power_of_2(n)
    # Keep the block within a sane register budget; the row loop handles
    # any remaining columns via the same grid size (columns masked off).
    num_warps = 8 if block >= 2048 else (4 if block >= 512 else 2)
    _gated_act_kernel[(m,)](
        x_2d,
        y_2d,
        x_2d.stride(0),
        y_2d.stride(0),
        n,
        ACT=act_id,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


def silu_and_mul(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """out = silu(input[..., :d]) * input[..., d:]"""
    if input.is_cuda:
        choice = _resolve_impl()
        if choice == "torch":
            return _gated_mul_impl(input, out, "silu", force_native=True)
        if choice == "lightop" and _LIGHTOP_IMPL is not None:
            if out is None:
                out = torch.empty(
                    input.shape[:-1] + (input.shape[-1] // 2,),
                    device=input.device, dtype=input.dtype,
                )
            res = _LIGHTOP_IMPL(input, out)  # lightop sig: (input, output)
            return out if res is None else res
    return _gated_mul_impl(input, out, "silu")


def gelu_and_mul(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """out = gelu(input[..., :d]) * input[..., d:] (erf-exact gelu)"""
    return _gated_mul_impl(input, out, "gelu")


def gelu_tanh_and_mul(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """out = gelu_tanh(input[..., :d]) * input[..., d:]"""
    return _gated_mul_impl(input, out, "gelu_tanh")


def gelu_quick(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """Quick-GELU: out = input * sigmoid(1.702 * input)"""
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError(f"out shape {out.shape} != input shape {input.shape}")

    use_triton = (
        HAS_TRITON
        and HAS_ERF
        and input.is_cuda
        and input.dtype in (torch.bfloat16, torch.float16, torch.float32)
    )
    if not use_triton:
        out.copy_((input.float() * torch.sigmoid(1.702 * input.float())).to(out.dtype))
        return out

    check_cuda_tensor(input, out)
    numel = input.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK"]),)
    _gelu_quick_kernel[grid](
        input.reshape(-1),
        out.reshape(-1),
        numel,
        BLOCK=1024,
        num_warps=4,
    )
    return out


def silu_and_mul_clamp(
    input: torch.Tensor, output: torch.Tensor, limit: float
) -> None:
    """DSV4/GLM swiglu-clamp gated activation (upstream CUDA/XPU-only op).

        gate, up = input[..., :d], input[..., d:]
        out = silu(clamp_max(gate, limit)) * clamp(up, -limit, limit)

    The gate half caps its max only (OAI/GLM convention, matching
    glm5_next.swiglu_clamped); the up half clamps both sides. Destination-
    passing like the upstream dsv4 op; fp32 math, CUDA-graph capture safe.
    """
    d = input.shape[-1] // 2
    if tuple(output.shape) != tuple(input.shape[:-1] + (d,)):
        raise ValueError(
            f"output shape {tuple(output.shape)} != "
            f"{tuple(input.shape[:-1] + (d,))}"
        )
    if (
        HAS_TRITON
        and input.is_cuda
        and input.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and input.shape[-1] % 2 == 0
    ):
        check_cuda_tensor(input, output)
        x_2d = input.reshape(-1, input.shape[-1])
        y_2d = output.reshape(-1, d)
        block = next_power_of_2(d)
        num_warps = 8 if block >= 2048 else (4 if block >= 512 else 2)
        _silu_mul_clamp_kernel[(x_2d.shape[0],)](
            x_2d,
            y_2d,
            x_2d.stride(0),
            y_2d.stride(0),
            d,
            float(limit),
            BLOCK=block,
            num_warps=num_warps,
        )
        return
    gate, up = input[..., :d], input[..., d:]
    gate = gate.float().clamp(max=limit)
    up = up.float().clamp(min=-limit, max=limit)
    output.copy_((_silu_native(gate) * up).to(output.dtype))
