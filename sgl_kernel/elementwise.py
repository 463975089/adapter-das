# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

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
from ._native import native_op

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


def _check_aot_alignment(input: torch.Tensor) -> None:
    """16-byte tail alignment, AOT contract (upstream elementwise.py:258-260).

    The CUDA kernel vectorizes to 16-byte chunks (vec_size = 16/itemsize,
    activation.cu dispatch); the upstream python wrapper rejects misaligned
    tails before launch -- mirrored here so bad shapes fail loudly on every
    backend, not as silent garbage on GPU."""
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")


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
    _check_aot_alignment(input)
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
    _check_aot_alignment(input)
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


# ---------------------------------------------------------------------------
# MLA concat copies + pinned H2D (upstream csrc/elementwise/concat_mla.cu,
# copy.cu; python wrappers at upstream elementwise.py:419-440)
# ---------------------------------------------------------------------------

_MLA_NUM_HEADS = 128
_MLA_NOPE_DIM = 128
_MLA_ROPE_DIM = 64
_MLA_K_DIM = _MLA_NOPE_DIM + _MLA_ROPE_DIM  # 192
_MLA_ABSORB_A_DIM = 512
_MLA_ABSORB_B_DIM = 64
_COPY_NO_CE_SIZES = (32, 64, 72)


def _check_mla_common(t, name, shape2):
    """Shared host checks mirroring concat_mla.cu check_tensor: 3D,
    bf16, last-dim unit stride, 16-byte pointer alignment. dim0/dim1 are
    checked by the callers (upstream raises RuntimeError via TORCH_CHECK)."""
    if t.dim() != 3:
        raise RuntimeError(f"{name} must be 3-D, got {t.dim()}")
    if t.shape[2] != shape2:
        raise RuntimeError(f"{name} last dim must be {shape2}, got {t.shape[2]}")
    if t.stride(2) != 1:
        raise RuntimeError(f"{name} must have unit last-dim stride")
    if t.dtype != torch.bfloat16:
        raise RuntimeError(f"{name} must be bf16, got {t.dtype}")
    if t.data_ptr() % 16 != 0:
        raise RuntimeError(f"{name} pointer must be 16-byte aligned")


if HAS_TRITON:

    @triton.jit
    def _concat_mla_k_kernel(
        KN,
        KR,
        K,
        nope_s0,
        nope_s1,
        k_s0,
        k_s1,
        rope_s0,
        NOPE: tl.constexpr,
        ROPE: tl.constexpr,
    ):
        # One program per (token, head): nope row copy + the broadcast
        # k_rope tail (upstream: k_rope is (T, 1, 64), one rope vector
        # shared by all 128 heads).
        t = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1).to(tl.int64)
        n_off = tl.arange(0, NOPE)
        nope = tl.load(KN + t * nope_s0 + h * nope_s1 + n_off)
        r_off = tl.arange(0, ROPE)
        rope = tl.load(KR + t * rope_s0 + r_off)
        base = K + t * k_s0 + h * k_s1
        tl.store(base + n_off, nope)
        tl.store(base + NOPE + r_off, rope)

    @triton.jit
    def _concat_mla_absorb_q_kernel(
        A,
        B,
        OUT,
        a_s0,
        a_s1,
        b_s0,
        b_s1,
        out_s0,
        out_s1,
        A_DIM: tl.constexpr,
        B_DIM: tl.constexpr,
    ):
        # One program per (row0, row1): a[...:512] then b[...:64] into out.
        i0 = tl.program_id(0).to(tl.int64)
        i1 = tl.program_id(1).to(tl.int64)
        a_off = tl.arange(0, A_DIM)
        b_off = tl.arange(0, B_DIM)
        a = tl.load(A + i0 * a_s0 + i1 * a_s1 + a_off)
        b = tl.load(B + i0 * b_s0 + i1 * b_s1 + b_off)
        tl.store(OUT + i0 * out_s0 + i1 * out_s1 + a_off, a)
        tl.store(OUT + i0 * out_s0 + i1 * out_s1 + A_DIM + b_off, b)


def concat_mla_k(k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor):
    """In-place MLA k concat (upstream concat_mla.cu:87-110):

        k[T, 128, 192] = [k_nope[T, 128, 128] | k_rope[T, 1, 64] * heads]

    k_rope is ONE rope vector per token, broadcast across all 128 heads.
    Contract checks mirror the upstream TORCH_CHECKs (3D, bf16, unit
    last-dim stride, 16B alignment); CUDA runs a triton row-copy tier,
    everything else the equivalent strided torch copies. In-place on k.
    """
    num_tokens = k.shape[0]
    _check_mla_common(k, "k", _MLA_K_DIM)
    _check_mla_common(k_nope, "k_nope", _MLA_NOPE_DIM)
    _check_mla_common(k_rope, "k_rope", _MLA_ROPE_DIM)
    if k.shape[1] != _MLA_NUM_HEADS or k_nope.shape[1] != _MLA_NUM_HEADS:
        raise RuntimeError("k/k_nope head dim must be 128")
    if k_rope.shape[1] != 1:
        raise RuntimeError("k_rope head dim must be 1")
    if k.shape[0] != k_nope.shape[0] or k.shape[0] != k_rope.shape[0]:
        raise RuntimeError("token counts must match")

    # cpp tier (setup_hip.py wheel): concat_mla_k_hcu.cu — uint4 rows need
    # 16-byte-aligned row strides; wider strides stay on the triton tier.
    native = native_op("concat_mla_k")
    if (
        native is not None
        and k.is_cuda
        and k.stride(0) % 8 == 0
        and k.stride(1) % 8 == 0
        and k_nope.stride(0) % 8 == 0
        and k_nope.stride(1) % 8 == 0
        and k_rope.stride(0) % 8 == 0
    ):
        native(k, k_nope, k_rope)
        return

    if HAS_TRITON and k.is_cuda and k_nope.is_cuda and k_rope.is_cuda:
        _concat_mla_k_kernel[(num_tokens, _MLA_NUM_HEADS)](
            k_nope,
            k_rope,
            k,
            k_nope.stride(0),
            k_nope.stride(1),
            k.stride(0),
            k.stride(1),
            k_rope.stride(0),
            NOPE=_MLA_NOPE_DIM,
            ROPE=_MLA_ROPE_DIM,
            num_warps=4,
        )
        return
    k[..., :_MLA_NOPE_DIM].copy_(k_nope)
    k[..., _MLA_NOPE_DIM:].copy_(k_rope)  # (T,1,64) broadcasts over heads


def concat_mla_absorb_q(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """MLA absorb-mode q concat (upstream concat_mla.cu:120-190):

        a[B0, B1, 512], b[B0, B1, 64] -> out[B0, B1, 576]

    The python wrapper allocates and returns out (upstream
    elementwise.py:431-440). Contract checks mirror the upstream
    TORCH_CHECKs; CUDA runs a triton copy tier, elsewhere torch.cat.
    """
    _check_mla_common(a, "a", _MLA_ABSORB_A_DIM)
    _check_mla_common(b, "b", _MLA_ABSORB_B_DIM)
    if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]:
        raise RuntimeError("a/b leading dims must match")
    batch_dims = a.shape[:-1]
    out = torch.empty(
        (*batch_dims, _MLA_ABSORB_A_DIM + _MLA_ABSORB_B_DIM),
        device=a.device,
        dtype=a.dtype,
    )
    # cpp tier (setup_hip.py wheel): concat_mla_absorb_q_hcu.cu — the
    # freshly allocated out is always contiguous/aligned. The native
    # kernel additionally requires a/b ROW strides in multiples of 8
    # bf16 elements (uint4 vectorization, check_concat_tensor); a
    # base-aligned but row-strided view like a=(2,3,513)[..., :512]
    # passes _check_mla_common yet would TORCH_CHECK-fail inside the
    # native lib — guard it here and fall through to the strided
    # triton/torch tiers instead (review #80).
    native = native_op("concat_mla_absorb_q")
    if (
        native is not None
        and a.is_cuda
        and a.stride(0) % 8 == 0
        and a.stride(1) % 8 == 0
        and b.stride(0) % 8 == 0
        and b.stride(1) % 8 == 0
    ):
        native(a, b, out)
        return out
    if HAS_TRITON and a.is_cuda and b.is_cuda and out.is_contiguous():
        _concat_mla_absorb_q_kernel[(a.shape[0], a.shape[1])](
            a,
            b,
            out,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            A_DIM=_MLA_ABSORB_A_DIM,
            B_DIM=_MLA_ABSORB_B_DIM,
            num_warps=4,
        )
        return out
    return torch.cat([a, b], dim=-1)


def copy_to_gpu_no_ce(input: torch.Tensor, output: torch.Tensor):
    """Pinned int32 H2D copy (upstream copy.cu: N in {32, 64, 72}).

    Upstream smuggles the values through kernel launch PARAMETERS to
    bypass the copy engine (a CUDA-runtime-only trick). This tier performs
    the identical copy via ``output.copy_(input)`` — numerically exact,
    but the transfer uses the regular H2D path; the mechanism difference
    is pinned in BACKEND_CATALOG (emu tier, no triton equivalent).
    Contract checks mirror the upstream TORCH_CHECKs exactly.
    """
    n = input.numel()
    if n not in _COPY_NO_CE_SIZES:
        raise RuntimeError(f"unexpected N {n} (upstream whitelist: 32/64/72)")
    if input.dim() != 1:
        raise RuntimeError("input must be 1-D")
    if not input.is_contiguous():
        raise RuntimeError("input must be contiguous")
    if input.dtype != torch.int32:
        raise RuntimeError("input dtype must be int32")
    if output.dim() != 1 or output.numel() != n:
        raise RuntimeError("output must be 1-D with matching numel")
    if output.dtype != torch.int32:
        raise RuntimeError("output dtype must be int32")
    if not input.is_cpu:
        raise RuntimeError("input must be a CPU tensor")
    if not output.is_cuda:
        raise RuntimeError("output must be a CUDA tensor")
    # cpp tier (setup_hip.py wheel): the REAL no-CE path — values ride the
    # kernel launch parameters (copy.cu InputArray trick, works on HIP's
    # runtime too), bypassing the copy engine entirely.
    native = native_op("copy_to_gpu_no_ce")
    if native is not None:
        native(input, output)
        return
    output.copy_(input)
