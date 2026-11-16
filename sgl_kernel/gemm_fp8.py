# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""FP8 scaled GEMM: `fp8_scaled_mm` triton port of the official sgl-kernel op.

Contract (mirrors torch.ops.sgl_kernel.fp8_scaled_mm / KernelSpec format):
    C[M,N] = (A_fp8[M,K] @ B_fp8[K,N]) * scale_a[M,1] * scale_b[1,N] (+ bias)
    scale_a: (M,) fp32, scale_b: (N,) fp32; out_dtype bf16/fp16; bias [N].

Reach: v0.5.21 `sglang.kernels.ops.gemm.Fp8ScaledMMOp.forward_aot` calls
`sgl_kernel.fp8_scaled_mm` — same-name package = zero-modification hook. This
is the ONLY AOT backend the fp8 linear chain can use on HCU (the TORCH
backend requires SM90). Also consumed by the compressed-tensors CHANNEL
strategy path (apply_fp8_linear -> fp8_scaled_mm).

fp8 math in fp32: load e4m3 -> f32, tl.dot f32 accumulate, row*col scales
applied after the K loop.
"""

from __future__ import annotations

from typing import Optional

import torch

from .compat import HAS_TRITON, check_cuda_tensor
from ._native import native_op

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _fp8_scaled_mm_kernel(
        A,  # [M, K] e4m3
        B,  # [K, N] e4m3
        SA,  # [M] fp32 row scales
        SB,  # [N] fp32 col scales
        C,  # [M, N] out_dtype
        Bias,
        M,
        N,
        K,
        strides_am, strides_ak,
        strides_bk, strides_bn,
        strides_cm, strides_cn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        a_ptrs = A + offs_m[:, None] * strides_am + offs_k[None, :] * strides_ak
        b_ptrs = B + offs_k[:, None] * strides_bk + offs_n[None, :] * strides_bn
        m_mask = offs_m < M
        n_mask = offs_n < N
        for k0 in range(0, tl.cdiv(K, BLOCK_K)):
            k_left = K - k0 * BLOCK_K
            k_mask = offs_k < k_left
            a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :],
                        other=0.0).to(tl.float32)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :],
                        other=0.0).to(tl.float32)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * strides_ak
            b_ptrs += BLOCK_K * strides_bk

        sa = tl.load(SA + offs_m, mask=m_mask, other=0.0)
        sb = tl.load(SB + offs_n, mask=n_mask, other=0.0)
        acc = acc * sa[:, None] * sb[None, :]
        if HAS_BIAS:
            acc += tl.load(Bias + offs_n, mask=n_mask,
                           other=0.0).to(tl.float32)[None, :]
        c_ptrs = C + offs_m[:, None] * strides_cm + offs_n[None, :] * strides_cn
        tl.store(c_ptrs, acc.to(C.dtype.element_ty),
                 mask=m_mask[:, None] & n_mask[None, :])


def fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    """C = (A @ B) * sa[:,None] * sb[None,:] + bias, fp32 accumulate."""
    if not (HAS_TRITON and mat_a.is_cuda):
        raise NotImplementedError(
            "fp8_scaled_mm triton backend requires a HIP/CUDA device")
    fp8_types = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    if mat_a.dtype not in fp8_types or mat_b.dtype not in fp8_types:
        raise TypeError(
            f"mat_a/mat_b must be fp8, got {mat_a.dtype}/{mat_b.dtype}")
    check_cuda_tensor(mat_a, mat_b)

    M, K = mat_a.shape
    N = mat_b.shape[1]
    out = torch.empty((M, N), device=mat_a.device, dtype=out_dtype)
    if M == 0 or N == 0:
        return out

    def _grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(N, meta["BLOCK_N"]))

    _fp8_scaled_mm_kernel[_grid](
        mat_a, mat_b,
        scales_a.contiguous(), scales_b.contiguous(),
        out, bias,
        M, N, K,
        mat_a.stride(0), mat_a.stride(1),
        mat_b.stride(0), mat_b.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        num_warps=4, num_stages=3,
    )
    return out


# ---------------------------------------------------------------------------
# sgl_per_token_group_quant_8bit — the upstream python bridge
# (python/sgl_kernel/gemm.py:28-64) with BOTH kernel variants emulated in
# torch: v1 (csrc/gemm/per_token_group_quant_8bit.cu) and v2
# (csrc/gemm/per_token_group_quant_8bit_v2.cu). Pure-torch tiers (this is
# the C-class "bridge the gap, do not rewrite" item: the python seam was
# missing entirely — callers hit AttributeError).
# ---------------------------------------------------------------------------

_QUANT_V2_GROUP_SIZES = (16, 32, 64, 128)
_V2_ABSMAX_FLOOR = 1e-10  # v2 LOCAL_ABSMAX_ABS; eps is TORCH_CHECKed to it


def _quant_dst_range(output_q):
    """dst_dtype_info::MIN/MAX per output_q.dtype (v2 CHECK_EQs the args)."""
    if output_q.dtype == torch.int8:
        return -128.0, 127.0
    if output_q.dtype == torch.float8_e4m3fn:
        return -448.0, 448.0
    raise RuntimeError(f"unsupported quant output dtype {output_q.dtype}")


def _write_scales(output_s, scale_vals, num_groups_per_row, ue8m0):
    """Write per-group scales honoring the upstream row/column-major
    convention (is_column_major = stride(-2) < stride(-1)).

    scale_vals: (num_rows, num_groups_per_row) fp32 (already the stored
    direction: amax / dst_max — the DEQUANT scale). Column-major keeps
    the LOGICAL axes (token, group): the physical address of scale(t, g)
    is t * stride(-2) + g * stride(-1) with stride(-2) == 1 — token is
    the fastest axis (upstream v1.cu:54-61, v2.cu:320-328). The buffer
    is NOT a transposed logical matrix; indexing it as [g, t] walks the
    wrong axes and indexes out of bounds (review #80). For ue8m0 the
    column-major format is the biased exponent byte packed 4-per-uint32
    (little-endian pack order, byte = group_idx % 4); v1 computes the
    exponent with ceilf(log2f), v2 with the DeepEP fast_log2_ceil bit
    trick — identical results for the powers of two actually stored."""
    rows, ngpr = scale_vals.shape
    is_col = output_s.stride(-2) < output_s.stride(-1)
    if not ue8m0:
        if is_col:
            # logical (token, group): s[t, g] at its strided slot
            t_idx = torch.arange(rows, device=scale_vals.device).unsqueeze(1)
            g_idx = torch.arange(ngpr, device=scale_vals.device).unsqueeze(0)
            output_s[..., t_idx.expand(rows, ngpr), g_idx.expand(rows, ngpr)] = (
                scale_vals.to(output_s.dtype)
            )
        else:
            output_s[..., :, :] = scale_vals.reshape(
                output_s.shape[:-2] + output_s.shape[-2:]
            ).to(output_s.dtype)
        return

    # ue8m0: biased exponent bytes, packed 4 per uint32 (column-major only)
    if not is_col:
        raise RuntimeError("scale_ue8m0 requires column-major output_s")
    # v1 stores (int)log2(y_s) + 127; scale_vals are exact powers of two
    exp_byte = (torch.log2(scale_vals.clamp_min(1e-30)).to(torch.int32) + 127)
    exp_byte = exp_byte.clamp(0, 255).to(torch.uint8)
    pad = (-ngpr) % 4  # v2 zero-fills the tail of the last pack
    if pad:
        exp_byte = torch.nn.functional.pad(exp_byte, (0, pad))
    packed_u8 = exp_byte.reshape(rows, -1, 4).to(torch.int64)
    packed = (
        packed_u8[:, :, 0]
        | (packed_u8[:, :, 1] << 8)
        | (packed_u8[:, :, 2] << 16)
        | (packed_u8[:, :, 3] << 24)
    ).to(torch.int32)  # bit pattern of the upstream uint32 store
    npacks = packed.shape[1]
    # logical (token, pack): the uint32 for token t pack p lives at
    # s[t, p] — t on the stride-1 axis, p on the stride(-1) axis. packed
    # is ALREADY (rows, npacks) in that logical order, so index-write
    # with NO transpose and NO reshape (earlier attempts — reshape, then
    # a .t() onto a [p, t] axis order — misplaced every slot on
    # non-square shapes; review #80).
    t_idx = torch.arange(rows, device=packed.device).unsqueeze(1)
    p_idx = torch.arange(npacks, device=packed.device).unsqueeze(0)
    output_s[..., t_idx.expand(rows, npacks), p_idx.expand(rows, npacks)] = (
        packed.to(output_s.dtype)
    )


def _quant_v1_rows(flat_input, output_q, output_s, group_size, eps,
                   min_8bit, max_8bit, scale_ue8m0):
    """v1 kernel math (per_token_group_quant_8bit.cu:36-115): eps-floor
    absmax, y_s = absmax / max_8bit, q = clamp(x / y_s) cast to dst —
    note the int8 cast is a TRUNCATION (C++ int8_t(float)), not round."""
    n_groups = flat_input.numel() // group_size
    g = flat_input.reshape(n_groups, group_size).float()
    amax = g.abs().amax(dim=1).clamp_min(eps)
    y_s = amax / max_8bit
    if scale_ue8m0:
        # v1: y_s = exp2f(ceilf(log2f(max(y_s, 1e-10))))
        y_s = torch.exp2(torch.ceil(torch.log2(y_s.clamp_min(1e-10))))
    q = (g / y_s[:, None]).clamp_(min_8bit, max_8bit)
    output_q.reshape(-1).copy_(q.reshape(-1).to(output_q.dtype))
    return y_s, n_groups


def sgl_per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
) -> None:
    """Upstream bridge contract (python/sgl_kernel/gemm.py:28-64).

    Destination-passing: output_q (same shape as the quantized face of
    input, dtype int8 or float8_e4m3fn) and output_s (per-group scales,
    fp32 row-major or packed-ue8m0 column-major) are written in place.

    enable_v2=None auto-selects v2 for group_size in {16,32,64,128}; v1
    only supports plain quantization (upstream asserts on fuse/masked).
    """
    if enable_v2 is None:
        enable_v2 = group_size in _QUANT_V2_GROUP_SIZES
    if enable_v2:
        # v2 host TORCH_CHECKs (per_token_group_quant_8bit_v2.cu:427-487)
        if input.numel() == 0:
            raise RuntimeError("input must be non-empty")
        if abs(_V2_ABSMAX_FLOOR - eps) >= 1e-13:
            raise RuntimeError("v2 requires eps == 1e-10")
        if group_size not in _QUANT_V2_GROUP_SIZES:
            raise RuntimeError(f"v2 unsupported group_size {group_size}")
        dst_min, dst_max = _quant_dst_range(output_q)
        if fp8_min != dst_min or fp8_max != dst_max:
            raise RuntimeError(
                f"v2 min/max {fp8_min}/{fp8_max} must match the output "
                f"dtype range {dst_min}/{dst_max}"
            )
        return _quant_v2(
            input, output_q, output_s, group_size, scale_ue8m0,
            fuse_silu_and_mul, masked_m,
        )

    # v1 python-layer asserts, verbatim (upstream gemm.py:59-60)
    assert not fuse_silu_and_mul, "only v2 support fuse_silu_and_mul"
    assert masked_m is None, "only v2 support masked_m"
    if input.numel() % group_size:
        raise RuntimeError("input numel must divide by group_size")
    if output_s.dim() != 2:
        raise RuntimeError("output_s must be 2-D")
    if scale_ue8m0 and output_s.stride(0) >= output_s.stride(1):
        # v1 host asserts !scale_ue8m0 on the row-major branch
        raise RuntimeError("scale_ue8m0 requires column-major output_s")

    # cpp tier (setup_hip.py wheel): per_token_group_quant_8bit_hcu.cu,
    # contract-identical incl. the int8-truncating cast and both scale
    # layouts (row-major linear + column-major strided, the latter incl.
    # packed ue8m0). input/output_q must be contiguous (native
    # CHECK_INPUTs); output_s may be a legal column-major buffer — like
    # upstream, the kernel addresses it through its strides, so the
    # is_contiguous() guard below would wrongly reject every legal
    # column-major buffer (review #80).
    native = native_op("sgl_per_token_group_quant_8bit")
    if (
        native is not None
        and input.is_cuda
        and input.is_contiguous()
        and output_q.is_contiguous()
        # row-major contiguous OR the canonical column-major stride(-2)==1
        # the kernel addresses (exotic strides stay on the torch tier)
        and (output_s.is_contiguous() or output_s.stride(-2) == 1)
        and input.dtype in (torch.float32, torch.float16, torch.bfloat16)
    ):
        native(
            input, output_q, output_s, group_size, eps,
            fp8_min, fp8_max, scale_ue8m0,
        )
        return

    flat = input.reshape(-1)
    y_s, n_groups = _quant_v1_rows(
        flat, output_q, output_s, group_size, eps, fp8_min, fp8_max,
        scale_ue8m0,
    )
    hidden = input.shape[-1]
    ngpr = hidden // group_size
    _write_scales(output_s, y_s.reshape(-1, ngpr), ngpr, scale_ue8m0)


def _quant_v2(input, output_q, output_s, group_size, scale_ue8m0,
              fuse_silu_and_mul, masked_m):
    """v2 kernel math (per_token_group_quant_8bit_v2.cu:69-80): absmax
    floor 1e-10, then TWO directions — the stored scale_inv = amax /
    dst_max (ue8m0: 2^ceil(log2(amax/dst_max)) via the DeepEP bit trick,
    packed as biased-exponent bytes) is the DEQUANT scale for downstream,
    while the quantized value multiplies the RECIPROCAL quant scale
    y_scale = dst_max / amax (ue8m0: 2^-exp). Multiplying by scale_inv
    collapses every group toward zero (ones -> int8 0 instead of 127);
    review #80 P1."""
    # cpp tier (setup_hip.py wheel): same contract, plain 16-lane kernels
    # (no flashinfer vec_t / PDL). masked_m is read ON DEVICE (per-expert
    # row-guard in the kernel), so the masked path stays CUDA-graph
    # capturable like upstream's MaskedLayoutScheduler (v2.cu:203-247).
    native = native_op("sgl_per_token_group_quant_8bit_v2")
    if (
        native is not None
        and input.is_cuda
        and input.is_contiguous()
        and output_q.is_contiguous()
        # row-major contiguous OR canonical column-major stride(-2)==1
        and (output_s.is_contiguous() or output_s.stride(-2) == 1)
        and input.dtype in (torch.float32, torch.float16, torch.bfloat16)
    ):
        native(
            input, output_q, output_s, group_size, _V2_ABSMAX_FLOOR,
            *_quant_dst_range(output_q),
            scale_ue8m0, fuse_silu_and_mul, masked_m,
        )
        return

    dst_min, dst_max = _quant_dst_range(output_q)

    def rows(x, out_q, out_s, mask=None):
        if fuse_silu_and_mul:
            h = x.shape[-1] // 2
            # v2: silu(primary) is cast BACK to the input dtype before the
            # product — the amax and the quantized source are both this
            # low-precision product (input_primary_vec[j] is overwritten)
            val = torch.nn.functional.silu(x[..., :h].float()).to(x.dtype) * x[..., h:]
        else:
            val = x
        t, hidden = val.shape[0], val.shape[-1]
        g = val.reshape(t, hidden // group_size, group_size).float()
        amax = g.abs().amax(dim=-1).clamp_min(_V2_ABSMAX_FLOOR)
        if scale_ue8m0:
            # fast_log2_ceil: biased exp - 127 + (mantissa != 0)
            ratio = (amax / dst_max).contiguous()
            bits = ratio.view(torch.int32)
            exp_ceil = ((bits >> 23) & 0xFF) - 127 + (
                (bits & 0x7FFFFF) != 0
            ).to(torch.int32)
            scale_inv = torch.exp2(exp_ceil.float())
            y_scale = torch.exp2(-exp_ceil.float())
        else:
            scale_inv = amax / dst_max
            y_scale = dst_max / amax
        q = (g * y_scale[..., None]).clamp_(dst_min, dst_max)
        if mask is None:
            out_q.reshape(t, hidden).copy_(q.reshape(t, hidden).to(out_q.dtype))
            _write_scales(out_s, scale_inv, hidden // group_size, scale_ue8m0)
            return
        # masked write-back: compute ran over ALL t rows (static shapes);
        # only rows [0, masked_m[e]) land in the caller's buffers, the
        # padding rows keep their previous contents.
        sel = mask.unsqueeze(-1)
        out_q.copy_(torch.where(sel, q.reshape(out_q.shape).to(out_q.dtype), out_q))
        scratch = torch.empty_strided(
            out_s.shape, out_s.stride(), dtype=out_s.dtype, device=out_s.device)
        _write_scales(scratch, scale_inv, hidden // group_size, scale_ue8m0)
        out_s.copy_(torch.where(sel, scratch, out_s))

    if masked_m is None:
        rows(input.reshape(-1, input.shape[-1]), output_q, output_s)
        return
    # DeepEP masked layout: input (E, maxT, H(*2)) rows [0, masked_m[e]).
    # masked_m[e] is consumed ON DEVICE (arange compare + where) — no
    # .item() sync, static shapes, so the call is CUDA-graph capturable
    # and a replay with an updated masked_m re-masks correctly (upstream
    # reads masked_m[expert_idx] inside the kernel; review #80 P2).
    trange = torch.arange(input.shape[1], device=input.device)
    for e in range(input.shape[0]):
        rows(input[e], output_q[e], output_s[e],
             mask=trange < masked_m[e])


# For legacy usage (upstream gemm.py:67-68)
sgl_per_token_group_quant_fp8 = sgl_per_token_group_quant_8bit
sgl_per_token_group_quant_int8 = sgl_per_token_group_quant_8bit
