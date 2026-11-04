# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""INT8 scaled GEMM: `int8_scaled_mm` triton port of the official sgl-kernel op.

Contract (mirrors torch.ops.sgl_kernel.int8_scaled_mm as called by
compressed_tensors_w8a8_int8.apply_weights):
    C[M,N] = (A_i8[M,K] @ B_i8[K,N]) * scale_a[M,1] * scale_b[N,1] + bias
    scale_a: (M,) or (M,1) fp32 (per-token dynamic quant),
    scale_b: (N,) or (N,1) fp32 (per-channel weight scale),
    out_dtype bf16/fp16; bias [N]. No asymmetric zero-point (azp) — the
    symmetric dynamic-quant path this adapter targets never sets it.

Reach: v0.5.21 CompressedTensorsW8A8Int8 (channel-strategy w8a8 int8, the
"Channel-INT8-w8a8" checkpoint family) — its `_is_cuda` gate skips the
`from sgl_kernel import int8_scaled_mm` import on HIP (NameError at call
time); sitecustomize re-binds the module attribute so this triton port
becomes the AOT backend on HCU.
"""

from __future__ import annotations

import torch

from .compat import HAS_TRITON, check_cuda_tensor

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _int8_scaled_mm_kernel(
        A, B, SA, SB, C, Bias,
        M, N, K,
        strides_am, strides_ak,
        strides_bk, strides_bn,
        strides_cm, strides_cn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        n_mask = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        a_ptrs = A + offs_m[:, None] * strides_am + offs_k[None, :] * strides_ak
        b_ptrs = B + offs_k[:, None] * strides_bk + offs_n[None, :] * strides_bn
        for k0 in range(0, tl.cdiv(K, BLOCK_K)):
            k_left = K - k0 * BLOCK_K
            k_mask = offs_k < k_left
            a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
            acc += tl.dot(a, b).to(tl.int32)
            a_ptrs += BLOCK_K * strides_ak
            b_ptrs += BLOCK_K * strides_bk

        acc = acc.to(tl.float32)
        sa = tl.load(SA + offs_m, mask=m_mask, other=0.0)
        sb = tl.load(SB + offs_n, mask=n_mask, other=0.0)
        acc = acc * sa[:, None] * sb[None, :]
        if HAS_BIAS:
            acc += tl.load(Bias + offs_n, mask=n_mask,
                           other=0.0).to(tl.float32)[None, :]
        c_ptrs = C + offs_m[:, None] * strides_cm + offs_n[None, :] * strides_cn
        tl.store(c_ptrs, acc.to(C.dtype.element_ty),
                 mask=m_mask[:, None] & n_mask[None, :])


def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    """C = (A @ B) * sa[:,None] * sb[:,None or 1,N] + bias, int32 accumulate."""
    if not (HAS_TRITON and mat_a.is_cuda):
        raise NotImplementedError(
            "int8_scaled_mm triton backend requires a HIP/CUDA device")
    if mat_a.dtype != torch.int8 or mat_b.dtype != torch.int8:
        raise TypeError(
            f"mat_a/mat_b must be int8, got {mat_a.dtype}/{mat_b.dtype}")
    check_cuda_tensor(mat_a, mat_b)

    M, K = mat_a.shape
    N = mat_b.shape[1]
    out = torch.empty((M, N), device=mat_a.device, dtype=out_dtype)
    if M == 0 or N == 0:
        return out

    sa = scales_a.reshape(M) if scales_a.numel() == M else scales_a.reshape(1, -1).reshape(-1)[:1].expand(M)
    sa = sa.contiguous().float()
    sb = scales_b.reshape(N).contiguous().float()

    def _grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(N, meta["BLOCK_N"]))

    _int8_scaled_mm_kernel[_grid](
        mat_a, mat_b, sa, sb, out, bias,
        M, N, K,
        mat_a.stride(0), mat_a.stride(1),
        mat_b.stride(0), mat_b.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        num_warps=4, num_stages=3,
    )
    return out
