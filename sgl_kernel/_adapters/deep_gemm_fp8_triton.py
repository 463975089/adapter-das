"""Triton w8a8 block-scaled fp8 grouped GEMMs for the DeepEP layouts.

The DTK deepgemm fork's fp8 grouped kernels are per-channel-scale-only with
broken signed numerics (R-04, 2026-10-18: contiguous rel 0.88 on unit
scales), which forced the DSV4-Flash DeepEP route onto bf16 experts (2x
weight bytes, 69 GB/card). This module re-implements the two DeepGEMM fp8
contracts in triton with the upstream DeepSeek scale recipe and NATIVE
[E, N, K] fp8 weights -- half the weight traffic of bf16, no packing, no
extra memory:

    masked (deepep low-latency decode):
        a [E, Mmax, K] fp8e4m3, a_s [E, Mmax, K//128] fp32 (per-token-group)
        b [E, N, K] fp8e4m3, b_s [E, N//128, K//128] fp32 (128x128 block)
        out [E, Mmax, N] bf16; rows >= masked_m[e] are not written
    contiguous (deepep normal prefill, alignment-padded groups):
        a [M, K] fp8, a_s [M, K//128] fp32, m_indices [M] int32
        b/out scales as above, out [M, N] bf16

Numerics: fp8 tiles are converted to bf16 and dotted with fp32
accumulation; because both scales are constant across each 128-wide
K-block, they fold onto the accumulator once per K-block (probe
tmp_fetch/probe_triton_fp8.py: elementwise convert exact, acc-fold
maxrel 3.8e-3 vs an fp32 dequant reference). tl.dot stays in bf16 --
on this memory-bound decode path the win is the halved fp8 reads, and
gfx938 bf16 MFMA runs at full rate.

The wrapper's TMA-aligned scale transform is bypassed with an identity
on the shim's layout module (deep_gemm_dtk) -- these kernels index the
LOGICAL [.., K//128] layout, whatever the strides.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_masked_kernel(
    a_ptr, as_ptr, b_ptr, bs_ptr, out_ptr, mm_ptr,
    Mmax, N, K,
    stride_ae, stride_am,
    stride_ase, stride_asm, stride_ask,
    stride_be, stride_bn,
    stride_bse, stride_bsn, stride_bsk,
    stride_oe, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    row0 = pid_m * BLOCK_M
    m_limit = tl.load(mm_ptr + e)
    if row0 >= m_limit:
        return

    offs_m = row0 + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_rows = a_ptr + e.to(tl.int64) * stride_ae + offs_m[:, None].to(tl.int64) * stride_am
    as_rows = as_ptr + e.to(tl.int64) * stride_ase + offs_m.to(tl.int64) * stride_asm
    b_rows = b_ptr + e.to(tl.int64) * stride_be + offs_n[None, :].to(tl.int64) * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        a8 = tl.load(a_rows + kk[None, :]).to(tl.bfloat16)
        # b[e] is [N, K]; load the tile transposed to [BLOCK_K, BLOCK_N]
        b8 = tl.load(b_rows + kk[:, None]).to(tl.bfloat16)
        p = tl.dot(a8, b8)
        # scale block is 128 regardless of BLOCK_K (BK=64 covers half a block)
        kb = k0 // 128
        a_s = tl.load(as_rows + kb * stride_ask)
        b_s = tl.load(
            bs_ptr
            + e.to(tl.int64) * stride_bse
            + pid_n * stride_bsn
            + kb * stride_bsk
        )
        acc += p * a_s[:, None] * b_s

    o = (
        out_ptr
        + e.to(tl.int64) * stride_oe
        + offs_m[:, None].to(tl.int64) * stride_om
        + offs_n[None, :]
    )
    tl.store(o, acc.to(tl.bfloat16), mask=(offs_m < m_limit)[:, None])


def _check_pair(pair, name):
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise TypeError(f"{name} must be a (tensor, scale) pair, got {type(pair)}")
    return pair


def fp8_m_grouped_gemm_nt_masked(
    lhs,
    rhs,
    out,
    masked_m,
    expected_m=None,
    max_block_n=None,
    enable_overlap=None,
    signal=None,
    recipe_a=None,
    recipe_b=None,
    **unknown,
):
    """d[e,m,:] = (a[e,m,:] * a_s[e,m,kb]) @ (b[e] * b_s[e,nb,kb]).T, m < masked_m[e]."""
    if unknown:
        raise TypeError(
            f"fp8_m_grouped_gemm_nt_masked: unexpected kwargs {sorted(unknown)}"
        )
    if recipe_a is not None or recipe_b is not None:
        raise TypeError(
            "fp4/mxfp8 scale recipes are not implemented by the triton "
            "fp8 grouped GEMM (128x128 DeepSeek recipe only)"
        )
    a, a_s = _check_pair(lhs, "lhs")
    b, b_s = _check_pair(rhs, "rhs")
    assert a.dim() == 3 and b.dim() == 3 and out.dim() == 3
    E, Mmax, K = a.shape
    E2, N, K2 = b.shape
    assert (E, K) == (E2, K2), f"lhs/rhs group/K mismatch {a.shape} vs {b.shape}"
    assert out.shape == (E, Mmax, N), f"out {out.shape} vs ({E},{Mmax},{N})"
    assert K % 128 == 0 and N % 128 == 0, "triton fp8 grouped GEMM needs K,N % 128"
    assert a_s.dim() == 3 and a_s.shape[:2] == (E, Mmax) and a_s.shape[2] == K // 128, (
        f"a_s {tuple(a_s.shape)} != ({E},{Mmax},{K // 128})"
    )
    assert b_s.dim() == 3 and b_s.shape == (E, N // 128, K // 128), (
        f"b_s {tuple(b_s.shape)} != ({E},{N // 128},{K // 128})"
    )
    assert a_s.dtype == torch.float32 and b_s.dtype == torch.float32
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if not a_s.is_contiguous():
        a_s = a_s.contiguous()
    if not b_s.is_contiguous():
        b_s = b_s.contiguous()

    block_m = 16 if (expected_m or 32) <= 32 else 32
    grid = (E, triton.cdiv(Mmax, block_m), triton.cdiv(N, 128))
    _fp8_masked_kernel[grid](
        a, a_s, b, b_s, out, masked_m.to(torch.int32),
        Mmax, N, K,
        a.stride(0), a.stride(1),
        a_s.stride(0), a_s.stride(1), a_s.stride(2),
        b.stride(0), b.stride(1),
        b_s.stride(0), b_s.stride(1), b_s.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=128, BLOCK_K=64,
        # w4/s3 is the sweep winner on gfx938 (nmz16 2026-10-21): the fp8
        # tile + convert working set at num_warps=8 serializes (11.9ms ->
        # 2.15ms on the E32 m16 K4096 N4096 bench); bf16 masked uses w8.
        num_warps=4, num_stages=3,
    )
    return out


@triton.jit
def _fp8_contig_kernel(
    x_ptr, xs_ptr, w_ptr, ws_ptr, out_ptr, midx_ptr,
    M, N, K,
    stride_xm,
    stride_xs_m, stride_xs_k,
    stride_we, stride_wn,
    stride_wse, stride_wsn, stride_wsk,
    stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row0 = pid_m * BLOCK_M

    # one expert per row-block (alignment-padded contiguous groups; BLOCK_M
    # divides the alignment so blocks never straddle a group boundary)
    e = tl.load(midx_ptr + row0)
    e = tl.maximum(e, 0).to(tl.int64)

    offs_m = row0 + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    x_rows = x_ptr + offs_m[:, None].to(tl.int64) * stride_xm
    w_rows = w_ptr + e * stride_we + offs_n[None, :].to(tl.int64) * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        x8 = tl.load(x_rows + kk[None, :], mask=m_mask[:, None], other=0.0).to(tl.bfloat16)
        w8 = tl.load(w_rows + kk[:, None]).to(tl.bfloat16)
        p = tl.dot(x8, w8)
        kb = k0 // 128
        x_s = tl.load(xs_ptr + offs_m.to(tl.int64) * stride_xs_m + kb * stride_xs_k,
                      mask=m_mask, other=0.0)
        w_s = tl.load(
            ws_ptr + e * stride_wse + pid_n * stride_wsn + kb * stride_wsk
        )
        acc += p * x_s[:, None] * w_s

    o_ptrs = out_ptr + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None])


def m_grouped_fp8_gemm_nt_contiguous(lhs, rhs, out, m_indices, **unknown):
    """d[m,:] = (a[m,:] * a_s[m,kb]) @ (b[m_indices[m]] * b_s[.,nb,kb]).T."""
    if unknown:
        raise TypeError(
            f"m_grouped_fp8_gemm_nt_contiguous: unexpected kwargs {sorted(unknown)}"
        )
    a, a_s = _check_pair(lhs, "lhs")
    b, b_s = _check_pair(rhs, "rhs")
    assert a.dim() == 2 and b.dim() == 3 and out.dim() == 2
    M, K = a.shape
    E, N, K2 = b.shape
    assert K == K2, f"K mismatch {K} vs {K2}"
    assert out.shape == (M, N) and out.is_contiguous()
    assert m_indices.shape[0] == M
    assert K % 128 == 0 and N % 128 == 0
    assert a_s.dim() == 2 and a_s.shape == (M, K // 128), (
        f"a_s {tuple(a_s.shape)} != ({M},{K // 128})"
    )
    assert b_s.dim() == 3 and b_s.shape == (E, N // 128, K // 128)
    assert a_s.dtype == torch.float32 and b_s.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if not a_s.is_contiguous():
        a_s = a_s.contiguous()
    if not b_s.is_contiguous():
        b_s = b_s.contiguous()

    grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
    _fp8_contig_kernel[grid](
        a, a_s, b, b_s, out, m_indices.to(torch.int32),
        M, N, K,
        a.stride(0),
        a_s.stride(0), a_s.stride(1),
        b.stride(0), b.stride(1),
        b_s.stride(0), b_s.stride(1), b_s.stride(2),
        out.stride(0),
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=64,
        num_warps=4, num_stages=3,
    )
    return out
