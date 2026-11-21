# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Triton masked grouped GEMM for DeepEP low-latency mode (bf16).

The DTK deepgemm fork's ``m_grouped_bf16_gemm_nt_masked`` only ships the
packed-layout masked kernel (BM64/BN256/BK128 HIPC): the bridge packs
[E, N, K] weights once and caches the copy — doubling expert memory
(DSV4-Flash EP: 69 GB raw + 69 GB packed > 144 GB card, OOM at graph
capture, validation-host-16 2026-10-21). This module reimplements the contract in
triton with NATIVE [E, N, K] weights: no packing, no extra memory, and
the same weight tensor serves the contiguous (prefill) path.

Contract (DeepGEMM m_grouped masked, matches the fork's entry):
    a [E, Mmax, K] bf16, b [E, N, K] bf16, d [E, Mmax, N] bf16,
    masked_m [E] int
    d[e, m, :] = a[e, m, :] @ b[e, :, :].T   for m < masked_m[e]
    (fp32 accumulate; rows >= masked_m[e] are left untouched)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_gemm_kernel(
    a_ptr, b_ptr, d_ptr, mm_ptr,
    Mmax, N, K,
    stride_ae, stride_am,
    stride_be, stride_bn,
    stride_de, stride_dm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_limit = tl.load(mm_ptr + pid_e)
    row0 = pid_m * BLOCK_M
    # whole row-block past this expert's real token count: skip (typical
    # decode step has 0-8 live tokens per expert, Mmax is the 128-token
    # dispatch buffer -- skipping dead blocks is most of the win)
    if row0 >= m_limit:
        return

    offs_m = row0 + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < m_limit
    n_mask = offs_n < N

    a_row = a_ptr + pid_e.to(tl.int64) * stride_ae + offs_m[:, None].to(tl.int64) * stride_am
    b_row = b_ptr + pid_e.to(tl.int64) * stride_be + offs_n[None, :].to(tl.int64) * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        km = kk < K
        at = tl.load(a_row + kk[None, :], mask=m_mask[:, None] & km[None, :], other=0.0)
        # b[e] is [N, K]; tile loaded transposed to [BLOCK_K, BLOCK_N]
        bt = tl.load(b_row + kk[:, None], mask=n_mask[None, :] & km[:, None], other=0.0)
        acc = tl.dot(at, bt, acc)

    o_ptrs = (
        d_ptr
        + pid_e.to(tl.int64) * stride_de
        + offs_m[:, None].to(tl.int64) * stride_dm
        + offs_n[None, :]
    )
    tl.store(o_ptrs, acc.to(d_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def m_grouped_bf16_gemm_nt_masked(
    a, b, d, masked_m=None, expected_m=None, expected_m_per_group=None,
    config=None, **unknown
):
    """d[e,m,:] = a[e,m,:] @ b[e].T for m < masked_m[e] (bf16, fp32 acc)."""
    if unknown:
        raise TypeError(
            f"m_grouped_bf16_gemm_nt_masked: unexpected kwargs {sorted(unknown)}"
        )
    if expected_m is None:
        expected_m = expected_m_per_group
    assert a.dim() == 3 and b.dim() == 3 and d.dim() == 3
    E, Mmax, K = a.shape
    E2, N, K2 = b.shape
    assert E == E2 and K == K2, f"shape mismatch a{a.shape} b{b.shape}"
    assert d.shape == (E, Mmax, N), f"bad out {d.shape}"
    assert masked_m is not None and int(masked_m.numel()) == E
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    assert d.is_contiguous(), "output must be contiguous"
    masked_m = masked_m.to(torch.int32)
    if not masked_m.is_contiguous():
        masked_m = masked_m.contiguous()

    # decode steps are skinny (few live tokens per expert): small BLOCK_M
    # avoids computing dead rows; expected_m is a per-group hint only.
    block_m = 16 if (expected_m or 0) <= 32 else 32
    BLOCK_M, BLOCK_N, BLOCK_K = block_m, 128, 64
    grid = (E, triton.cdiv(Mmax, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _masked_gemm_kernel[grid](
        a, b, d, masked_m,
        Mmax, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        d.stride(0), d.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=3,
    )
    return d
