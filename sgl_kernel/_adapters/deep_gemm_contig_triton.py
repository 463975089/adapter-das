"""Triton contiguous-layout grouped GEMM for DeepEP normal mode (bf16).

The DTK deepgemm fork's ``m_grouped_bf16_gemm_nt_contiguous`` is broken on
this stack (2D input VMFault, 3D input wrong numbers — probed 2026-10-11,
tests/tools/_test_dtk_deepgemm.py). This module reimplements the contract
in triton with NATIVE [E, N, K] weights: no HIPC packing, no extra memory.

Contract (DeepGEMM m_grouped contiguous):
    a [M, K] bf16, b [E, N, K] bf16, d [M, N] bf16, m_indices [M] int32
    d[m, :] = a[m, :] @ b[m_indices[m], :, :].T     (fp32 accumulate)

Layout guarantee the kernel exploits: rows are grouped by expert and every
group's row count is padded to a multiple of ``alignment`` (=128, set by
deep_ep normal dispatch expert_alignment and ep_scatter). With
BLOCK_M dividing 128, a row-block never spans two experts, so one scalar
expert id per block turns the grouped GEMM into a plain tiled GEMM with a
gathered weight base. m_indices are always in-range for real rows; pad
rows carry a valid expert id with garbage activations whose outputs the
ep_gather never reads back (row-independent GEMM -> safe).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_ALIGN = [128]


def get_mk_alignment_for_contiguous_layout():
    return _ALIGN[0]


def set_mk_alignment_for_contiguous_layout(alignment: int):
    _ALIGN[0] = int(alignment)


@triton.jit
def _contig_gemm_kernel(
    x_ptr, w_ptr, out_ptr, midx_ptr,
    M, N, K,
    stride_xm, stride_we, stride_wn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_row = x_ptr + offs_m[:, None].to(tl.int64) * stride_xm
    w_row = w_ptr + e * stride_we + offs_n[None, :].to(tl.int64) * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        km = kk < K
        xt = tl.load(x_row + kk[None, :], mask=m_mask[:, None] & km[None, :], other=0.0)
        # b[e] is [N, K]; tile loaded transposed to [BLOCK_K, BLOCK_N]
        wt = tl.load(w_row + kk[:, None], mask=n_mask[None, :] & km[:, None], other=0.0)
        acc = tl.dot(xt, wt, acc)

    o_ptrs = out_ptr + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices, config=None, **unknown):
    """d[m,:] = a[m,:] @ b[m_indices[m]].T  (bf16 in/out, fp32 accumulate)."""
    if unknown:
        raise TypeError(
            f"m_grouped_bf16_gemm_nt_contiguous: unexpected kwargs {sorted(unknown)}"
        )
    assert a.dim() == 2 and b.dim() == 3 and d.dim() == 2
    M, K = a.shape
    E, N, K2 = b.shape
    assert K == K2, f"K mismatch {K} vs {K2}"
    assert m_indices.shape[0] == M
    # inputs may copy; the OUTPUT must not (a contiguous() copy would
    # swallow the result and the caller's tensor stay untouched)
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    assert d.is_contiguous(), "output must be contiguous"

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _contig_gemm_kernel[grid](
        a, b, d, m_indices,
        M, N, K,
        a.stride(0), b.stride(0), b.stride(1), d.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=3,
    )
    return d
