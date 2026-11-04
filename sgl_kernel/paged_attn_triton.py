# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Paged attention (decode + wide-window prefill) — triton kernels
(fa3-facing, tier-③ triton-first).

Single-kernel streaming online-softmax over the whole KV sequence per
program: paged GQA pool in HND layout (pages, page, Hkv, D), bottom-right
aligned windows, fp32 accumulation (the emu's numerics contract).

- decode: one (batch, head) program, L=1 window, returns (o, lse) in the
  emu's ([B,1,H,D], [B,1,H]) layout.
- prefill: one (batch, head, q-chunk) program, L>1 windows on the paged
  pool (sglang's paged extend face), bottom-right causal — row t of the
  window sees keys j <= seq_len - L + t. L is a runtime arg so the merge
  of graph-capture bucket shapes shares one compile.

Design note: a split-K + merge two-phase variant was tried first and
numerically verified against raw math, but the merged version diverged from
the reference by a constant factor on HCU while this direct form matches
bit-for-bit at acl level — keep it simple until profiling shows the single
kernel is the bottleneck (decode sequences on HCU are ≤ 32K; a 64-lane CU
with BLOCK_N=page covers them at full memory throughput).

fp8 KV works naturally: the pool stores fp8 (uint8 storage), tl.load brings
raw bytes, compute upcasts to fp32. No warp-size assumptions anywhere
(masks + tl reductions only). Page ids are widened to int64 before address
math: a real server pool reaches millions of pages and (page * PAGE * HKV
* D) breaks int32 long before the table runs out of rows.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    q_ptr,            # [B, H, D]
    k_ptr,            # [P, PAGE, HKV, D]
    v_ptr,            # [P, PAGE, HKV, D]
    pt_ptr,           # [B, PT_W] int32
    seqlens_ptr,      # [B] int32
    o_ptr,            # [B, H, D] out (dtype = q's compute dtype)
    lse_ptr,          # [B, H] fp32
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PT_W: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,        # H // HKV
    BLOCK_N: tl.constexpr,  # tokens per inner iteration
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    seq_len = tl.load(seqlens_ptr + pid_b)
    d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)

    q = tl.load(q_ptr + pid_b * H * D + pid_h * D + d).to(tl.float32)
    kv_h = pid_h // G

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    for t0 in range(0, seq_len, BLOCK_N):
        offs = t0 + offs_n
        mask = offs < seq_len
        page_idx = offs // PAGE
        in_page = offs % PAGE
        ph = tl.load(pt_ptr + pid_b * PT_W + page_idx, mask=mask, other=0)
        base = ((ph.to(tl.int64) * PAGE + in_page) * HKV + kv_h) * D
        k = tl.load(k_ptr + base[:, None] + d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1) * scale
        s = tl.where(mask, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        v = tl.load(v_ptr + base[:, None] + d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    denom = tl.maximum(l_i, 1e-10)
    o = acc / denom
    tl.store(o_ptr + (pid_b * H + pid_h) * D + d, o)
    tl.store(lse_ptr + pid_b * H + pid_h, m_i + tl.log(denom))


def paged_attention_decode_triton(
    q,              # [B, 1, H, D]
    k_cache,        # [P, PAGE, HKV, D]
    v_cache,
    page_table,     # [B, PT_W] int32
    cache_seqlens,  # [B] int32
    softmax_scale=None,
    kv_scales=None,
):
    B, L, H, D = q.shape
    assert L == 1, "decode kernel takes L=1 windows (prefill stays on emu)"
    _, PAGE, HKV, _ = k_cache.shape
    scale = softmax_scale if softmax_scale is not None else D**-0.5

    seqlens = cache_seqlens.to(torch.int32)
    o16 = q.dtype if q.dtype.itemsize >= 2 else torch.bfloat16
    o = torch.empty(B, 1, H, D, dtype=o16, device=q.device)
    lse = torch.empty(B, 1, H, dtype=torch.float32, device=q.device)

    BLOCK_N = max(PAGE, 64)
    _paged_decode_kernel[(B, H)](
        q.reshape(B, H, D), k_cache, v_cache, page_table, seqlens,
        o.reshape(B, H, D), lse.reshape(B, H),
        scale,
        H=H, D=D, PAGE=PAGE, PT_W=page_table.shape[1],
        HKV=HKV, G=H // HKV, BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return o, lse


@triton.jit
def _paged_prefill_kernel(
    q_ptr,            # [B, LQ, H, D]
    k_ptr,            # [P, PAGE, HKV, D]
    v_ptr,
    pt_ptr,           # [B, PT_W] int32
    seqlens_ptr,      # [B] int32
    o_ptr,            # [B, LQ, H, D] out (dtype = q's compute dtype)
    lse_ptr,          # [B, LQ, H] fp32
    scale,
    LQ,               # q window length (runtime: one compile serves all buckets)
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PT_W: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,        # H // HKV
    BLOCK_M: tl.constexpr,  # q rows per program
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    seq_len = tl.load(seqlens_ptr + pid_b)
    kv_h = pid_h // G
    d = tl.arange(0, D)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = offs_m < LQ
    # bottom-right causal: window row t sees keys j <= seq_len - LQ + t
    bound = seq_len - LQ + offs_m

    q_offs = ((pid_b * LQ + offs_m)[:, None] * H + pid_h) * D + d[None, :]
    q = tl.load(q_ptr + q_offs, mask=row_ok[:, None], other=0.0)
    q = q.to(tl.float32)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # tl.minimum is broken on the HCU triton backend (decode-kernel find
    # #1); compare+where instead.
    hi = seq_len - LQ + (pid_m + 1) * BLOCK_M
    hi = tl.where(hi < seq_len, hi, seq_len)
    for t0 in range(0, hi, BLOCK_N):
        offs_n = t0 + tl.arange(0, BLOCK_N)
        n_ok = offs_n < seq_len
        page_idx = offs_n // PAGE
        in_page = offs_n % PAGE
        ph = tl.load(pt_ptr + pid_b * PT_W + page_idx, mask=n_ok, other=0)
        base = ((ph.to(tl.int64) * PAGE + in_page) * HKV + kv_h) * D
        k = tl.load(k_ptr + base[:, None] + d[None, :],
                    mask=n_ok[:, None], other=0.0).to(tl.float32)
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        keep = n_ok[None, :] & (offs_n[None, :] <= bound[:, None])
        s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # a fully masked row (partial last q chunk) must not poison exp
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        v = tl.load(v_ptr + base[:, None] + d[None, :],
                    mask=n_ok[:, None], other=0.0).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new

    denom = tl.maximum(l_i, 1e-10)
    o = acc / denom[:, None]
    tl.store(o_ptr + q_offs, o, mask=row_ok[:, None])
    tl.store(lse_ptr + (pid_b * LQ + offs_m) * H + pid_h,
             m_i + tl.log(denom), mask=row_ok)


def paged_attention_prefill_triton(
    q,              # [B, L, H, D] with L > 1 (paged extend windows)
    k_cache,        # [P, PAGE, HKV, D]
    v_cache,
    page_table,     # [B, PT_W] int32
    cache_seqlens,  # [B] int32 (prefix + this window's tokens)
    softmax_scale=None,
    kv_scales=None,
):
    B, L, H, D = q.shape
    assert L > 1, "wide-window kernel takes L>1 (decode uses the L=1 kernel)"
    _, PAGE, HKV, _ = k_cache.shape
    scale = softmax_scale if softmax_scale is not None else D**-0.5

    seqlens = cache_seqlens.to(torch.int32)
    o16 = q.dtype if q.dtype.itemsize >= 2 else torch.bfloat16
    o = torch.empty(B, L, H, D, dtype=o16, device=q.device)
    lse = torch.empty(B, L, H, dtype=torch.float32, device=q.device)
    pt = page_table.to(torch.int32).contiguous()

    _paged_prefill_kernel[(B, H, triton.cdiv(L, 64))](
        q.contiguous(), k_cache, v_cache, pt, seqlens, o, lse,
        scale, L,
        H=H, D=D, PAGE=PAGE, PT_W=pt.shape[1],
        HKV=HKV, G=H // HKV, BLOCK_M=64, BLOCK_N=max(PAGE, 64),
        num_warps=4,
    )
    return o, lse
