"""Paged decode attention — triton kernel (fa3-facing, tier-③ triton-first).

Single-kernel streaming online-softmax over the whole KV sequence per
(batch, head) program: paged GQA pool in HND layout (pages, page, Hkv, D),
bottom-right-aligned L=1 decode window, fp32 accumulation (the emu's
numerics contract), returns (o [B,1,H,D], lse [B,1,H]) in the emu's layout.

Design note: a split-K + merge two-phase variant was tried first and
numerically verified against raw math, but the merged version diverged from
the reference by a constant factor on HCU while this direct form matches
bit-for-bit at acl level — keep it simple until profiling shows the single
kernel is the bottleneck (decode sequences on HCU are ≤ 32K; a 64-lane CU
with BLOCK_N=page covers them at full memory throughput).

fp8 KV works naturally: the pool stores fp8 (uint8 storage), tl.load brings
raw bytes, compute upcasts to fp32. No warp-size assumptions anywhere
(masks + tl reductions only).
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
        base = ((ph * PAGE + in_page) * HKV + kv_h) * D
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
