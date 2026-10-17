"""Self-written triton paged-MQA logits (R-02 homegrown tier).

Writer arbitration (2026-10-18 probe campaign, nmz26 GPU4,
tmp_fetch/probe_r02_poollayout.py / probe_r02_scalepos.py): the production
index-K pool is written by ``fused_store_index_k_cache`` (the ``_is_cuda``
store arm — ``can_use_dsa_fused_store()`` returns True on HCU, measured),
whose csrc writes each page as k ROW-MAJOR (slot s ->
page*PS*132 + s*128) plus PS fp32 scales at the page tail
(page*PS*132 + PS*128 + 4*s). SetKAndS (the generic triton writer) has the
same row-major+tail layout without preshuffle, and a 16-tile interleave
branch under the aiter-preshuffle gate — that branch LOSES the store
arbitration in production and is supported here only as an opt-in arm
(HCU_MQA_TRITON_PRESHUFFLE=1) for stacks where it does write.

The torch emu's historical row-level unpack (flat (T,132) [k|scale] rows)
misread that page-level layout — k misaligned by 4*s bytes per slot,
scales read off k bytes — so production decode DSA top-k ran
deterministically wrong; fixed in dsa_paged_mqa.py the same day
(page-level split, PS=1 world byte-identical).

This kernel reads the real layout directly, fusing fp8-e4m3fn decode +
paged gather + weighted head contraction + scale fold into ONE launch,
and early-exits page programs beyond each row's context — the captured
decode graph runs the full 64K window, so most programs exit after two
scalar loads. Numerics: q decoded host-side, k decoded in-kernel via
integer bit math (portable; dodges any DTK fp8-conversion quirk), fp32
accumulation throughout — tighter than the emu's bf16 matmuls. Measured:
exact vs direct math (rel 0.0, full top-k) on a real
fused_store-written pool; decode shape N=8 / 64K window 13.2x over the
vec emu (0.19 vs 2.52 ms).

HCU_MQA_LOGITS_IMPL=triton selects this tier; the vec/loop emu (now
layout-fixed) stays the default. Gate: tests/kernels/test_mqa_triton_
gfx938.py (production-writer pool, fixed-emu cross-check, SetKAndS arms).
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - container always has triton
    triton = None
    tl = None


if tl is not None:

    @triton.jit
    def _paged_mqa_logits_kernel(
        q_ptr,  # (N, H, D) fp32, pre-decoded
        w_ptr,  # (N, H) fp32
        pool_ptr,  # (P, PS*132) uint8
        sf_ptr,  # (P, PS*33) fp32 view of pool_ptr
        bt_ptr,  # (N, bt_cols) int32
        lens_ptr,  # (N,) int32
        out_ptr,  # (N, T_ALL) fp32, pre-zeroed
        T_ALL,
        BT_STRIDE,
        H: tl.constexpr,
        D: tl.constexpr,
        PS: tl.constexpr,
        SLOT_TILE: tl.constexpr,
        TILE: tl.constexpr,
        PRESHUFFLE: tl.constexpr,
    ):
        i = tl.program_id(0)
        nt = PS // SLOT_TILE
        pid = tl.program_id(1)
        p = pid // nt  # page position in walk order
        s0 = (pid % nt) * SLOT_TILE
        L = tl.load(lens_ptr + i)
        if p * PS + s0 >= L:
            return  # pages past this row's context: exit before any loads
        page = tl.load(bt_ptr + i * BT_STRIDE + p)
        slot = s0 + tl.arange(0, SLOT_TILE)
        d = tl.arange(0, D)
        if PRESHUFFLE:
            # SetKAndS aiter-preshuffle arm: 16x16 tile interleave
            st = slot // TILE
            si = slot % TILE
            dt = d // TILE
            di = d % TILE
            koff = (
                st[:, None] * (TILE * D)
                + dt[None, :] * (TILE * TILE)
                + si[:, None] * TILE
                + di[None, :]
            )
        else:
            # production layout: k row-major per page
            koff = slot[:, None] * D + d[None, :]
        base = page.to(tl.int64) * (PS * 132)
        u = tl.load(pool_ptr + base + koff).to(tl.int32)  # (SLOT_TILE, D)
        # e4m3fn decode: sign(1) exp(4,bias7) man(3). Subnormal e=0:
        # m/8 * 2^-6. NaN payload (e=15,m=7 -> 480 here) is never produced
        # by the quantizer.
        sgn = (u >> 7) & 1
        e = (u >> 3) & 15
        m = u & 7
        pw = tl.where(e == 0, -6, e - 7)
        mant = tl.where(e == 0, m, m + 8).to(tl.float32) * 0.125
        two = ((pw + 127) << 23).to(tl.float32, bitcast=True)  # 2^pw
        kf = tl.where(sgn == 1, -(mant * two), mant * two)
        acc = tl.zeros((SLOT_TILE,), tl.float32)
        for h in tl.static_range(H):
            qv = tl.load(q_ptr + (i * H + h) * D + d)
            wh = tl.load(w_ptr + i * H + h)
            acc += wh * tl.sum(kf * qv[None, :], axis=1)
        # fp32 scales at the page tail (both writer arms agree on this)
        sfbase = page.to(tl.int64) * (PS * 33) + PS * 32
        sf = tl.load(sf_ptr + sfbase + slot)
        col = p * PS + slot
        tl.store(
            out_ptr + i.to(tl.int64) * T_ALL + col,
            acc * sf,
            mask=(col < L) & (col < T_ALL),
        )


def triton_paged_mqa_logits(
    q_fp8,
    kv_cache_fp8,
    weights,
    context_lens,
    block_table,
    schedule_metadata=None,
    max_context_len=None,
    clean_logits=False,
    **_ignored,
):
    """Adapter entry for HCU_MQA_LOGITS_IMPL=triton (homegrown kernel)."""
    from .dsa_paged_mqa import _as_view, _fp8_view_dtype

    q_flat = (
        q_fp8.reshape(-1, q_fp8.shape[-2], q_fp8.shape[-1])
        if q_fp8.dim() == 4
        else q_fp8
    )
    q_b = _as_view(q_flat, _fp8_view_dtype()).to(torch.float32).contiguous()
    N, H, D = q_b.shape

    kv = kv_cache_fp8
    if kv.dim() == 2:
        kv = kv.reshape(-1, 64, 1, kv.shape[-1])
    PS = kv.shape[1] if kv.dim() == 4 else 1
    pool = kv.reshape(kv.shape[0], -1).contiguous()
    if pool.dtype == torch.int8:
        pool = pool.view(torch.uint8)
    sf = pool.view(torch.float32)  # (P, PS*33)

    w = weights.reshape(q_b.shape[0], -1) if weights.dim() == 3 else weights
    w = w.reshape(N, H).to(torch.float32).contiguous()
    ctx = context_lens.to(torch.int32).reshape(-1).contiguous()
    bt = block_table.to(torch.int32).contiguous()

    out_width = int(max_context_len or bt.shape[1] * PS)
    T_all = min(out_width, bt.shape[1] * PS)
    out = torch.zeros((N, out_width), dtype=torch.float32, device=q_b.device)
    if N == 0 or T_all == 0:
        return out

    slot_tile = PS if PS < 64 else 64
    if PS % slot_tile or slot_tile & (slot_tile - 1):
        # pool geometry outside {1, 64, 256, ...}: not reachable via the
        # writer contract; refuse rather than guess.
        raise NotImplementedError(
            f"mqa_triton: page_size {PS} not a power-of-2 multiple of its "
            "slot tile"
        )
    # production writer (fused_store) is row-major; the SetKAndS
    # aiter-preshuffle interleave arm is opt-in for stacks that write it
    preshuffle = os.environ.get("HCU_MQA_TRITON_PRESHUFFLE", "0") == "1"
    grid = (N, ((T_all + PS - 1) // PS) * (PS // slot_tile))
    _paged_mqa_logits_kernel[grid](
        q_b,
        w,
        pool,
        sf,
        bt,
        ctx,
        out,
        T_all,
        bt.stride(0),
        H=H,
        D=D,
        PS=PS,
        SLOT_TILE=slot_tile,
        TILE=16,
        PRESHUFFLE=preshuffle,
        num_warps=4,
    )
    return out
