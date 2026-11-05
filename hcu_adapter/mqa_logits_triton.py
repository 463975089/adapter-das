# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Self-written triton paged-MQA logits (R-02 homegrown tier).

Pool layout (2026-10-18 R-02 audit, final — writer census, nmz26 GPU4):
GLM-5.3 runs IndexerKPool (kpool=4), whose store flow NEVER calls the
base Indexer's `_store_index_k_cache` — so fused_store (row-major) never
fires. The pool is written by the kpool writer family
(``kpool_softmax_rotate_write_cache`` etc., stock triton -> gfx938
div-4 bug -> ``kpool_write_emu``, interleave via ``_TILE=16``) and
``SetKAndS`` under the aiter-preshuffle gate the adapter forces on
(``HCU_DSA_PRESHUFFLE=1`` default). Both lay a page out as k in the
16-tile INTERLEAVE order + PS fp32 scales at the tail — and the prefill
gather (``_kpool_cache_k_offsets``) reads the same order, so prefill is
consistent. Only the decode paged-MQA reader was wrong: the torch emu's
row-level 132-byte split misread the page-level layout entirely (k
shifted, scales read off k bytes) — decode DSA top-k ran
deterministically wrong in production; fixed in dsa_mqa_logits_emu.py
(interleave-aware page-level unpack) the same day.

This kernel reads the real layout directly, fusing fp8-e4m3fn decode +
de-interleave gather + weighted head contraction + scale fold into ONE
launch, and early-exits page programs beyond each row's context — the
captured decode graph runs the full 64K window, so most programs exit
after two scalar loads. Numerics: q decoded host-side, k decoded in-kernel
via integer bit math (portable; dodges any DTK fp8-conversion quirk), fp32
accumulation throughout — tighter than the emu's bf16 matmuls. Measured:
exact vs direct math (rel 0.0, full top-k) on real-writer pools (both
arms); decode shape N=8 / 64K window 13.2x over the vec emu (0.19 vs
2.52 ms).

The arm follows the ecosystem switch ``HCU_DSA_PRESHUFFLE`` (default on
= interleave = GLM-5.3 production; ``=0`` selects the row-major arm for
the legacy page_size=1 world, where fused_store/kpool=1 pools are
row-major). HCU_MQA_LOGITS_IMPL=triton selects this tier; the (fixed)
vec/loop emu stays the default. Gate: tests/kernels/test_mqa_triton_
gfx938.py (SetKAndS-written pools in both layouts, fixed-emu cross-check).
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
    from .dsa_mqa_logits_emu import _as_view, _fp8_view_dtype

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
            f"mqa_logits_triton: page_size {PS} not a power-of-2 multiple of its "
            "slot tile"
        )
    # arm follows the ecosystem switch: interleave under the forced
    # preshuffle gate (GLM-5.3 kpool writer family / SetKAndS), row-major
    # in the legacy HCU_DSA_PRESHUFFLE=0 world (fused_store / kpool=1)
    preshuffle = os.environ.get("HCU_DSA_PRESHUFFLE", "1") != "0"
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


if tl is not None:

    @triton.jit
    def _ragged_mqa_logits_kernel(
        q_ptr,  # (N, H, D) fp32, pre-decoded
        w_ptr,  # (N, H) fp32
        k_ptr,  # (T, D) uint8 (fp8 storage)
        sf_ptr,  # (T,) fp32
        ks_ptr,  # (N,) int32
        ke_ptr,  # (N,) int32
        out_ptr,  # (N, T) fp32, written for every column
        T,
        H: tl.constexpr,
        D: tl.constexpr,
        TT: tl.constexpr,
    ):
        # RAGGED sibling of _paged_mqa_logits_kernel (C4 nonpaged plan,
        # dsv4/indexer.py:_nonpaged_mqa_logits -> deep_gemm.fp8_mqa_logits):
        # k is a CONTIGUOUS gathered buffer (no page table, no interleave —
        # the gather already deinterleaved), each row reads [ks, ke) and is
        # -inf outside, mirroring mqa_logits_emu._ragged_fp8_mqa_logits
        # exactly. The emu is ~58 aten kernels per call (fingerprinted
        # 2026-10-24 nmz18: 38 manual_unroll + 31 vectorized + 6 Cijk bmm +
        # 6 reduce + arange) x 44 C4 layers/step — the biggest single
        # elementwise family in the wave6 decode trace.
        i = tl.program_id(0)
        t0 = tl.program_id(1) * TT
        t = t0 + tl.arange(0, TT)
        tmask = t < T
        ks = tl.load(ks_ptr + i)
        ke = tl.load(ke_ptr + i)
        valid = (t >= ks) & (t < ke) & tmask
        d = tl.arange(0, D)
        if t0 >= ke or (t0 + TT) <= ks:
            # outside the window: only the -inf fill matters
            tl.store(
                out_ptr + i.to(tl.int64) * T + t,
                tl.full((TT,), float("-inf"), tl.float32),
                mask=tmask,
            )
            return
        u = tl.load(
            k_ptr + t.to(tl.int64)[:, None] * D + d[None, :], mask=tmask[:, None], other=0
        ).to(tl.int32)
        sgn = (u >> 7) & 1
        e = (u >> 3) & 15
        m = u & 7
        pw = tl.where(e == 0, -6, e - 7)
        mant = tl.where(e == 0, m, m + 8).to(tl.float32) * 0.125
        two = ((pw + 127) << 23).to(tl.float32, bitcast=True)
        kf = tl.where(sgn == 1, -(mant * two), mant * two)
        acc = tl.zeros((TT,), tl.float32)
        for h in tl.static_range(H):
            qv = tl.load(q_ptr + (i * H + h) * D + d)
            wh = tl.load(w_ptr + i * H + h)
            acc += wh * tl.sum(kf * qv[None, :], axis=1)
        sf = tl.load(sf_ptr + t, mask=tmask, other=0.0)
        tl.store(
            out_ptr + i.to(tl.int64) * T + t,
            tl.where(valid, acc * sf, float("-inf")),
            mask=tmask,
        )


if tl is not None:

    @triton.jit
    def _paged_ps1_mqa_logits_kernel(
        q_ptr,  # (N, H, D) fp32, pre-decoded
        w_ptr,  # (N, H) fp32
        pool_ptr,  # (T, 132) uint8: 128B row-major k + 4B fp32 scale
        sf_ptr,  # (T, 33) fp32 view of pool_ptr
        bt_ptr,  # (N, bt_cols) int32 TOKEN ids (page_size=1 world)
        lens_ptr,  # (N,) int32
        out_ptr,  # (N, T_ALL) fp32, pre-zeroed (window-out = 0.0)
        T_ALL,
        BT_STRIDE,
        H: tl.constexpr,
        D: tl.constexpr,
        TT: tl.constexpr,
    ):
        # PS=1 sibling of _paged_mqa_logits_kernel for the aiter
        # paged-MQA face the vllm/emu world sees (mqa_logits_emu.
        # _paged_mqa_logits contract): the pool is a flat (T, 132) token
        # table -- k row-major per token with the fp32 scale INTERLEAVED at
        # each row tail, unlike the page pools above whose scales sit
        # gathered at the page tail. block_tables carry token ids
        # directly, so no page/slot decomposition applies.
        i = tl.program_id(0)
        t0 = tl.program_id(1) * TT
        L = tl.load(lens_ptr + i)
        if t0 >= L:
            return  # window-out stays at the pre-zeroed 0.0
        t = t0 + tl.arange(0, TT)
        tmask = t < L
        tok = tl.load(bt_ptr + i * BT_STRIDE + t, mask=tmask, other=0)
        d = tl.arange(0, D)
        u = tl.load(
            pool_ptr + tok.to(tl.int64)[:, None] * 132 + d[None, :],
            mask=tmask[:, None],
            other=0,
        ).to(tl.int32)
        # e4m3fn decode (same integer bit math as the paged kernel above)
        sgn = (u >> 7) & 1
        e = (u >> 3) & 15
        m = u & 7
        pw = tl.where(e == 0, -6, e - 7)
        mant = tl.where(e == 0, m, m + 8).to(tl.float32) * 0.125
        two = ((pw + 127) << 23).to(tl.float32, bitcast=True)
        kf = tl.where(sgn == 1, -(mant * two), mant * two)
        acc = tl.zeros((TT,), tl.float32)
        for h in tl.static_range(H):
            qv = tl.load(q_ptr + (i * H + h) * D + d)
            wh = tl.load(w_ptr + i * H + h)
            acc += wh * tl.sum(kf * qv[None, :], axis=1)
        sf = tl.load(
            sf_ptr + tok.to(tl.int64) * 33 + 32, mask=tmask, other=0.0
        )
        tl.store(
            out_ptr + i.to(tl.int64) * T_ALL + t,
            acc * sf,
            mask=tmask & (t < T_ALL),
        )


def triton_paged_ps1_mqa_logits(
    q_fp8,
    kv_cache_fp8,
    weights,
    seq_lens,
    block_tables,
    max_seq_len,
    *,
    preshuffle,
    kv_block_size,
    **_ignored,
):
    """aiter paged-MQA face on the legacy page_size=1 pool (132B rows,
    interleaved row-tail scales). Mirrors mqa_logits_emu._paged_mqa_logits
    exactly (window-out = 0.0); e4m3fn-only -- the fnuz convention keeps
    the emu (torch decode is convention-correct by construction)."""
    from .dsa_mqa_logits_emu import _as_view, _fp8_view_dtype

    if preshuffle or kv_block_size != 1:
        raise NotImplementedError(
            "ps1 triton arm is the legacy page_size=1 no-preshuffle pool only"
        )
    if _fp8_view_dtype() is not torch.float8_e4m3fn:
        raise NotImplementedError("fnuz pool: ps1 triton arm is e4m3fn-only")
    q_flat = (
        q_fp8.reshape(-1, q_fp8.shape[-2], q_fp8.shape[-1])
        if q_fp8.dim() == 4
        else q_fp8
    )
    q_b = _as_view(q_flat, _fp8_view_dtype()).to(torch.float32).contiguous()
    N, H, D = q_b.shape
    pool = (
        kv_cache_fp8.reshape(kv_cache_fp8.shape[0], -1).contiguous()
    )
    if pool.dtype == torch.int8:
        pool = pool.view(torch.uint8)
    if pool.shape[1] != 132:
        raise NotImplementedError(
            f"ps1 pool expects 132B rows, got {pool.shape[1]}"
        )
    sf = pool.view(torch.float32)  # (T, 33)
    w = weights.reshape(N, H).to(torch.float32).contiguous()
    ctx = seq_lens.to(torch.int32).reshape(-1).contiguous()
    bt = block_tables.to(torch.int32).contiguous()
    T_all = min(bt.shape[1], int(max_seq_len))
    out = torch.zeros((N, int(max_seq_len)), dtype=torch.float32, device=q_b.device)
    if N == 0 or T_all == 0:
        return out
    TT = 128
    _paged_ps1_mqa_logits_kernel[(N, triton.cdiv(T_all, TT))](
        q_b, w, pool, sf, bt, ctx, out,
        T_all, bt.stride(0),
        H=H, D=D, TT=TT,
        num_warps=4,
    )
    return out


def triton_ragged_mqa_logits(q_fp8, kv, kv_scales, weights, cu_starts, cu_ends):
    """deep_gemm.fp8_mqa_logits face (ragged): out [N,T] fp32, -inf outside
    each row's [ks, ke). Same contract as mqa_logits_emu._ragged_fp8_mqa_logits
    (which stays the golden + the HCU_MQA_LOGITS_IMPL!=triton path)."""
    from .dsa_mqa_logits_emu import _as_view, _fp8_view_dtype

    k_fp8, scale = kv, kv_scales
    if k_fp8.dtype in (torch.uint8, torch.int8):
        k_u8 = k_fp8.contiguous().view(torch.uint8)
    else:
        k_u8 = _as_view(k_fp8.contiguous(), _fp8_view_dtype()).view(torch.uint8)
    T, D = k_u8.shape
    if _fp8_view_dtype() is not torch.float8_e4m3fn:
        raise NotImplementedError("fnuz pool: ragged triton arm is e4m3fn-only")
    q_flat = q_fp8.reshape(-1, q_fp8.shape[-2], q_fp8.shape[-1]) if q_fp8.dim() == 4 else q_fp8
    q_b = _as_view(q_flat, _fp8_view_dtype()).to(torch.float32).contiguous()
    N, H, _ = q_b.shape
    w = weights.reshape(N, H).to(torch.float32).contiguous()
    ks = cu_starts.to(torch.int32).reshape(-1).contiguous()
    ke = cu_ends.to(torch.int32).reshape(-1).contiguous()
    sf = kv_scales.reshape(-1).to(torch.float32).contiguous()
    out = torch.empty((N, T), dtype=torch.float32, device=q_b.device)
    if N == 0 or T == 0:
        return out
    TT = 256
    _ragged_mqa_logits_kernel[(N, triton.cdiv(T, TT))](
        q_b,
        w,
        k_u8,
        sf,
        ks,
        ke,
        out,
        T,
        H=H,
        D=D,
        TT=TT,
        num_warps=4,
    )
    return out
