# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""gfx938-safe torch emulations for the DSA kpool index-K write kernels.

The DTK triton stock mis-compiles the int64 scalar loads inside the real
kernels for this shape family (probe on g53-tp8 GPU4, 2026-10-15): with
loc=[64,128,192,256] the row-signature scales land at page0, token offsets
16/32/48/64 — every value read through `tl.load(ptr + row)` arrives exactly
divided by four, so all writes collapse into the head of the pool and the
index-K cache stays zero wherever the MQA emu reads it (kv_abs=0 → zero
logits → degenerate topk → the GLM-5.3 first-token flatline). Mini-probes of
the same load expressions compile fine; it takes the full kernel body to
trigger, mirroring the causal_conv1d initial-state read-dead signature. All
three entrypoints in this module read the same int64 arrays:
kpool_softmax_rotate_write_cache (prefill pooled write), 
kpool_decode_update_and_maybe_write_cache (decode tail + pooled write) and
scatter_kpool_tail_updates (extend tail write), so the whole family moves to
mathematically identical torch code behind HCU_KPOOL_WRITE_EMU (default on).

The torch path keeps bit-level parity with the kernel numerics that DO work
(evidence: RETURN_COMPRESSED output is fully healthy on gfx938 — softmax,
hadamard and the fp8 quantize all match): bf16 round-trips before/after the
hadamard, absmax scale (or its ceil log2 rounding), e4m3 clamp/saturate and
the 16-wide preshuffle of `_kpool_cache_k_offsets` are reproduced exactly;
tests/kernels/test_kpool_write_gfx938.py pins the triton path red and the
emu green against an independent row-stepped reference.

Read side (2026-11): this module also hosts the tier switch for
``GetKAndS.aiter``. The default on gfx93x is the preshuffle-aware triton
gather in ``kpool_read_triton.py``; the torch gather below is the explicit
``HCU_KPOOL_READ_EMU=1`` A/B fallback (the aiter cp_gather kernel is dead on
this arch).
"""

from __future__ import annotations

import torch

_INDEX_HEAD_DIM = 128
_FP8_MAX = 448.0
_HADM_NORM = 0.08838834764831845  # 1/sqrt(128), same constant as the kernel
_TILE = 16  # INDEXER_K_CACHE_PRESHUFFLE_TILE; layout is fixed by the aiter cache


def _hadamard128() -> torch.Tensor:
    """Normalized 128-point Walsh-Hadamard matrix == the kernel's 7 butterfly stages."""
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < 128:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return h * _HADM_NORM


_H128: torch.Tensor | None = None


def _softmax_rotate_quant(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    round_scale: bool,
):
    """The shared healthy math: pool-softmax weight, hadamard, fp8 quantize.

    slot_k [N, P, D] bf16, slot_score [N, P, D], ape [P, D] fp32 ->
    (q8 [N, D] e4m3, scale [N] fp32)
    """
    global _H128
    if _H128 is None or _H128.device != slot_k.device:
        _H128 = _hadamard128().to(slot_k.device)
    s = slot_score.float() + ape.float().unsqueeze(0)
    m = s.max(dim=1, keepdim=True).values
    p = torch.exp(s - m)
    x = (slot_k.float() * p).sum(dim=1) / p.sum(dim=1)
    x = x.to(torch.bfloat16).float()
    x = (x @ _H128).to(torch.bfloat16).float()
    absmax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(absmax / _FP8_MAX)))
    else:
        scale = absmax / _FP8_MAX
    q = (x / scale).clamp_(-_FP8_MAX, _FP8_MAX)
    return q.to(torch.float8_e4m3fn), scale.reshape(-1)


def _scatter_rows(
    buf: torch.Tensor,
    q8: torch.Tensor,
    scale: torch.Tensor,
    loc: torch.Tensor,
    slots_per_page: int,
) -> None:
    """Replicate `_kpool_cache_k_offsets` + the two stores, fully vectorized."""
    tile = _TILE
    d = q8.shape[1]
    buf_numel_per_page = buf.shape[1]
    page = torch.div(loc, slots_per_page, rounding_mode="floor").unsqueeze(1)
    tok = (loc % slots_per_page).unsqueeze(1)
    col = torch.arange(d, device=buf.device, dtype=torch.int64).unsqueeze(0)
    offsets = (
        page * buf_numel_per_page
        + torch.div(tok, tile, rounding_mode="floor") * (tile * d)
        + torch.div(col, tile, rounding_mode="floor") * (tile * tile)
        + (tok % tile) * tile
        + (col % tile)
    )
    buf.view(torch.float8_e4m3fn).view(-1)[offsets.reshape(-1)] = (
        q8.to(torch.float8_e4m3fn).reshape(-1)
    )
    s_offsets = (
        page * (buf_numel_per_page // 4)
        + slots_per_page * d // 4
        + tok
    )
    buf.view(torch.float32).view(-1)[s_offsets.view(-1)] = scale


def kpool_softmax_rotate_write_cache_emu(
    pool,
    buf: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor | None = None,
    round_scale: bool = False,
    return_compressed: bool = False,
    write_cache: bool = True,
):
    assert slot_k.ndim == 3 and slot_score.shape == slot_k.shape
    assert ape.shape == slot_k.shape[1:]
    assert slot_k.shape[2] == _INDEX_HEAD_DIM
    assert slot_k.dtype == torch.bfloat16
    assert buf.dtype == torch.uint8
    assert loc.dtype == torch.int64
    if slot_k.shape[0] == 0:
        if return_compressed:
            e = slot_k.device
            return (
                torch.empty((0, slot_k.shape[2]), dtype=torch.float8_e4m3fn, device=e),
                torch.empty((0,), dtype=torch.float32, device=e),
            )
        return None

    q8, scale = _softmax_rotate_quant(slot_k, slot_score, ape, round_scale)
    if write_cache:
        _scatter_rows(buf, q8, scale, loc.contiguous(), pool.page_size)
    if return_compressed:
        return q8, scale
    return None


def kpool_decode_update_and_maybe_write_cache_emu(
    pool,
    buf: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    block_tables: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    round_scale: bool = False,
) -> None:
    batch = key.shape[0]
    if batch == 0:
        return
    n_req = tail_k.shape[0]
    tail_size = tail_k.shape[1]
    pool_size = pool.index_kpool
    slots_per_page = pool.slots_per_page

    for row in range(batch):
        req_raw = int(req_pool_indices[row])
        req_valid = 0 <= req_raw < n_req
        req = min(max(req_raw, 0), n_req - 1)
        pos = int(positions[row])
        seq_len = int(seq_lens[row])
        cache_loc = int(out_cache_loc[row])
        pos_valid = (
            req_valid and cache_loc != 0 and 0 <= pos < seq_len
        )
        slot = max(pos, 0) % pool_size
        phys = max(pos, 0) % tail_size
        if pos_valid:
            tail_k[req, phys] = key[row]
            tail_score[req, phys] = slot_score[row]
        if pos_valid and slot == pool_size - 1:
            start = pos - slot
            ks = torch.empty(pool_size, tail_k.shape[2], device=key.device)
            ss = torch.empty(pool_size, tail_k.shape[2], device=key.device)
            for pslot in range(pool_size):
                p = (start + pslot) % tail_size
                ks[pslot] = key[row] if pslot == slot else tail_k[req, p]
                ss[pslot] = slot_score[row] if pslot == slot else tail_score[req, p]
            q8, scale = _softmax_rotate_quant(
                ks.unsqueeze(0).to(torch.bfloat16),
                ss.unsqueeze(0),
                ape,
                round_scale,
            )
            pool_id = pos // pool_size
            page_group = pool_id // slots_per_page
            table_row = min(max(page_group * pool_size, 0), block_tables.shape[1] - 1)
            loc_page = int(block_tables[row, table_row])
            loc1 = torch.tensor(
                [loc_page * slots_per_page + pool_id % slots_per_page],
                device=buf.device,
                dtype=torch.int64,
            )
            _scatter_rows(buf, q8, scale, loc1, slots_per_page)


def scatter_kpool_tail_updates_emu(
    pool,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
) -> None:
    if req_pool_idx.shape[0] == 0:
        return
    tail_size = tail_k.shape[1]
    for row in range(req_pool_idx.shape[0]):
        req = int(req_pool_idx[row])
        dst = int(dst_logical_start[row])
        src = int(chunk_src_start[row])
        for slot in range(int(n_write[row])):
            tail_k[req, (dst + slot) % tail_size] = chunk_k[src + slot]
            tail_score[req, (dst + slot) % tail_size] = chunk_score[src + slot]


def cp_gather_indexer_k_quant_cache_emu(
    kv_cache: torch.Tensor,
    k_out: torch.Tensor,
    scale_out: torch.Tensor,
    page_indices: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    preshuffle: bool = True,
) -> None:
    """torch twin of aiter's cp_gather_indexer_k_quant_cache.

    ABI is aiter's / ``GetKAndS.aiter``'s: ``page_indices`` is the request-
    major page table ``[n_req, pages_per_req]`` and ``cu_seq_lens`` is
    ``[n_req + 1]`` cumulative pooled lengths. A token of request r maps to
    ``page_indices[r, (token - cu[r]) // page_size]``, in-page slot
    ``(token - cu[r]) % page_size`` (aiter's kernel walks the row, not one
    page per row).

    The aiter GPU kernel illegal-addresses / returns zeros on gfx938 (probe
    GPU4 2026-10-15), zeroing the whole prefill topk score. Same byte
    economics as the writer: k_out rows gather D bytes through the 16-wide
    preshuffle map (row-major when ``preshuffle=False``), scale_out rows
    copy the 4 raw fp32 bytes next to the k region of the page.
    """
    if page_indices.numel() == 0 or k_out.numel() == 0:
        return
    page_size = int(kv_cache.shape[1])
    d = int(k_out.shape[1])
    s_bytes = int(scale_out.shape[1])
    if s_bytes != 4:
        raise NotImplementedError(
            f"kpool gather emu expects one fp32 scale per token, got {s_bytes} bytes"
        )
    page_bytes = page_size * (d + s_bytes)
    flat = kv_cache.reshape(-1).view(torch.uint8)
    dev = kv_cache.device

    pages = page_indices.to(torch.int64)
    if pages.dim() == 1:
        pages = pages.view(1, -1)
    cu = cu_seq_lens.to(torch.int64)
    n_req = int(cu.numel()) - 1
    if n_req <= 0:
        return
    if pages.shape[0] != n_req:
        raise ValueError(
            f"page_indices batch {pages.shape[0]} != cu_seq_lens requests {n_req}"
        )
    lens = cu[1:] - cu[:n_req]
    total = int(cu[-1])
    if total > int(k_out.shape[0]):
        raise ValueError(
            f"cu_seq_lens total {total} > k_out rows {int(k_out.shape[0])}"
        )

    tok = torch.arange(total, device=dev, dtype=torch.int64)
    req = torch.repeat_interleave(torch.arange(n_req, device=dev), lens)
    inreq = tok - cu[req]
    page = pages[req, torch.div(inreq, page_size, rounding_mode="floor")]
    slot = inreq % page_size
    tile = _TILE
    col = torch.arange(d, device=dev, dtype=torch.int64).unsqueeze(0)
    if preshuffle:
        slot_off = (
            torch.div(slot, tile, rounding_mode="floor").unsqueeze(1) * (tile * d)
            + torch.div(col, tile, rounding_mode="floor") * (tile * tile)
            + (slot % tile).unsqueeze(1) * tile
            + (col % tile)
        )
    else:
        slot_off = slot.unsqueeze(1) * d + col
    byte_off = page.unsqueeze(1) * page_bytes + slot_off
    k_out.view(torch.uint8).view(-1, d)[:total].copy_(flat[byte_off])
    s_off = page * page_bytes + page_size * d + slot * s_bytes
    scale_out.view(torch.uint8).view(-1, s_bytes)[:total].copy_(
        flat[s_off.unsqueeze(1) + torch.arange(s_bytes, device=dev).unsqueeze(0)]
    )


def install_patch() -> None:
    import sglang.srt.layers.attention.dsa.kpool_fp8_index as kfi

    # keep the stock kernels reachable for the red gate in
    # tests/kernels/test_kpool_write_gfx938.py (DTK-fix watchpoint)
    kfi._hcu_stock_softmax_rotate = kfi.kpool_softmax_rotate_write_cache
    kfi._hcu_stock_decode_update = kfi.kpool_decode_update_and_maybe_write_cache
    kfi._hcu_stock_scatter = kfi.scatter_kpool_tail_updates

    kfi.kpool_softmax_rotate_write_cache = kpool_softmax_rotate_write_cache_emu
    kfi.kpool_decode_update_and_maybe_write_cache = (
        kpool_decode_update_and_maybe_write_cache_emu
    )
    kfi.scatter_kpool_tail_updates = scatter_kpool_tail_updates_emu


def install_read_patch(mode: str = "triton") -> None:
    """Route GetKAndS.aiter to the selected read tier.

    ``triton`` (default on gfx93x): preshuffle-aware triton gather
    (kpool_read_triton.py). ``emu``: torch reference (bit-level A/B). The
    upstream aiter kernel is left in place by the caller on non-gfx93
    platforms.
    """
    # 替代档: kpool_read_triton.py（本文件 torch gather emu 为显式回退）
    try:
        # 替代档: 新 sglang 走 kvcache 门面（dsa_index_k_cache 为其模块体——门面无非此档）
        from sglang.kernels.ops.kvcache import dsa_index_k_cache as iba
    except ImportError:
        # 替代档: kpool_read_triton.py（旧 sglang 树走 index_buf_accessor）
        from sglang.kernels.ops.attention.dsa import index_buf_accessor as iba

    if mode == "triton":
        iba.GetKAndS.aiter = _aiter_triton_wrapper
    elif mode == "emu":
        iba.GetKAndS.aiter = _aiter_emu_wrapper
    else:
        raise ValueError(f"unknown kpool read mode {mode!r}")


def _aiter_triton_wrapper(cls, *args, **kwargs):
    from .kpool_read_triton import cp_gather_indexer_k_quant_cache_triton

    pool, buf, page_indices, seq_len_tensor, seq_len_sum, max_seq_len = args
    # installed sglang versions disagree on the attribute name (`page_size`
    # on the older index_buf_accessor, `index_page_size` after the page-64
    # refactor); both are 64 on the preshuffle path.
    page_size = getattr(pool, "index_page_size", None) or pool.page_size
    index_head_dim = pool.index_head_dim
    scale_elems = index_head_dim // pool.quant_block_size
    s_bytes = scale_elems * 4
    kv_cache = buf.view(-1, page_size, index_head_dim + s_bytes)
    dst_k = torch.empty(
        (seq_len_sum, index_head_dim), dtype=torch.uint8, device=buf.device
    )
    dst_scale = torch.empty(
        (seq_len_sum, s_bytes), dtype=torch.uint8, device=buf.device
    )
    cu_seq_lens = torch.zeros(
        seq_len_tensor.shape[0] + 1, dtype=torch.int32, device=buf.device
    )
    torch.cumsum(seq_len_tensor.to(torch.int32), dim=0, out=cu_seq_lens[1:])
    cp_gather_indexer_k_quant_cache_triton(
        kv_cache,
        dst_k,
        dst_scale,
        page_indices,
        cu_seq_lens,
        preshuffle=True,
        max_seq_len=max_seq_len,
    )
    fp8 = torch.float8_e4m3fn
    return dst_k.view(fp8), dst_scale


def _aiter_emu_wrapper(cls, *args, **kwargs):
    if not getattr(_aiter_emu_wrapper, "_hit", 0):
        _aiter_emu_wrapper._hit = 1
        print("[HCU kprd] aiter-arm -> torch gather emu", flush=True)
    elif getattr(_aiter_emu_wrapper, "_hit", 0) < 3:
        _aiter_emu_wrapper._hit += 1
        print("[HCU kprd] called", flush=True)
    else:
        _aiter_emu_wrapper._hit += 1
    pool, buf, page_indices, seq_len_tensor, seq_len_sum, max_seq_len = args
    page_size = pool.page_size
    index_head_dim = pool.index_head_dim
    quant_block_size = pool.quant_block_size
    scale_elems = index_head_dim // quant_block_size
    kv_cache = buf.view(-1, page_size, index_head_dim + scale_elems * 4).view(
        torch.float8_e4m3fn
    )
    dst_k = torch.empty(
        (seq_len_sum, index_head_dim), dtype=torch.uint8, device=buf.device
    )
    dst_scale = torch.empty(
        (seq_len_sum, scale_elems * 4), dtype=torch.uint8, device=buf.device
    )
    cu_seq_lens = torch.zeros(
        seq_len_tensor.shape[0] + 1, dtype=torch.int32, device=buf.device
    )
    torch.cumsum(seq_len_tensor.to(torch.int32), dim=0, out=cu_seq_lens[1:])
    cp_gather_indexer_k_quant_cache_emu(
        kv_cache,
        dst_k,
        dst_scale,
        page_indices,
        cu_seq_lens,
        preshuffle=True,
    )
    fp8 = torch.float8_e4m3fn
    return dst_k.view(fp8), dst_scale
