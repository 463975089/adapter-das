# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Preshuffle-aware triton gather for the DSA index-K cache (kpool read).

The aiter ``cp_gather_indexer_k_quant_cache(preshuffle=True)`` kernel
illegal-addresses / returns zeros on gfx93x (probe 2026-10-15), so the read
face (``GetKAndS.aiter``) used a torch emulation. That emu is bit-correct but
does per-row Python indexing; this module is the triton tier the priority
ladder asks for: same 16-tile interleave map, same outputs, one launch.

Layout (production preshuffle / kpool=4 world)::

    page (page_size=64 slots) = K region [0, page_size*D) in 16-tile order
                                + scale region [page_size*D, page_size*(D+S))
    K byte for slot s, dim c = page*page_bytes
                               + (s//16)*(16*D) + (c//16)*(16*16)
                               + (s%16)*16 + (c%16)
    scale bytes for slot s   = page*page_bytes + page_size*D + s*S

``preshuffle=False`` keeps the row-major (kpool=1 / fused_store) layout:
``page*page_bytes + s*D + c``.

The kernel reads page/cu arrays as int32 (the int64 scalar-load pattern is the
one that mis-compiles on gfx938 in the writer family) and does all byte
arithmetic in int64, so large pools do not overflow. It is a drop-in for the
aiter call: ``(kv_cache, k_out, scale_out, page_indices, cu_seq_lens,
preshuffle=True)`` with caller-owned destination buffers.

ABI (identical to aiter's kernel / ``GetKAndS.aiter``): ``page_indices`` is
the request-major page table ``[n_req, pages_per_req]``; ``cu_seq_lens`` is
``[n_req + 1]`` cumulative pooled lengths. Token ``g`` of request ``b`` has
in-request index ``g - cu[b]``, page ``page_indices[b, idx // page_size]``
and slot ``idx % page_size`` -- one row is a request, not a page.
"""

from __future__ import annotations

import torch

_TILE = 16  # INDEXER_K_CACHE_PRESHUFFLE_TILE


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _gather_kernel():
    # Import lazily: the module is imported by CPU tests / L0 where triton may
    # be absent. The kernel is cached on first GPU use.
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL

    import triton
    import triton.language as tl

    @triton.jit
    def _kpool_gather_kernel(
        buf_ptr,
        k_out_ptr,
        s_out_ptr,
        pages_ptr,
        cu_ptr,
        page_size: tl.constexpr,
        d: tl.constexpr,
        s_bytes: tl.constexpr,
        page_bytes: tl.constexpr,
        pages_per_req: tl.constexpr,
        BLOCK_D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        PRESHUFFLE: tl.constexpr,
    ):
        b = tl.program_id(0)
        tok = tl.program_id(1)
        blk = tl.program_id(2)

        start = tl.load(cu_ptr + b).to(tl.int64)
        end = tl.load(cu_ptr + b + 1).to(tl.int64)
        if tok >= end - start:
            return

        page_idx = tok // page_size
        if page_idx >= pages_per_req:
            return
        page = tl.load(pages_ptr + b * pages_per_req + page_idx).to(tl.int64)
        slot = tok % page_size
        page_base = page * page_bytes

        c = blk * BLOCK_D + tl.arange(0, BLOCK_D)
        cmask = c < d
        if PRESHUFFLE:
            off = (
                page_base
                + (slot // 16) * (16 * d)
                + (c // 16) * (16 * 16)
                + (slot % 16) * 16
                + (c % 16)
            )
        else:
            off = page_base + slot * d + c
        vals = tl.load(buf_ptr + off, mask=cmask, other=0)
        row = (start + tok) * d
        tl.store(k_out_ptr + row + c, vals, mask=cmask)

        if blk == 0:
            si = tl.arange(0, S_BLOCK)
            smask = si < s_bytes
            s_off = page_base + page_size * d + slot * s_bytes + si
            sv = tl.load(buf_ptr + s_off, mask=smask, other=0)
            s_row = (start + tok) * s_bytes
            tl.store(s_out_ptr + s_row + si, sv, mask=smask)

    _KERNEL = _kpool_gather_kernel
    return _KERNEL


_KERNEL = None


def cp_gather_indexer_k_quant_cache_triton(
    kv_cache: torch.Tensor,
    k_out: torch.Tensor,
    scale_out: torch.Tensor,
    page_indices: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    preshuffle: bool = True,
    max_seq_len: int | None = None,
) -> None:
    """Gather index-K bytes + fp32 scales out of the paged pool.

    ``kv_cache`` is the uint8 page buffer (``[pages, page_bytes]`` or
    ``[pages, page_size, D+S]``); ``k_out`` / ``scale_out`` are caller-owned
    uint8 destinations; ``page_indices`` is the request-major table
    ``[n_req, pages_per_req]`` (a flat list is accepted for n_req == 1);
    ``cu_seq_lens`` is the int32 cumulative length array (``[n_req+1]``).
    """
    if page_indices.numel() == 0 or k_out.numel() == 0:
        return

    buf = kv_cache.reshape(-1).view(torch.uint8)
    d = int(k_out.shape[1])
    s_bytes = int(scale_out.shape[1])
    if kv_cache.dim() == 3:
        page_size = int(kv_cache.shape[1])
    elif kv_cache.dim() == 2:
        page_size = int(kv_cache.shape[1]) // (d + s_bytes)
    else:
        page_size = 0
    if page_size <= 0 or d <= 0 or s_bytes <= 0:
        raise ValueError("invalid kpool gather shapes")
    if s_bytes != 4:
        raise NotImplementedError(
            f"kpool gather expects one fp32 scale per token, got {s_bytes} bytes"
        )
    page_bytes = page_size * (d + s_bytes)

    pages = page_indices.to(torch.int32).contiguous()
    if pages.dim() == 1:
        pages = pages.view(1, -1)
    cu = cu_seq_lens.to(torch.int32).contiguous()
    n_req = int(cu.numel()) - 1
    if n_req <= 0:
        return
    if pages.shape[0] != n_req:
        raise ValueError(
            f"page_indices batch {pages.shape[0]} != cu_seq_lens requests {n_req}"
        )
    pages_per_req = int(pages.shape[1])
    lens = cu[1:] - cu[:n_req]
    span = int(lens.max().item()) if lens.numel() else 0
    if int(cu[-1]) > int(k_out.shape[0]):
        raise ValueError(
            f"cu_seq_lens total {int(cu[-1])} > k_out rows {int(k_out.shape[0])}"
        )
    # cu_seq_lens is authoritative (max_seq_len is only an allocation upper
    # bound in some callers): grid over the real span so every dst row is
    # written, and let the kernel re-guard per request.
    if span <= 0:
        return
    max_seq_len = span

    grid = (n_req, int(max_seq_len), (d + 127) // 128)
    _gather_kernel()[grid](
        buf,
        k_out.view(torch.uint8),
        scale_out.view(torch.uint8),
        pages,
        cu,
        page_size=page_size,
        d=d,
        s_bytes=s_bytes,
        page_bytes=page_bytes,
        pages_per_req=pages_per_req,
        BLOCK_D=128,
        S_BLOCK=_next_pow2(max(s_bytes, 1)),
        PRESHUFFLE=bool(preshuffle),
    )
