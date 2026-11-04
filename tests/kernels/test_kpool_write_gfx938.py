"""Regression gate for the DSA kpool index-K write path (gfx938).

The 2026-10-15 wave around these kernels ended with the *probe* being wrong:
hand-built 4-D buffers (pages, 64, 1, 384) fed shape[1]=64 to the launcher as
BUF_NUMEL_PER_PAGE, collapsing every write into the head of the tiny probe
pool — the stock triton kernels are healthy on correct 2-D (pages, PAGE*(4+D))
buffers. This file pins that health: the pooled write must land each row's
bytes and fp32 scale exactly where the oracle-compressed arm (RETURN_COMPRESSED
skips the loc scatter, hence also skips the load family under suspicion) says
they belong. If DTK ever regresses, this turns red.

HCU_KPOOL_WRITE_EMU arm: when enabled the emu must be bit-equal to the stock
kernels (same softmax/hadamard/e4m3 quantize math) on this 2-D layout.
"""

import os

import pytest
import torch

import sglang.srt.layers.attention.dsa.kpool_fp8_index as kfi

PAGE = 64  # slots per page == BLOCK_SIZE_K
D = 128
POOL = 4
TILE = 16


def _mkbuf(pages, dev):
    return torch.zeros(pages, PAGE * (4 + D), device=dev, dtype=torch.uint8)


class _Pool:
    page_size = PAGE
    index_page_size = PAGE
    index_head_dim = D
    index_kpool = POOL
    slots_per_page = PAGE


def _case(seed, n, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    pages = n + 2
    slot_k = torch.randn(n, POOL, D, generator=g, device=dev).to(torch.bfloat16)
    slot_score = torch.randn(n, POOL, D, generator=g, device=dev) * 0.7
    ape = torch.randn(POOL, D, generator=g, device=dev) * 0.1
    # collision-free loc: duplicate (page, tok) draws make the byte outcome
    # write-order dependent (stock kernel vs row-by-row reference can pick
    # different winners) -- seed 2 of the old randint draw hit exactly that
    # (rerun3 forensics 2026-10-18). randperm keeps the same value range.
    loc = (
        torch.randperm((pages - 1) * PAGE - 1, generator=g, device=dev)[:n]
        .to(torch.int64)
        + 1
    )
    return pages, slot_k, slot_score, ape, loc


def _oracle(slot_k, slot_score, ape):
    """Healthy arm: loc-scatter-free RETURN_COMPRESSED output as ground truth."""
    stock = getattr(kfi, "_hcu_stock_softmax_rotate", None) or (
        kfi.kpool_softmax_rotate_write_cache
    )
    q8, scale = stock(
        pool=_Pool(),
        buf=_mkbuf(8, slot_k.device),
        slot_k=slot_k,
        slot_score=slot_score,
        ape=ape,
        loc=torch.zeros(slot_k.shape[0], device=slot_k.device, dtype=torch.int64),
        write_mask=None,
        round_scale=False,
        return_compressed=True,
        write_cache=False,
    )
    return q8.float(), scale


def _scatter_ref(q8, scale, loc, pages, dev):
    """Independent row-by-row scatter onto the true 2-D layout."""
    buf = _mkbuf(pages, dev)
    bf8 = buf.view(torch.float8_e4m3fn)
    bf32 = buf.view(torch.float32)
    page_bytes = PAGE * (4 + D)
    for r in range(q8.shape[0]):
        page = int(loc[r]) // PAGE
        tok = int(loc[r]) % PAGE
        for c in range(D):
            off = (
                page * page_bytes
                + (tok // TILE) * (TILE * D)
                + (c // TILE) * (TILE * TILE)
                + (tok % TILE) * TILE
                + (c % TILE)
            )
            bf8.view(-1)[off] = q8[r, c].to(torch.float8_e4m3fn)
        bf32.view(-1)[
            page * (page_bytes // 4) + PAGE * D // 4 + tok
        ] = scale[r]
    return buf


def _paged_write(slot_k, slot_score, ape, loc, pages, dev, fn):
    buf = _mkbuf(pages, dev)
    fn(
        pool=_Pool(), buf=buf, slot_k=slot_k, slot_score=slot_score,
        ape=ape, loc=loc, write_mask=None, round_scale=False,
        return_compressed=False, write_cache=True,
    )
    return buf


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu")
def test_stock_paged_scatter_lands_at_loc():
    """Health pin: stock kernel scatter == oracle bytes at the real loc.

    Failed (as a deliberate red gate) during the false-positive window when
    probes ran with a wrong buffer layout; with the true 2-D layout this is
    the standing regression signal for the DTK triton stack.
    """
    dev = "cuda"
    for seed in range(3):
        pages, slot_k, slot_score, ape, loc = _case(seed, 4, dev)
        q8o, scaleo = _oracle(slot_k, slot_score, ape)
        buf = _paged_write(
            slot_k, slot_score, ape, loc, pages, dev,
            getattr(kfi, "_hcu_stock_softmax_rotate", None)
            or kfi.kpool_softmax_rotate_write_cache,
        )
        ref = _scatter_ref(q8o, scaleo, loc, pages, dev)
        assert torch.equal(
            buf.view(torch.uint8), ref.view(torch.uint8)
        ), f"stock kpool scatter misplaced bytes at seed {seed}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu")
def test_gather_read_roundtrip():
    """Write emu scatter -> GetKAndS.aiter emu must reproduce the written
    rows bit-for-bit (k bytes and fp32 scales), incl. cross-page and partial
    requests. Guards the gfx938 torch gather (aiter read kernel is dead:
    illegal address / zero output)."""
    from hcu_adapter import kpool_write_emu as kw

    dev = "cuda"
    n_pages, n_tok = 4, 3 * PAGE
    buf = torch.zeros(n_pages, PAGE * (4 + D), device=dev, dtype=torch.uint8)
    g = torch.Generator(device=dev).manual_seed(3)
    q = torch.randn(n_tok, D, generator=g, device=dev).to(torch.bfloat16).to(torch.float8_e4m3fn)
    sc = torch.rand(n_tok, generator=g, device=dev) * 0.5
    loc = torch.arange(n_tok, device=dev, dtype=torch.int64)
    kw._scatter_rows(buf, q, sc, loc, PAGE)
    kv = buf.view(-1, PAGE, D + 4).view(torch.float8_e4m3fn)

    dst_k = torch.empty(n_tok, D, dtype=torch.uint8, device=dev)
    dst_s = torch.empty(n_tok, 4, dtype=torch.uint8, device=dev)
    kw.cp_gather_indexer_k_quant_cache_emu(
        kv, dst_k, dst_s,
        torch.tensor([0, 1, 2], dtype=torch.int32, device=dev),
        torch.tensor([0, PAGE, 2 * PAGE, 3 * PAGE], dtype=torch.int32, device=dev),
        True,
    )
    assert torch.equal(dst_k, q.view(torch.uint8)), "gather k mismatch"
    assert torch.allclose(
        dst_s.view(torch.float32).view(-1), sc, atol=1e-6, rtol=0
    ), "gather scale mismatch"

    dst_k2 = torch.empty(7, D, dtype=torch.uint8, device=dev)
    dst_s2 = torch.empty(7, 4, dtype=torch.uint8, device=dev)
    kw.cp_gather_indexer_k_quant_cache_emu(
        kv, dst_k2, dst_s2,
        torch.tensor([1], dtype=torch.int32, device=dev),
        torch.tensor([0, 7], dtype=torch.int32, device=dev),
        True,
    )
    assert torch.equal(dst_k2, q[64:71].view(torch.uint8)), "partial gather mismatch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu")
def test_emu_bit_equal_stock():
    """When HCU_KPOOL_WRITE_EMU=1, the emu must match the stock kernels byte
    for byte on this layout — correctness keep-out for any future enablement."""
    from hcu_adapter import kpool_write_emu

    kpool_write_emu.install_patch()
    dev = "cuda"
    for seed in range(3):
        pages, slot_k, slot_score, ape, loc = _case(seed + 20, 5, dev)
        stock = kfi._hcu_stock_softmax_rotate
        q8o, scaleo = _oracle(slot_k, slot_score, ape)
        buf_s = _paged_write(
            slot_k, slot_score, ape, loc, pages, dev, stock
        )
        buf_e = _paged_write(
            slot_k, slot_score, ape, loc, pages, dev,
            kfi.kpool_softmax_rotate_write_cache,
        )
        assert torch.equal(buf_s.view(torch.uint8), buf_e.view(torch.uint8)), (
            f"emu != stock at seed {seed}"
        )
        q8, scale = kfi.kpool_softmax_rotate_write_cache(
            pool=_Pool(), buf=_mkbuf(8, dev), slot_k=slot_k,
            slot_score=slot_score, ape=ape, loc=loc, round_scale=False,
            return_compressed=True, write_cache=False,
        )
        assert q8.dtype == torch.float8_e4m3fn
        assert torch.equal(q8.view(torch.uint8), q8o.to(torch.float8_e4m3fn).view(torch.uint8))
        assert torch.equal(scale, scaleo)
