# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Parity gate for the preshuffle-aware index-K gather (kpool read).

The ABI is aiter's (also ``GetKAndS.aiter``'s): ``page_indices`` is the
request-major table ``[n_req, pages_per_req]`` and ``cu_seq_lens`` is
``[n_req + 1]`` cumulative pooled lengths -- one row is a request spanning
pages, not one page. The production pool stores K in the 16-tile interleave
order followed by fp32 scales in the page tail, so the triton gather
(``HCU_KPOOL_READ_EMU=0`` default on gfx93x) must reproduce the torch
reference byte-for-byte on that layout, and on the row-major layout behind
``preshuffle=False``.

Run on a GPU node:
    pytest -q tests/kernels/test_kpool_read_gfx938.py
"""

import pytest
import torch

from hcu_adapter.kpool_read_triton import cp_gather_indexer_k_quant_cache_triton
from hcu_adapter.kpool_write_emu import cp_gather_indexer_k_quant_cache_emu

PAGE = 64
D = 128
S = 4


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires HCU/CUDA device"
)


def _mkpool(pages, dev):
    return torch.randint(
        0, 256, (pages, PAGE * (D + S)), device=dev, dtype=torch.uint8
    )


def _cu(lens, dev):
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device=dev)
    torch.cumsum(torch.tensor(lens, dtype=torch.int32, device=dev), dim=0, out=cu[1:])
    return cu


def _ref(pool, pages, cu, preshuffle):
    """Independent token-stepped reference (Python loop, no shared helpers)."""
    flat = pool.reshape(-1)
    page_bytes = PAGE * (D + S)
    total = int(cu[-1])
    k_out = torch.zeros((total, D), dtype=torch.uint8, device=pool.device)
    s_out = torch.zeros((total, S), dtype=torch.uint8, device=pool.device)
    tile = 16
    for b in range(pages.shape[0]):
        start = int(cu[b])
        for t in range(int(cu[b + 1] - cu[b])):
            page = int(pages[b, t // PAGE])
            slot = t % PAGE
            base = page * page_bytes
            for c in range(D):
                if preshuffle:
                    off = (
                        base
                        + (slot // tile) * (tile * D)
                        + (c // tile) * (tile * tile)
                        + (slot % tile) * tile
                        + (c % tile)
                    )
                else:
                    off = base + slot * D + c
                k_out[start + t, c] = flat[off]
            s_out[start + t] = flat[
                base + PAGE * D + slot * S : base + PAGE * D + (slot + 1) * S
            ]
    return k_out, s_out


def _run(fn, pool, pages, cu, preshuffle):
    k_out = torch.empty((int(cu[-1]), D), dtype=torch.uint8, device=pool.device)
    s_out = torch.empty((int(cu[-1]), S), dtype=torch.uint8, device=pool.device)
    fn(
        pool.view(-1, PAGE, D + S),
        k_out,
        s_out,
        pages,
        cu,
        preshuffle=preshuffle,
    )
    return k_out, s_out


@requires_gpu
@pytest.mark.parametrize("preshuffle", [True, False])
def test_emu_matches_token_reference(preshuffle):
    """The explicit fallback must model the real request-major ABI."""
    torch.manual_seed(7)
    dev = "cuda"
    pool = _mkpool(6, dev)
    # request 0: 2 full pages + 1 token; request 1: 1 full page; request 2: 5
    pages = torch.tensor([[4, 1, 0], [3, 5, 2], [2, 0, 1]], device=dev, dtype=torch.int32)
    cu = _cu([2 * PAGE + 1, PAGE, 5], dev)

    ref_k, ref_s = _ref(pool, pages, cu, preshuffle)
    k_emu, s_emu = _run(
        cp_gather_indexer_k_quant_cache_emu, pool, pages, cu, preshuffle
    )
    assert torch.equal(k_emu, ref_k)
    assert torch.equal(s_emu, ref_s)


@requires_gpu
@pytest.mark.parametrize("preshuffle", [True, False])
def test_triton_matches_emu(preshuffle):
    """Triton tier (default on gfx93x) vs the torch reference, bit for bit."""
    torch.manual_seed(11)
    dev = "cuda"
    pool = _mkpool(6, dev)
    pages = torch.tensor([[4, 1, 0], [3, 5, 2], [2, 0, 1]], device=dev, dtype=torch.int32)
    cu = _cu([2 * PAGE + 1, PAGE, 5], dev)

    k_emu, s_emu = _run(
        cp_gather_indexer_k_quant_cache_emu, pool, pages, cu, preshuffle
    )
    k_tri, s_tri = _run(
        cp_gather_indexer_k_quant_cache_triton, pool, pages, cu, preshuffle
    )
    assert torch.equal(k_tri, k_emu)
    assert torch.equal(s_tri, s_emu)
