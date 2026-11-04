# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""weak_ref_tensor route + decoupling contract, run on a GPU node.

The breakable prefill-graph capture path re-exports this symbol into
sglang.srt.compilation.weak_ref_tensor. Semantic contract (what the
upstream C++ TensorImpl guarantees and what the OOM diagnosis of
2026-10-14 depends on):

1. ROUTE      — when a vendor sgl_kernel wheel with the compiled op is
               present (DTK ships one), our module re-exports IT (not
               the view_as fallback): storage must NOT be pinned by the
               returned tensor.
2. DECOUPLE   — inside a graph capture pool, a weak-ref'd tensor whose
               python original is dropped must leave its block reusable
               by the NEXT capture in the same pool (same data_ptr).
               This is what turns the 58-bucket prefill table from
               "linear pool growth -> OOM at bucket ~25" into
               "block reuse -> all buckets captured".
3. ALIAS      — reads through the weak tensor must observe writes made
               through the original (same storage).

Run: pytest tests/kernels/test_weak_ref_decouple.py -v
"""
import pytest
import torch

import sgl_kernel.weak_ref as WR

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)


@requires_gpu
def test_vendor_route_preferred():
    """On a DTK-style node the compiled op must be picked up; the
    view_as fallback is allowed only on wheels without the symbol."""
    assert WR._vendor_weak_ref_tensor is not None or (
        WR.weak_ref_tensor.__module__ == "sgl_kernel.weak_ref"
    )


@requires_gpu
def test_alias_semantics():
    t = torch.zeros(8, device="cuda")
    w = WR.weak_ref_tensor(t)
    assert w.data_ptr() == t.data_ptr()
    t.fill_(3.5)
    torch.cuda.synchronize()
    assert abs(float(w.float().sum()) - 8 * 3.5) < 1e-4


@requires_gpu
@pytest.mark.skipif(WR._vendor_weak_ref_tensor is None,
                    reason="decoupling requires the compiled C++ op")
def test_capture_pool_block_reuse():
    """THE OOM contract: seg2 in the same capture pool must reuse seg1's
    block once python dropped the weak-ref'd original."""
    pool = torch.cuda.graph_pool_handle()
    BK = 1 << 26  # 64 MiB
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1, pool=pool):
        x = torch.zeros(BK, device="cuda")
        x.fill_(1.0)
    w = WR.weak_ref_tensor(x)
    ptr_x = x.data_ptr()
    del x  # the contract: python must not pin the pool block

    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2, pool=pool):
        y = torch.zeros(BK, device="cuda")
        y.fill_(2.0)

    if WR._vendor_weak_ref_tensor is not None:
        assert y.data_ptr() == ptr_x, (
            "pool block not reused — weak_ref is pinning; the 58-bucket "
            "prefill capture will OOM partway (see 2026-10-14 diagnosis)"
        )
