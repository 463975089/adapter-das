# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""weak_ref_tensor route + decoupling contract, run on a GPU node.

The breakable prefill-graph capture path re-exports this symbol into
sglang.srt.compilation.weak_ref_tensor. Semantic contract (what the
upstream C++ TensorImpl guarantees and what the OOM diagnosis of
2026-10-14 depends on):

1. ROUTE      — vendor common_ops.so when present (re-exported verbatim),
               else the torch non-owning route (admitted when its torch._C
               entry points exist), else view_as. On a device node with
               torch >= the entry points, view_as must not be chosen:
               storage must NOT be pinned by the returned tensor. Import
               must not initialize CUDA.
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
import gc

import pytest
import torch

import sgl_kernel.weak_ref as WR

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)


@requires_gpu
def test_route_never_view_as_on_device():
    """vendor > torch > view_as; with a device present at least the torch
    route must be admitted (view_as only when its entry points are gone)."""
    if WR._vendor_weak_ref_tensor is not None:
        assert WR.ROUTE == "vendor"
        assert WR.weak_ref_tensor is WR._vendor_weak_ref_tensor
    else:
        assert WR.ROUTE == "torch", WR._degrade_reason
        assert WR.weak_ref_tensor is WR._torch_weak_ref_impl


@requires_gpu
def test_alias_semantics():
    t = torch.zeros(8, device="cuda")
    w = WR.weak_ref_tensor(t)
    assert w.data_ptr() == t.data_ptr()
    t.fill_(3.5)
    torch.cuda.synchronize()
    assert abs(float(w.float().sum()) - 8 * 3.5) < 1e-4


def _pool_reuse(fn):
    pool = torch.cuda.graph_pool_handle()
    BK = 1 << 26  # 64 MiB
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1, pool=pool):
        x = torch.zeros(BK, device="cuda")
        x.fill_(1.0)
    w = fn(x)
    ptr_x = x.data_ptr()
    del x  # the contract: python must not pin the pool block
    gc.collect()

    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2, pool=pool):
        y = torch.zeros(BK, device="cuda")
        y.fill_(2.0)
    reused = y.data_ptr() == ptr_x
    del w, y, g1, g2
    gc.collect()
    return reused


@requires_gpu
def test_capture_pool_block_reuse():
    """THE OOM contract: seg2 in the same capture pool must reuse seg1's
    block once python dropped the weak-ref'd original — for whichever
    route is active (vendor or torch)."""
    assert WR.ROUTE != "view_as", WR._degrade_reason
    assert _pool_reuse(WR.weak_ref_tensor), (
        "pool block not reused — weak_ref is pinning; the 58-bucket "
        "prefill capture will OOM partway (see 2026-10-14 diagnosis)"
    )


# ---- torch route (no common_ops.so) ---------------------------------------

_torch_route = pytest.mark.skipif(
    not hasattr(torch._C, "_construct_storage_from_data_pointer"),
    reason="torch build lacks the non-owning storage entry points",
)


def _layouts():
    base = torch.arange(4 * 6 * 5, device="cuda", dtype=torch.bfloat16)
    base = base.view(4, 6, 5)
    return base, [
        base,
        base.transpose(0, 2),
        base[1:, 2:5, ::2],
        base[2],
        base.view(-1)[7:9],
        base.expand(3, 4, 6, 5),
    ]


@requires_gpu
@_torch_route
def test_torch_route_layout():
    _, views = _layouts()
    for t in views:
        w = WR._torch_weak_ref_impl(t)
        assert w.shape == t.shape
        assert w.stride() == t.stride()
        assert w.dtype == t.dtype and w.device == t.device
        assert w.data_ptr() == t.data_ptr()
        assert w.storage_offset() == 0
        assert torch.equal(w, t)


@requires_gpu
@_torch_route
def test_torch_route_alias():
    base, views = _layouts()
    ws = [WR._torch_weak_ref_impl(t) for t in views]
    base.fill_(3)
    torch.cuda.synchronize()
    for t, w in zip(views, ws):
        assert torch.equal(w, t)
    ws[2].fill_(-1)  # write through the weak view lands in the original
    torch.cuda.synchronize()
    assert bool((base[1:, 2:5, ::2] == -1).all())
    assert float(base[0].float().sum()) == 3 * 6 * 5


@requires_gpu
@_torch_route
def test_torch_route_passthrough():
    e = torch.empty(0, 3, device="cuda")
    assert WR._torch_weak_ref_impl(e) is e
    assert WR._torch_weak_ref_impl(7) == 7
    with pytest.raises(ValueError):
        WR._torch_weak_ref_impl(torch.zeros(2))


@requires_gpu
@_torch_route
def test_torch_route_pool_reuse():
    """Torch route must decouple even when the vendor .so is loaded."""
    assert _pool_reuse(WR._torch_weak_ref_impl)
    assert WR._probe_pool_reuse(WR._torch_weak_ref_impl)


@requires_gpu
def test_view_as_fails_probe():
    """The import-time probe really discriminates: view_as pins."""
    assert not WR._probe_pool_reuse(WR._view_as_weak_ref_tensor)


@requires_gpu
@_torch_route
@pytest.mark.skipif(WR._vendor_weak_ref_tensor is None,
                    reason="needs the vendor common_ops.so to compare")
def test_torch_route_matches_cpp_op():
    _, views = _layouts()
    for t in views:
        a = WR._vendor_weak_ref_tensor(t)
        b = WR._torch_weak_ref_impl(t)
        assert a.shape == b.shape
        assert a.stride() == b.stride()
        assert a.dtype == b.dtype and a.device == b.device
        assert a.storage_offset() == b.storage_offset()
        assert a.data_ptr() == b.data_ptr()
        assert a.untyped_storage().nbytes() == b.untyped_storage().nbytes()
        assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
        assert torch.equal(a, b)


def test_torch_route_degrades_without_entry_points(monkeypatch):
    monkeypatch.delattr(torch._C, "_construct_storage_from_data_pointer",
                        raising=False)
    fn, reason = WR._load_torch_impl()
    assert fn is None and "missing" in reason


def test_import_does_not_initialize_cuda():
    """Route selection is static: importing must not create a device context."""
    import os
    import subprocess
    import sys

    code = (
        "import torch, sgl_kernel.weak_ref as WR; "
        "assert not torch.cuda.is_initialized(), WR.ROUTE; print(WR.ROUTE)"
    )
    # Host bridges are out of scope here; isolate the module's own import.
    env = dict(os.environ, HCU_ADAPTER_HOST="none")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
