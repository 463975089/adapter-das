# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for optional HCU paged allocator compatibility faces."""

from __future__ import annotations

import sys
import os
import types
import subprocess

import pytest
import torch

from sgl_kernel import kvcacheio


@pytest.fixture(autouse=True)
def isolated_vendor_ops(monkeypatch):
    """Exercise absence independently of extensions installed on the runner."""
    monkeypatch.setattr(torch.ops, "sgl_kernel", types.SimpleNamespace())


@pytest.mark.parametrize(
    "name",
    [
        "hcu_alloc_decode_kernel",
        "hcu_alloc_extend_kernel",
        "hcu_assign_req_to_token_pool",
        "hcu_get_last_loc",
        "hcu_assign_extend_cache_locs",
        "hcu_create_extend_after_decode_spec_info",
        "hcu_create_chunked_prefix_cache_kv_indices",
        "hcu_align_evict_mask_to_page_size",
    ],
)
def test_registered_vendor_op_takes_precedence(monkeypatch, name):
    import inspect

    calls = []
    result = object()

    def vendor(*args):
        calls.append(args)
        return result

    monkeypatch.setattr(torch.ops.sgl_kernel, name, vendor, raising=False)
    wrapper = getattr(kvcacheio, name)
    args = tuple(object() for _ in inspect.signature(wrapper).parameters)
    actual = wrapper(*args)
    assert calls == [args]
    if name == "hcu_get_last_loc":
        assert actual is result


def test_vendor_failure_is_not_hidden_by_fallback(monkeypatch):
    def vendor(*args):
        raise RuntimeError("vendor execution failed")

    monkeypatch.setattr(
        torch.ops.sgl_kernel, "hcu_get_last_loc", vendor, raising=False
    )
    with pytest.raises(RuntimeError, match="vendor execution failed"):
        kvcacheio.hcu_get_last_loc(None, None, None)


def test_speculative_cache_read_fallback():
    pool = torch.arange(24, dtype=torch.int32).reshape(3, 8)
    original = pool.clone()
    output = torch.full((6,), -2, dtype=torch.int64)

    kvcacheio.hcu_assign_extend_cache_locs(
        torch.tensor([2, 1, 0]),
        pool,
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 2, 6]),
        output,
        pool_len=8,
        bs=3,
    )

    assert output.tolist() == [17, 18, 19, 3, 4, 5]
    torch.testing.assert_close(pool, original)


class _Kernel:
    def __init__(self):
        self.grid = None
        self.args = None

    def __getitem__(self, grid):
        self.grid = grid

        def launch(*args):
            self.args = args

        return launch


def _install_allocator(monkeypatch, decode, extend):
    module = types.ModuleType("sglang.kernels.ops.memory.allocator")
    module.alloc_decode_kernel = decode
    module.alloc_extend_kernel = extend
    for name in (
        "sglang",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.memory",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_decode_face_routes_to_upstream_triton(monkeypatch):
    decode, extend = _Kernel(), _Kernel()
    _install_allocator(monkeypatch, decode, extend)
    tensors = [torch.empty(1) for _ in range(4)]

    kvcacheio.hcu_alloc_decode_kernel(*tensors, bs=3, page_size=16)

    assert decode.grid == (3,)
    assert decode.args == (*tensors, 4, 16)
    assert extend.args is None


def test_request_pool_compatibility_faces():
    req_to_token = torch.full((3, 8), -1, dtype=torch.int64)
    reqs = torch.tensor([2, 0], dtype=torch.int32)
    starts = torch.tensor([1, 2], dtype=torch.int32)
    ends = torch.tensor([3, 5], dtype=torch.int32)
    cache = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int64)

    kvcacheio.hcu_assign_req_to_token_pool(
        reqs, req_to_token, starts, ends, cache, shape=8, bs=2
    )

    assert req_to_token[2, 1:3].tolist() == [10, 11]
    assert req_to_token[0, 2:5].tolist() == [20, 21, 22]
    assert kvcacheio.hcu_get_last_loc(
        req_to_token,
        reqs,
        torch.tensor([3, 0], dtype=torch.int32),
    ).tolist() == [11, -1]


def test_extend_face_routes_to_upstream_triton(monkeypatch):
    decode, extend = _Kernel(), _Kernel()
    _install_allocator(monkeypatch, decode, extend)
    tensors = [torch.empty(1) for _ in range(5)]

    kvcacheio.hcu_alloc_extend_kernel(*tensors, bs=5, page_size=32)

    assert extend.grid == (5,)
    assert extend.args == (*tensors, 8, 32)
    assert decode.args is None


def test_shadow_import_does_not_import_sglang():
    code = (
        "import sys; import sgl_kernel.kvcacheio; "
        "assert not any(name == 'sglang' or name.startswith('sglang.') "
        "for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ, HCU_ADAPTER_HOST="none", VLLM_PLUGINS="none"),
    )
    assert result.returncode == 0, result.stderr


def test_allocator_route_logs_vendor_and_fallback_once(monkeypatch, caplog):
    from hcu_adapter import kvcacheio_allocator

    caplog.set_level("DEBUG", logger="hcu.adapter")
    kvcacheio_allocator._ROUTE_LOGGED.clear()
    monkeypatch.setattr(torch.ops, "sgl_kernel", types.SimpleNamespace())
    assert kvcacheio_allocator.resolve("hcu_get_last_loc") is not None
    assert kvcacheio_allocator.resolve("hcu_get_last_loc") is not None
    assert sum("route=fallback" in record.message for record in caplog.records) == 1

    vendor = lambda *args: None
    monkeypatch.setattr(
        torch.ops.sgl_kernel, "hcu_get_last_loc", vendor, raising=False
    )
    assert kvcacheio_allocator.resolve("hcu_get_last_loc") is vendor
    assert sum("route=vendor" in record.message for record in caplog.records) == 1
