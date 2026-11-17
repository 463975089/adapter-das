# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for optional HCU paged allocator compatibility faces."""

# [L1 cold-curation 2026-11-17] 15 in-process cases -> 6: precedence parametrization keeps one op per family (decode-alloc/pool-assign/evict-mask); removed cases stay in git history.
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
        "hcu_assign_req_to_token_pool",
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


