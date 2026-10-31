"""Contracts for optional HCU paged allocator compatibility faces."""

from __future__ import annotations

import sys
import types

import torch

from sgl_kernel import kvcacheio


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
