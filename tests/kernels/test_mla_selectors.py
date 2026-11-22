# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Selector oracle contracts and real HCU parity; CPU does not qualify Triton."""
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from hcu_adapter.infra import hcu_config
from hcu_adapter.kernels import mla_sparse_decode_dtk as kernel


@pytest.fixture(autouse=True)
def fused_jit(monkeypatch):
    monkeypatch.setenv(kernel._FUSED_SELECTORS_ENV, '1')
    monkeypatch.setenv('HCU_TRITON_FAST_LAUNCH', '0')


@pytest.fixture(params=['cpu', 'cuda'])
def selector_device(request):
    if request.param == 'cuda':
        if not torch.cuda.is_available() or torch.version.hip is None:
            pytest.skip('GPU selector parity requires HCU; CPU is oracle-only')
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
        if arch not in kernel.SUPPORTED_ARCHES:
            pytest.skip('selector candidate requires gfx936/gfx938')
    return request.param


def _candidate(*args):
    source = args[2] if args[2] is not None else args[0]
    if source.is_cuda:
        return kernel._selectors(*args)
    if os.environ.get('TRITON_INTERPRET') != '1':
        pytest.skip('CPU algorithm checks require TRITON_INTERPRET=1; no GPU evidence')
    from hcu_adapter.kernels.mla_selectors_triton import selectors

    # Interpret the actual kernel bodies on CPU, without a CUDA device guard.
    # This does not compile or run device code and cannot qualify GPU behavior.
    with patch('torch.cuda.device', return_value=nullcontext()):
        return selectors(*args)


def _strided_copy(value, device):
    backing = torch.empty((*value.shape[:-1], value.shape[-1] * 2),
                          dtype=value.dtype, device=device)
    view = backing[..., ::2]
    view.copy_(value)
    return view


def _inputs(mode, rows, width, device, strided):
    slots = 31
    dense = torch.arange(rows * width, dtype=torch.int32).reshape(rows, width) % slots
    if width:
        dense[:, 0] = 7
    if width > 1:
        dense[:, 1] = 7  # Duplicates must remain distinct entries.
    if width > 2:
        dense[:, 2] = -1
    if width > 3:
        dense[:, 3] = slots
    lengths = torch.tensor([width if r == 0 else (0 if r == 1 else width // 2)
                            for r in range(rows)], dtype=torch.int32)
    move = (lambda t: _strided_copy(t, device)) if strided else (lambda t: t.to(device))
    if mode == 'csr':
        flat = torch.full((rows * width,), -1, dtype=torch.int32)
        indptr = [0]
        cursor = 0
        for r in range(rows):
            n = int(lengths[r])
            flat[cursor:cursor + n] = dense[r, :n]
            cursor += n
            indptr.append(cursor)
        return None, None, move(flat), move(torch.tensor(indptr, dtype=torch.int32))
    dense = move(dense)
    if mode == 'dense3':
        dense = dense.unsqueeze(1)
    return dense, move(lengths), None, None


@pytest.mark.parametrize('mode', ['csr', 'dense', 'dense3'])
@pytest.mark.parametrize('width', [0, 1, 37, 63, 64, 65, 129, 512, 4096])
@pytest.mark.parametrize('strided', [False, True])
def test_outputs_match_eager(selector_device, mode, width, strided):
    rows, slots = 3, 31
    args = _inputs(mode, rows, width, selector_device, strided)
    expected = kernel._selectors_eager(*args, rows, slots)
    # CUDA runs the candidate; CPU can exercise its bodies through the interpreter.
    actual = _candidate(*args, rows, slots)
    assert actual[0].dtype == actual[1].dtype == torch.int32
    assert actual[0].is_contiguous() and actual[1].is_contiguous()
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[0].shape == (rows, 1, max(64, (width + 63) // 64 * 64))


@pytest.mark.parametrize('width', [0, 1, 65, 4096])
def test_single_row(selector_device, width):
    args = _inputs('csr', 1, width, selector_device, True)
    expected = kernel._selectors_eager(*args, 1, 31)
    actual = _candidate(*args, 1, 31)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize('pointers', [
    [-2, 4, 4, 18],        # Negative offsets, empty row, offsets beyond capacity.
    [2, 5, 9, 10],        # Actual CSR offsets; unused capacity at the end.
    [9, 3, 12, 12],       # Negative row length is inactive, not an unsafe read.
    [-2147483648, 2147483647, 0, 12],  # int32 length subtraction overflow.
])
def test_csr_offset_filtering(selector_device, pointers):
    flat = torch.tensor([4, 4, -1, 7, 8, 31, 30, 2, -9, 6, 6, 0], dtype=torch.int32)
    flat = _strided_copy(flat, selector_device)
    indptr = _strided_copy(torch.tensor(pointers, dtype=torch.int32), selector_device)
    args = (None, None, flat, indptr, 3, 31)
    expected = kernel._selectors_eager(*args)
    actual = _candidate(*args)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_known_values_and_empty_rows(selector_device):
    dense = torch.tensor([[7, 7, -1, 31, 2, 30], [1, 2, 3, 4, 5, 6],
                          [-1, 31, 99, -2, 32, -3]], dtype=torch.int32)
    lengths = torch.tensor([6, 0, 6], dtype=torch.int32)
    actual = kernel._selectors(dense.to(selector_device), lengths.to(selector_device),
                               None, None, 3, 31)
    expected = torch.full((3, 1, 64), -1, dtype=torch.int32, device=selector_device)
    expected[0, 0, :4] = torch.tensor([2, 7, 7, 30], dtype=torch.int32,
                                     device=selector_device)
    assert torch.equal(actual[0], expected)
    assert torch.equal(actual[1], torch.tensor([4, 0, 0], dtype=torch.int32,
                                              device=selector_device))


def test_large_width_retains_eager(selector_device, monkeypatch):
    args = _inputs('csr', 1, kernel._SELECTORS_FUSED_MAX_WIDTH + 1,
                   selector_device, False)
    original = kernel._selectors_eager
    calls = []

    def record(*a):
        calls.append(True)
        return original(*a)

    monkeypatch.setattr(kernel, '_selectors_eager', record)
    actual = kernel._selectors(*args, 1, 31)
    expected = original(*args, 1, 31)
    assert calls == [True]
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize('fused,fast,width,is_cuda,use_triton', [
    (None, '0', 129, True, True),
    (None, '0', 4097, True, False),
    (None, '0', 129, False, False),
    ('0', '0', 129, True, False),
    ('0', '1', 129, True, False),
    ('1', '0', 129, True, True),
    ('1', '0', 4096, True, True),
    ('1', '0', 4097, True, False),
    ('1', '0', 129, False, False),
])
def test_native_selector_dispatch(monkeypatch, fused, fast, width, is_cuda, use_triton):
    """Verify the production gate and width fallback without touching a device."""
    if fused is None:
        monkeypatch.delenv(kernel._FUSED_SELECTORS_ENV, raising=False)
        monkeypatch.setattr(hcu_config, 'lookup', lambda name: None)
    else:
        monkeypatch.setenv(kernel._FUSED_SELECTORS_ENV, fused)
    monkeypatch.setenv('HCU_TRITON_FAST_LAUNCH', fast)
    triton_marker, eager_marker = object(), object()
    module = SimpleNamespace(selectors=lambda *args: triton_marker)
    monkeypatch.setitem(sys.modules, 'hcu_adapter.kernels.mla_selectors_triton', module)
    monkeypatch.setattr(kernel, '_selectors_eager', lambda *args: eager_marker)
    dense = SimpleNamespace(numel=lambda: 2 * width, is_cuda=is_cuda)
    assert kernel._selectors(dense, None, None, None, 2, 31) is (
        triton_marker if use_triton else eager_marker)


def test_graph_replay(selector_device):
    if selector_device == 'cpu':
        pytest.skip('CPU oracle cannot qualify graph capture/replay')
    args = _inputs('csr', 3, 129, selector_device, True)
    for _ in range(3):
        kernel._selectors(*args, 3, 31)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _candidate(*args, 3, 31)
    for step in range(3):
        args[2].fill_(step + 1)
        args[3].copy_(torch.tensor([0, 4 + step, 7 + step, 10 + step],
                                   dtype=torch.int32, device=selector_device))
        expected = kernel._selectors_eager(*args, 3, 31)
        graph.replay()
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
