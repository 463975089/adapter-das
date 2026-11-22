# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU policy/cache tests only; GPU numerical acceptance is a separate probe."""
from types import SimpleNamespace
import gc

import pytest
import torch

from hcu_adapter.kernels.mhc_mega import MhcMegaProvider, MhcMegaRuntime


def test_provider_detects_rebinding():
    function = lambda: None
    owner = SimpleNamespace(mega_mhc=function)
    provider = MhcMegaProvider(((owner, 'mega_mhc', function),), function,
                               function, 'digest', 'test')
    provider.require_healthy()
    owner.mega_mhc = lambda: None
    with pytest.raises(RuntimeError, match='provider changed'):
        provider.require_healthy()


def _runtime(monkeypatch):
    calls = []
    def prepare(fn):
        calls.append(fn.shape)
        return fn.clone()
    runtime = MhcMegaRuntime(SimpleNamespace(prepare=prepare), None)
    # Only cache bookkeeping is under test, never pretend these are GPU events.
    monkeypatch.setattr(runtime, '_record', lambda value, stream: (value, None, 0))
    monkeypatch.setattr(runtime, '_ready', lambda entry, stream: entry[0])
    return runtime, calls


def test_prepack_cache_invalidates_on_in_place_weight_update(monkeypatch):
    runtime, calls = _runtime(monkeypatch)
    fn = torch.zeros(24, 16)
    first = runtime._prepare(fn, None)
    assert runtime._prepare(fn, None) is first and len(calls) == 1
    fn.add_(1)
    second = runtime._prepare(fn, None)
    assert second is not first and torch.equal(second, fn) and len(calls) == 2
    assert not torch.equal(first, fn)
    stats = runtime.cache_stats()
    assert (stats['repack'], stats['reuse'], stats['calls'], stats['versionless']) == (2, 1, 3, 0)
    del fn
    gc.collect()
    assert not runtime._packed


def test_versionless_inference_weights_are_not_cached(monkeypatch):
    runtime, calls = _runtime(monkeypatch)
    with torch.inference_mode():
        fn = torch.zeros(24, 16)
        runtime._prepare(fn, None)
        fn.add_(1)
        second = runtime._prepare(fn, None)
        assert torch.equal(second, fn)
    assert len(calls) == 2 and not runtime._packed
    stats = runtime.cache_stats()
    assert (stats['repack'], stats['reuse'], stats['versionless']) == (2, 0, 2)


def test_versionless_warning_and_warmup_summary_are_logged_once(monkeypatch, caplog):
    runtime, calls = _runtime(monkeypatch)
    caplog.set_level('INFO', logger='hcu.adapter')
    with torch.inference_mode():
        fn = torch.zeros(24, 16)
        for _ in range(130):
            runtime._prepare(fn, None)
    stats = runtime.cache_stats()
    assert (stats['repack'], stats['reuse'], stats['calls']) == (130, 0, 130)
    messages = [record.getMessage() for record in caplog.records]
    assert sum('no mutation version' in text for text in messages) == 1
    assert sum('after warmup' in text for text in messages) == 1
    assert any('repack=128 reuse=0 versionless=128 calls=128' in text for text in messages)


def test_y_scratch_reuses_shape_but_separates_shapes_and_streams(monkeypatch):
    runtime, _ = _runtime(monkeypatch)
    r = torch.empty(2, 4, 8, dtype=torch.bfloat16)
    stream = SimpleNamespace(cuda_stream=0)
    first = runtime._y_scratch(r, stream)
    assert runtime._y_scratch(r, stream) is first
    assert runtime._y_scratch(torch.empty_like(r), stream) is first
    other_tokens = runtime._y_scratch(torch.empty(3, 4, 8, dtype=r.dtype), stream)
    other_hidden = runtime._y_scratch(torch.empty(2, 4, 16, dtype=r.dtype), stream)
    other_stream = runtime._y_scratch(r, SimpleNamespace(cuda_stream=1))
    assert other_tokens.shape == (3, 8) and other_hidden.shape == (2, 16)
    assert len({t.data_ptr() for t in (first, other_tokens, other_hidden, other_stream)}) == 4
    assert (r.device, 2, 8) in runtime._scratch
    assert runtime.cache_stats()['scratch_allocations'] == 4
    assert runtime.cache_stats()['scratch_reuses'] == 2


def test_cpu_or_unsupported_seams_never_enter_mega():
    residual = torch.zeros(2, 4, 5120, dtype=torch.bfloat16)
    fn = torch.zeros(24, 20480)
    scale, base = torch.ones(3), torch.zeros(24)
    prev, x = torch.ones(2, 4), torch.zeros(2, 5120, dtype=torch.bfloat16)
    post, comb = torch.ones(2, 4, 1), torch.ones(2, 4, 4)
    arguments = (residual, fn, scale, base, prev, None, None, x, post, comb, 20)
    assert not MhcMegaRuntime.supported(*arguments)
    args = list(arguments); args[4] = None
    assert not MhcMegaRuntime.supported(*args)
    args = list(arguments); args[5] = x
    assert not MhcMegaRuntime.supported(*args)
    args = list(arguments); args[6] = torch.ones(5120)
    assert not MhcMegaRuntime.supported(*args)


def test_meta_outputs_preserve_contract_and_own_storage():
    residual = torch.empty(0, 4, 5120, device='meta', dtype=torch.bfloat16)
    result = MhcMegaRuntime.seam_meta(residual, None, None, None,
        1e-6, 1e-6, 1e-6, 2.0, 20, None, None, None, None)
    assert [tuple(t.shape) for t in result] == [
        (0, 4, 5120), (0, 4, 1), (0, 4, 4), (0, 5120), (0, 4)]
    assert [t.dtype for t in result] == [torch.bfloat16, torch.float32,
        torch.float32, torch.bfloat16, torch.float32]
    assert result[0] is not residual


def test_registered_mega_schema_and_fake_tensor(monkeypatch):
    from typing import get_type_hints

    from torch._library.infer_schema import infer_schema
    from torch._subclasses.fake_tensor import FakeTensorMode
    from hcu_adapter.bridges import vllm_mhc

    monkeypatch.setattr(vllm_mhc, 'torch', torch, raising=False)
    runtime = MhcMegaRuntime(None, None)
    seam, fake = vllm_mhc._declare_mega(runtime)
    name = 'mega_seam'
    library = torch.library.Library('mhc_mega_contract', 'FRAGMENT')
    # Torch 2.4 CPU uses evaluated annotations for schema inference.
    seam.__annotations__ = get_type_hints(seam)
    library.define(name + infer_schema(seam, mutates_args=[]))
    library.impl(name, seam, 'CPU')
    library._register_fake(name, fake)
    with FakeTensorMode():
        residual = torch.empty(2, 4, 16, dtype=torch.bfloat16)
        fn, scale, base = torch.empty(24, 64), torch.empty(3), torch.empty(24)
        pre = torch.empty(2, 4)
        x = torch.empty(2, 16, dtype=residual.dtype)
        post, comb = torch.empty(2, 4, 1), torch.empty(2, 4, 4)
        result = torch.ops.mhc_mega_contract.mega_seam(
            residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20, pre, x, post, comb)
    assert [(t.shape, t.dtype) for t in result] == [
        ((2, 4, 16), torch.bfloat16), ((2, 4, 1), torch.float32),
        ((2, 4, 4), torch.float32), ((2, 16), torch.bfloat16), ((2, 4), torch.float32),
    ]
