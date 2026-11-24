# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HY4 vendor selection and ABI ownership; numerical checks run on devices."""
from types import SimpleNamespace as NS

import pytest

from hcu_adapter.bridges import vllm as bridge
from hcu_adapter.features.dsa import hy4_sparse_indexer as owner


def providers(monkeypatch, arch='gfx938', missing=None, native=None):
    calls = []
    def lightop(Q, K, Weights, ks, ke, kv_scale=None, clean_logit=True, D_out=None):
        calls.append(('lightop', Q, K, Weights, ks, ke, kv_scale))
        return 'vendor'
    def aiter(Q, KV, kv_scales, weights, cu_starts, cu_ends):
        calls.append(('aiter', Q, KV, kv_scales, weights, cu_starts, cu_ends))
        return 'fallback'
    lightop_module = NS(mqa_logits=native or lightop)
    aiter_module = NS(fp8_mqa_logits=aiter)
    monkeypatch.setattr(bridge, 'detect_hcu', lambda: NS(detected=True, architecture=arch))
    def load(name):
        if name == 'lightop.gemmopt':
            if missing:
                raise ModuleNotFoundError('missing dependency', name=missing)
            return lightop_module
        if name.startswith('aiter.'):
            return aiter_module
        raise AssertionError(name)
    monkeypatch.setattr(owner, 'import_module', load)
    return lightop_module, aiter_module, calls


def test_gfx938_restores_lightop_and_forwards_existing_buffers(monkeypatch):
    lightop, _, calls = providers(monkeypatch)
    original = lightop.mqa_logits
    binding, scorer = owner._provider()
    query, keys, scales, weights, starts, ends = [object() for _ in range(6)]
    assert scorer(query, keys, scales, weights, starts, ends) == 'vendor'
    assert calls == [('lightop', query, keys, weights, starts, ends, scales)]
    assert binding.fp8_mqa_logits is scorer
    assert lightop.mqa_logits is original


def test_unqualified_arch_retains_aiter(monkeypatch):
    _, aiter, calls = providers(monkeypatch, arch='gfx936')
    binding, scorer = owner._provider()
    assert binding is aiter and scorer is aiter.fp8_mqa_logits
    assert calls == []


def test_absent_lightop_retains_original_aiter_provider(monkeypatch):
    _, aiter, _ = providers(monkeypatch, missing='lightop')
    assert owner._provider() == (aiter, aiter.fp8_mqa_logits)


def test_broken_dependency_does_not_silently_change_provider(monkeypatch):
    providers(monkeypatch, missing='vendor_private_dependency')
    with pytest.raises(ModuleNotFoundError, match='missing dependency'):
        owner._provider()


def test_incompatible_lightop_abi_is_rejected(monkeypatch):
    def changed(Q, K, Weights, ks, ke, scale):
        pass
    providers(monkeypatch, native=changed)
    with pytest.raises(RuntimeError, match='LightOp.*ABI'):
        owner._provider()


@pytest.mark.parametrize('removed', [False, True])
def test_vendor_owner_drift_is_latched(monkeypatch, removed):
    import os
    import torch
    lightop, _, _ = providers(monkeypatch)
    raw = lightop.mqa_logits
    binding, scorer = owner._provider()
    monkeypatch.setattr(owner, '_failure', None)
    operator = object()
    monkeypatch.setattr(torch.ops.vllm, owner.OP_NAME, operator, raising=False)
    monkeypatch.setattr(owner, '_state', (os.getpid(), None, operator, binding, scorer))
    monkeypatch.setattr(owner, 'op_disabled', lambda name: False)
    if removed:
        del lightop.mqa_logits
    else:
        lightop.mqa_logits = object()
    with pytest.raises(RuntimeError, match='owner'):
        owner._check()
    lightop.mqa_logits = raw
    with pytest.raises(RuntimeError, match='owner'):
        owner._check()
