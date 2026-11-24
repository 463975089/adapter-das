# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Early cache alias resolution; no GPU, provider execution or model load."""
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from hcu_adapter.bridges import vllm_sparse_flashmla as bridge


def config(backend='FLASHMLA_SPARSE', dtype='fp8_e4m3'):
    return NS(attention_config=NS(backend=NS(name=backend)),
              cache_config=NS(cache_dtype=dtype))


@pytest.mark.parametrize('dtype', ['fp8', 'fp8_e4m3'])
def test_alias_uses_core_rule_before_sparse_backend_validation(monkeypatch, dtype):
    calls = []
    backend = object()
    monkeypatch.setattr(bridge, 'require_available', lambda: calls.append('healthy'))
    registry = NS(AttentionBackendEnum=NS(FLASHMLA_SPARSE=NS(get_class=lambda: backend)))
    monkeypatch.setattr(bridge, '_backend_registry', lambda: registry)
    module = ModuleType('vllm.model_executor.layers.attention.mla_attention')
    def canonicalize(owner, requested):
        calls.append((owner, requested))
        return 'fp8_ds_mla'
    module._canonicalize_sparse_mla_kv_cache_dtype = canonicalize
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cfg = config(dtype=dtype)
    bridge.configure_cache_dtype(cfg)
    assert cfg.cache_config.cache_dtype == 'fp8_ds_mla'
    assert calls == ['healthy', (backend, dtype)]
    bridge.configure_cache_dtype(cfg)
    assert calls == ['healthy', (backend, dtype)]


@pytest.mark.parametrize('backend', [None, 'ROCM_AITER_MLA', 'FLASHINFER_MLA_SPARSE_SM120'])
def test_other_backend_keeps_requested_dtype(monkeypatch, backend):
    monkeypatch.setattr(bridge, 'require_available', lambda: pytest.fail('unselected provider'))
    cfg = config(backend=backend)
    bridge.configure_cache_dtype(cfg)
    assert cfg.cache_config.cache_dtype == 'fp8_e4m3'


@pytest.mark.parametrize('dtype', ['auto', 'bfloat16', 'fp8_e5m2', 'fp8_ds_mla', 'nvfp4_ds_mla'])
def test_other_dtype_keeps_its_contract(monkeypatch, dtype):
    monkeypatch.setattr(bridge, 'require_available', lambda: pytest.fail('unchanged dtype'))
    cfg = config(dtype=dtype)
    bridge.configure_cache_dtype(cfg)
    assert cfg.cache_config.cache_dtype == dtype


def test_unavailable_owner_cannot_publish_packed_cache_dtype(monkeypatch):
    def unavailable():
        raise RuntimeError('provider unavailable')
    monkeypatch.setattr(bridge, 'require_available', unavailable)
    cfg = config()
    with pytest.raises(RuntimeError, match='provider unavailable'):
        bridge.configure_cache_dtype(cfg)
    assert cfg.cache_config.cache_dtype == 'fp8_e4m3'


def test_partial_config_does_not_import_core(monkeypatch):
    monkeypatch.setattr(bridge, 'require_available', lambda: pytest.fail('partial config'))
    bridge.configure_cache_dtype(NS())
    bridge.configure_cache_dtype(NS(attention_config=NS(backend=None)))
