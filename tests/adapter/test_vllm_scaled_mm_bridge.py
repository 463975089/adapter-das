# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU binding contracts for both supported upstream INT8 call sites.

Run with HCU_ADAPTER_HOST=none; vendor/custom-op doubles execute no device math.
"""
from types import ModuleType, SimpleNamespace

import pytest
import torch

from hcu_adapter.infra import registry
from hcu_adapter.infra import hcu_config
from hcu_adapter.bridges import vllm as host, vllm_scaled_mm as bridge


@pytest.fixture
def startup(monkeypatch):
    module = ModuleType('vllm.model_executor.kernels.linear.scaled_mm.triton')
    exec('''
def triton_scaled_mm(input, weight, scale_a, scale_b, out_dtype, bias=None):
    return 'official'
class TritonFP8ScaledMMLinearKernel:
    def apply_scaled_mm(self, *args):
        return triton_scaled_mm(*args)
class TritonInt8ScaledMMLinearKernel:
    def apply_weights(self, *args):
        return triton_scaled_mm(*args)
''', module.__dict__)
    def vendor(a, b, scale_a, scale_b, m, n, k, transpose_flag, out_dtype, bias):
        raise AssertionError('CPU tests must not execute vendor math')
    monkeypatch.setattr(bridge, 'import_module', lambda name: {
        module.__name__: module,
        'lightop.gemm_ops': SimpleNamespace(hipblaslt_w8a8_channelwise_gemm=vendor),
    }[name])
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, 'gfx938', 'fixture'))
    monkeypatch.setattr(hcu_config, '_STATE', {'loaded': True, 'path': None,
                         'ops': {}, 'overrides': {}, 'hosts': {}})
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    publications = []
    class Op:
        def register_fake(self, fake):
            pass
        def __call__(self, *args):
            return 'custom-op'
    def custom_op(*args, **kwargs):
        publications.append(args[0])
        return Op()
    monkeypatch.setattr(torch.library, 'custom_op', custom_op)
    return module, publications


@pytest.mark.parametrize('site', ['v032-direct', 'v033-custom-op'])
@pytest.mark.parametrize('choice', ['auto', 'lightop'])
def test_both_supported_int8_sites_keep_official_methods(startup, site, choice):
    module, publications = startup
    hcu_config._STATE['ops']['vllm_scaled_mm'] = choice
    if site == 'v033-custom-op':
        exec('''
def _int8_scaled_mm_func(*args):
    return triton_scaled_mm(*args)
def apply_weights(self, *args):
    return _int8_scaled_mm_func(*args)
TritonInt8ScaledMMLinearKernel.apply_weights = apply_weights
''', module.__dict__)
    int8_method = module.TritonInt8ScaledMMLinearKernel.apply_weights
    fp8_method = module.TritonFP8ScaledMMLinearKernel.apply_scaled_mm
    assert bridge.install()
    assert module.TritonInt8ScaledMMLinearKernel.apply_weights is int8_method
    assert module.TritonFP8ScaledMMLinearKernel.apply_scaled_mm is fp8_method
    assert module.TritonInt8ScaledMMLinearKernel().apply_weights(*range(5)) == 'custom-op'
    assert module.TritonFP8ScaledMMLinearKernel().apply_scaled_mm(*range(5)) == 'custom-op'
    assert bridge.install() is False
    assert len(publications) == 1


@pytest.mark.parametrize('choice', ['auto', 'lightop'])
def test_unknown_host_site_stays_stock_in_auto_and_refuses_explicit(startup, choice):
    module, publications = startup
    hcu_config._STATE['ops']['vllm_scaled_mm'] = choice
    del module.TritonInt8ScaledMMLinearKernel
    original = module.triton_scaled_mm
    if choice == 'auto':
        assert bridge.install() is False
    else:
        with pytest.raises(host.VllmCompatibilityError, match='call site'):
            bridge.install()
    assert module.triton_scaled_mm is original
    assert registry.get('vllm_scaled_mm') is None
    assert not publications
