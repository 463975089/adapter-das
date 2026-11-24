# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Native CT constructors/loaders with CPU doubles only at device kernels."""
from dataclasses import replace
import gc
import weakref

import pytest
import torch
import test_int8_empty_layer_contract as native
from hcu_adapter.features.ct_w4a8 import lightop_int8_linear as owned
from vllm import _custom_ops as ops
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation, get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kInt8DynamicTokenSym

native_cpu_contract = native.native_cpu_contract


class Config(native._OwnedCT):
    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if type(method) is native.owned.HcuDSV41Int8LinearMethod:
            return owned.HcuLightopInt8LinearMethod(self)
        return method


@pytest.fixture(autouse=True)
def vendor_cpu(monkeypatch):
    monkeypatch.setattr(owned, '_device_supported', lambda device: True)
    monkeypatch.setattr(owned, '_token_quant', lambda x: native.quantize(x)[:2])
    def fused(x, *, output, scales):
        q, s, _ = native.quantize(torch.nn.functional.silu(x[..., :x.shape[-1] // 2]) * x[..., x.shape[-1] // 2:])
        output.copy_(q); scales.copy_(s)
    monkeypatch.setattr(owned, '_fused_quant', fused)
    from types import SimpleNamespace
    from vllm.config import CompilationConfig, set_current_vllm_config
    cfg = SimpleNamespace(compilation_config=CompilationConfig(mode=0, custom_ops=['none']),
                          kernel_config=SimpleNamespace(linear_backend='auto'))
    with set_current_vllm_config(cfg, check_compile=False):
        yield


def build(**kwargs):
    return native.build(cfg=Config.from_config(native.ct_description(**kwargs)))[0]


def activation(layer, rows=3):
    k = layer.weight.shape[0]
    return QuantizedActivation(torch.ones(rows, k, dtype=torch.int8),
        torch.ones(rows, 1), torch.bfloat16, torch.Size((rows, k)), kInt8DynamicTokenSym)


def test_native_kernel_private_binding_and_qa(monkeypatch):
    original = native.TritonInt8ScaledMMLinearKernel.apply_weights
    original_quant = ops.scaled_int8_quant
    layer = build()
    assert get_input_quant_key(layer) == kInt8DynamicTokenSym
    assert type(layer.scheme.kernel) is native.TritonInt8ScaledMMLinearKernel
    x = torch.ones(3, native.K, dtype=torch.bfloat16)
    ordinary = layer(x)[0]
    actual = layer(activation(layer))[0]
    assert torch.equal(actual, ordinary)
    assert native.TritonInt8ScaledMMLinearKernel.apply_weights is original
    assert ops.scaled_int8_quant is original_quant


@pytest.mark.parametrize('kwargs', [dict(static=True), dict(symmetric=False), dict(strategy='tensor')])
def test_unsupported_schemes_preserve_native(kwargs):
    layer = build(**kwargs)
    assert get_input_quant_key(layer) is None
    assert layer.scheme.kernel.apply_weights.__func__ is native.TritonInt8ScaledMMLinearKernel.apply_weights
    with pytest.raises(TypeError, match='QuantizedActivation'):
        layer(activation(layer))
    assert layer(torch.ones(2, native.K, dtype=torch.bfloat16))[0].shape == (2, native.N)


@pytest.mark.parametrize('field,value', [
    ('quant_key', None), ('orig_dtype', torch.float16),
    ('orig_shape', torch.Size((3, native.K + 1))),
    ('scale', torch.ones(3, 2)), ('data', torch.ones(3, native.K)),
])
def test_bad_qa_metadata_rejected(field, value):
    layer = build()
    with pytest.raises((TypeError, ValueError), match='activation'):
        layer(replace(activation(layer), **{field: value}))


def test_empty_and_weight_identity_invalidation():
    layer = build()
    assert layer(torch.empty(0, native.K, dtype=torch.bfloat16))[0].shape == (0, native.N)
    layer.weight = torch.nn.Parameter(layer.weight.clone(), requires_grad=False)
    with pytest.raises(RuntimeError, match='witness'):
        layer(activation(layer))


def test_producer_fallback_weak_reference_and_native_parameter_names():
    layer = build()
    producer = owned.FusedSiluInt8Producer(SiluAndMul(), layer)
    assert list(producer.named_parameters()) == []
    x = torch.randn(3, native.K * 2, dtype=torch.bfloat16)
    assert isinstance(producer(x), QuantizedActivation)
    layer.requires_unquantized_input = True
    assert isinstance(producer(x), torch.Tensor)
    assert torch.equal(producer(x), producer.original(x))
    layer.requires_unquantized_input = False
    assert producer(x.float()).dtype == torch.float32
    assert producer(x[:0]).shape == (0, native.K)
    ref = weakref.ref(layer)
    del layer; gc.collect()
    assert ref() is None
    assert isinstance(producer(x), torch.Tensor)


def test_failed_repeated_finalize_revokes_capability():
    layer = build()
    with pytest.raises(RuntimeError, match='cannot be repeated'):
        layer.quant_method.process_weights_after_loading(layer)
    assert get_input_quant_key(layer) is None
    with pytest.raises((RuntimeError, TypeError)):
        layer(activation(layer))


@pytest.mark.parametrize('keywords', [False, True])
def test_native_layerwise_reload_preserves_storage_and_named_loader(keywords):
    from vllm.model_executor.model_loader.reload import layerwise
    layer, weight, scales = native.build(cfg=Config.from_config(native.ct_description()), finalize=False)
    layerwise.record_metadata_for_reloading(layer)
    layer.quant_method.process_weights_after_loading(layer)
    stable_weight, stable_scale = layer.weight, layer.weight_scale
    for increment in [1, 2]:
        layerwise.initialize_layerwise_reload(layer)
        assert layer.weight.device.type == 'meta'
        for name, value in [('weight', weight + increment), ('weight_scale', scales + increment)]:
            param = getattr(layer, name)
            if keywords: param.weight_loader(param=param, loaded_weight=value)
            else: param.weight_loader(param, value)
        assert layer.weight is stable_weight and layer.weight_scale is stable_scale
        assert torch.equal(layer.weight, (weight + increment).t())
        assert torch.equal(layer.weight_scale, scales + increment)
        assert layer(activation(layer))[0].shape == (3, native.N)
        assert layer(torch.empty(0,native.K,dtype=torch.bfloat16))[0].shape == (0,native.N)
