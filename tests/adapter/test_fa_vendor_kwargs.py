# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType

import pytest
import torch

from hcu_adapter import fa_vendor, registry
from sgl_kernel import flash_attn


@pytest.fixture
def vendor_router(monkeypatch):
    calls = []
    provider = ModuleType('flash_attn')

    def varlen(q, *args, **kwargs):
        calls.append((args, kwargs))
        return torch.full_like(q, 7)

    provider.flash_attn_varlen_func = varlen
    monkeypatch.setitem(sys.modules, 'flash_attn', provider)
    monkeypatch.setitem(
        sys.modules, 'sglang.kernels.ops.attention.flash_attention_v3', None,
    )
    monkeypatch.setenv('HCU_FA_IMPL', 'vendor')
    monkeypatch.setenv('HCU_DISABLE_OPS', '')
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    # This fixture tests vendor call arguments with a fake provider, not a
    # SGLang startup. The E5M2 guard has its own tests in test_flash_attn.py.
    monkeypatch.setattr(fa_vendor, '_install_e5m2_backend_guard', lambda: None)
    for name in ('flash_attn_varlen_func', 'flash_attn_with_kvcache'):
        monkeypatch.setattr(flash_attn, name, getattr(flash_attn, name))
    fa_vendor.install()
    assert registry.get('flash_attention') is not None
    return flash_attn.flash_attn_varlen_func, calls


@pytest.fixture(params=[torch.float16, torch.bfloat16])
def inputs(request):
    q = torch.ones(2, 3, 4, 64, dtype=request.param, device='cpu')
    k = torch.ones(2, 5, 2, 64, dtype=request.param, device='cpu')
    return q, k, k.clone()


def invoke(router, inputs, dense, **kwargs):
    q, k, v = inputs
    if dense:
        cu_q = cu_k = None
    else:
        q, k, v = (tensor.flatten(0, 1) for tensor in inputs)
        cu_q = torch.tensor([0, 3, 6], dtype=torch.int32, device='cpu')
        cu_k = torch.tensor([0, 5, 10], dtype=torch.int32, device='cpu')
    return router(q, k, v, cu_q, cu_k, 3, 5, **kwargs)


@pytest.mark.parametrize('dense', [False, True])
@pytest.mark.parametrize('option', ['alibi', 'dropout', 'both'])
def test_unsupported_bias_or_dropout_is_rejected_before_vendor(
    vendor_router, inputs, dense, option,
):
    router, calls = vendor_router
    extra = {}
    if option in ('alibi', 'both'):
        extra['alibi_slopes'] = torch.tensor([0.25, 0.5, 0.75, 1.0], device='cpu')
    if option in ('dropout', 'both'):
        extra['dropout_p'] = 0.25
    with pytest.raises(NotImplementedError, match='ALiBi|dropout'):
        invoke(router, inputs, dense, **extra)
    assert calls == []
    assert all(torch.all(tensor == 1) for tensor in inputs)


@pytest.mark.parametrize('dense', [False, True])
@pytest.mark.parametrize('extra', [
    {}, {'alibi_slopes': None}, {'dropout_p': 0.0},
    {'alibi_slopes': None, 'dropout_p': 0.0},
])
def test_inference_defaults_still_reach_vendor(vendor_router, inputs, dense, extra):
    router, calls = vendor_router
    actual = invoke(router, inputs, dense, **extra)
    assert actual.shape == (inputs[0].shape if dense else (6, 4, 64))
    assert torch.all(actual == 7)
    assert len(calls) == 1


@pytest.mark.parametrize('dense', [False, True])
def test_undeclared_keyword_is_not_silently_dropped(vendor_router, inputs, dense):
    router, calls = vendor_router
    with pytest.raises(TypeError, match='unexpected_bias'):
        invoke(router, inputs, dense, unexpected_bias=torch.ones(4, device='cpu'))
    assert calls == []
