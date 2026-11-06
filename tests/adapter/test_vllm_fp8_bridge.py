# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Offline public-registration contract; GPU math lives in the real provider probe."""
from types import SimpleNamespace

import pytest

from hcu_adapter.bridges import vllm as bridge
from hcu_adapter.bridges import vllm_fp8 as fallback
from hcu_adapter import registry


@pytest.fixture(params=("gfx936", "gfx938"))
def fake_seams(monkeypatch, request):
    calls = []

    class OfficialKernel:
        def apply_scaled_mm(self, *, A, B, out_dtype, As, Bs, bias, output_shape):
            raise AssertionError('offline tests must not execute provider math')

    def register_linear_kernel(kernel_class, platform, kernel_type='mp'):
        calls.append((kernel_class, platform, kernel_type))

    linear = SimpleNamespace(register_linear_kernel=register_linear_kernel)
    native = SimpleNamespace(_supports_torch_fp8_scaled_mm=lambda: False)
    modules = {
        'vllm.model_executor.kernels.linear': linear,
        'vllm.model_executor.kernels.linear.scaled_mm.pytorch': native,
        'vllm.model_executor.kernels.linear.scaled_mm.triton': SimpleNamespace(TritonFP8ScaledMMLinearKernel=OfficialKernel),
        'vllm.platforms': SimpleNamespace(PlatformEnum=SimpleNamespace(ROCM='rocm')),
    }
    monkeypatch.setattr(fallback, 'import_module', modules.__getitem__)
    monkeypatch.setattr(bridge, 'detect_hcu', lambda: bridge.HcuDetection(True, request.param, ''))
    records = []
    monkeypatch.setattr(registry, 'register', lambda **kw: records.append(kw))
    return linear, native, OfficialKernel, calls, records


def test_registers_the_exact_official_class_last_once(fake_seams):
    linear, native, kernel, calls, records = fake_seams
    original_predicate = native._supports_torch_fp8_scaled_mm
    assert fallback.install() is True
    assert fallback.install() is False
    assert calls == [(kernel, 'rocm', 'fp8')]
    assert native._supports_torch_fp8_scaled_mm is original_predicate
    assert records[0]['mechanism'] == registry.MECH_SEAM


@pytest.mark.parametrize('arch,detected', [('gfx928',True),('gfx92a',True),('gfx942',False),('gfx950',False),(None,False)])
def test_unvalidated_architectures_do_not_import_or_register(monkeypatch, arch, detected):
    monkeypatch.setattr(bridge, 'detect_hcu', lambda: bridge.HcuDetection(detected,arch,''))
    monkeypatch.setattr(fallback, 'import_module', lambda name: pytest.fail('unvalidated host imported '+name))
    assert fallback.install() is False


def test_official_native_support_is_an_exit_condition(fake_seams):
    _, native, _, calls, records = fake_seams
    native._supports_torch_fp8_scaled_mm = lambda: True
    assert fallback.install() is False
    assert calls == records == []


def test_missing_or_changed_registration_abi_fails_closed(fake_seams):
    linear, _, _, calls, records = fake_seams
    linear.register_linear_kernel = lambda unknown: None
    with pytest.raises(bridge.VllmCompatibilityError, match='register_linear_kernel'):
        fallback.install()
    assert calls == records == []


def test_changed_provider_abi_fails_closed(fake_seams):
    _, _, kernel, calls, records = fake_seams
    kernel.apply_scaled_mm = lambda self, A, B: None
    with pytest.raises(bridge.VllmCompatibilityError, match='apply_scaled_mm'):
        fallback.install()
    assert calls == records == []


def test_changed_native_support_abi_fails_closed(fake_seams):
    _, native, _, calls, records = fake_seams
    native._supports_torch_fp8_scaled_mm = lambda device: False
    with pytest.raises(bridge.VllmCompatibilityError, match='zero-argument'):
        fallback.install()
    assert calls == records == []


def test_non_boolean_native_predicate_fails_closed(fake_seams):
    _, native, _, calls, records = fake_seams
    native._supports_torch_fp8_scaled_mm = lambda: (False, 'unsupported')
    with pytest.raises(bridge.VllmCompatibilityError, match='return bool'):
        fallback.install()
    assert calls == records == []


def test_missing_native_seam_fails_closed(fake_seams):
    _, native, _, calls, records = fake_seams
    del native._supports_torch_fp8_scaled_mm
    with pytest.raises(bridge.VllmCompatibilityError, match='ABI changed'):
        fallback.install()
    assert calls == records == []
