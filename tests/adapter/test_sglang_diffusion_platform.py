# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import logging
import subprocess
import sys
import os
from types import SimpleNamespace

import pytest

from hcu_adapter.infra import registry
from hcu_adapter.bridges import sglang_diffusion as diffusion


@pytest.fixture
def detection(monkeypatch):
    torch = SimpleNamespace(
        version=SimpleNamespace(hip='test-hip'),
        cuda=SimpleNamespace(device_count=lambda: 2),
    )
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(diffusion.hcu_env, 'resolve', lambda name: ('cuda', 'default'))
    monkeypatch.setattr(diffusion.hcu_env, 'arch_name', lambda: 'gfx936')
    return torch


def test_activation_requires_actual_supported_hip_identity(detection, monkeypatch, caplog):
    assert diffusion.platform_plugin() == diffusion._PLATFORM_CLASS
    assert not caplog.records

    monkeypatch.setattr(diffusion.hcu_env, 'arch_name', lambda: 'gfx950')
    assert diffusion.platform_plugin() is None
    [(name, level, message)] = caplog.record_tuples
    assert name == 'hcu.adapter' and level == logging.WARNING
    assert 'requires gfx936' in message and 'detected: gfx950' in message
    assert all(symbol in message for symbol in ('mul_rn_f32', 'div_rn_f32', 'rsqrt_approx_f32'))

    caplog.clear()
    monkeypatch.setattr(diffusion.hcu_env, 'arch_name', lambda: 'gfx936')
    detection.version.hip = None
    assert diffusion.platform_plugin() is None
    assert not caplog.records


def test_hidden_devices_and_legacy_policy_do_not_activate(detection, monkeypatch):
    detection.cuda.device_count = lambda: 0
    assert diffusion.platform_plugin() is None
    monkeypatch.setattr(diffusion.hcu_env, 'resolve', lambda name: ('hip', 'explicit'))
    detection.cuda.device_count = lambda: (_ for _ in ()).throw(AssertionError('GPU probe'))
    assert diffusion.platform_plugin() is None


def test_missing_architecture_is_not_a_silent_fallback(detection, monkeypatch):
    monkeypatch.setattr(diffusion.hcu_env, 'arch_name', lambda: '')
    with pytest.raises(RuntimeError, match='visible HIP architecture'):
        diffusion.platform_plugin()


def test_cold_import_keeps_runtime_libraries_lazy(tmp_path):
    code = '''
import importlib.abc,sys
assert not any(name in sys.modules for name in ('torch','sglang','vllm','aiter','triton'))
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','sglang','vllm','aiter','triton'}:
   raise AssertionError(fullname)
sys.meta_path.insert(0,Deny())
from hcu_adapter.bridges import sglang_diffusion
from hcu_adapter.bridges import sglang
assert callable(sglang.hcu_diffusion_platform_plugin)
assert not any(name in sys.modules for name in ('torch','sglang','vllm','aiter','triton'))
'''
    (tmp_path / 'sitecustomize.py').write_text('')
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def native_platform(monkeypatch):
    pytest.importorskip('sglang.multimodal_gen.runtime.platforms.cuda')
    monkeypatch.setattr(diffusion, '_PLATFORM_TYPE', None)
    return diffusion.get_platform_class()


def test_native_oot_contract_keeps_hardware_metadata(native_platform, monkeypatch):
    import torch
    from sglang.multimodal_gen.runtime.platforms import Platform, PlatformEnum

    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device_id=0: (9, 3))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device_id=0: 'BW')
    hip = torch.version.hip
    platform = native_platform()
    assert isinstance(platform, Platform)
    assert platform._enum is PlatformEnum.OOT
    assert platform.is_out_of_tree() and platform.is_cuda() and platform.is_cuda_alike()
    assert not platform.is_rocm()
    assert platform.get_dispatch_key_name() == 'cuda'
    assert platform.get_torch_library_dispatch_key() == 'CUDA'
    assert platform.get_torch_distributed_backend_str() == 'nccl'
    assert platform.get_device(0) == torch.device('cuda', 0)
    assert platform.get_device(1) == torch.device('cuda', 1)
    assert platform.get_all_to_all_communicator_cls() == platform.get_device_communicator_cls()
    assert 'cuda_communicator' in platform.get_device_communicator_cls()
    assert platform.supports_distributed_device_id()
    assert platform.get_device_capability() == (9, 3)
    assert platform.get_device_name() == 'BW'
    assert torch.version.hip == hip
    assert not platform.device_shares_host_memory()


def test_vae_uses_original_module_without_cuda_only_optimizers(native_platform, monkeypatch):
    from sglang.multimodal_gen.runtime.platforms.cuda import NonNvmlCudaPlatform

    def reject(*args):
        raise AssertionError('CUDA-only VAE optimizer')

    monkeypatch.setattr(NonNvmlCudaPlatform, 'optimize_vae', reject)
    module = object()
    assert native_platform.optimize_vae(module) is module


def test_memory_query_preserves_the_native_argument_contract(native_platform, monkeypatch):
    from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform

    calls = []

    def memory(cls, device_id=None, distributed=False, empty_cache=True, cpu_group=None):
        calls.append((cls, device_id, distributed, empty_cache, cpu_group))
        return 55.0

    monkeypatch.setattr(RocmPlatform, 'get_available_gpu_memory', classmethod(memory))
    group = object()
    assert native_platform.get_available_gpu_memory(1, True, False, group) == 55.0
    assert calls == [(native_platform, 1, True, False, group)]


def test_backend_init_registers_the_public_seam_before_use(native_platform, monkeypatch):
    from hcu_adapter.bridges import sglang_diffusion_numerics

    calls = []
    monkeypatch.setattr(sglang_diffusion_numerics, 'install', lambda: calls.append('numerics'))
    monkeypatch.setattr(diffusion, 'install_collective_policy', lambda: calls.append('collective'))
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    native_platform().init_backend()
    assert calls == ['numerics', 'collective']
    record = registry.get('platform_detect')
    assert record['implementation'] == 'cuda'
    assert record['mechanism'] == registry.MECH_SEAM
    assert record['target'] == 'sglang.multimodal_gen.runtime.platforms.current_platform'


def test_registration_failure_aborts_backend_initialization(native_platform, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError('registration rejected')

    monkeypatch.setattr(registry, 'register', reject)
    with pytest.raises(ValueError, match='registration rejected'):
        native_platform().init_backend()


@pytest.mark.parametrize(
    'requested, expected_class',
    [('FA', 'HcuFlashAttentionBackend'), ('FA2', 'HcuFlashAttention2Backend')],
)
def test_component_attention_selection_preserves_requested_backend(
    native_platform, monkeypatch, requested, expected_class
):
    import torch
    from pkgutil import resolve_name
    from sglang.multimodal_gen.runtime import platforms
    from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
        AttentionRequirements,
    )
    from sglang.multimodal_gen.runtime.layers.attention.selector import (
        _cached_get_attn_backend,
        component_attn_backend_context_manager,
        get_attn_backend,
    )

    from hcu_adapter.infra import kill_switch

    pytest.importorskip('flash_attn')
    monkeypatch.setattr(kill_switch, 'op_disabled', lambda op: False)
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    monkeypatch.setattr(platforms, 'current_platform', native_platform())
    _cached_get_attn_backend.cache_clear()
    requested_backend = platforms.AttentionBackendEnum[requested]
    try:
        with component_attn_backend_context_manager(
            requested_backend, component_name='transformer'
        ):
            backend = get_attn_backend(
                128,
                torch.bfloat16,
                attention_requirements=AttentionRequirements(packed_varlen=True),
            )
    finally:
        _cached_get_attn_backend.cache_clear()

    assert backend is resolve_name(
        f'hcu_adapter.bridges.sglang_diffusion_fa2.{expected_class}'
    )
    assert backend.get_enum() is requested_backend
    assert backend.supports_packed_varlen()
    if requested == 'FA':
        from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
            FlashAttentionImpl,
        )

        impl = backend.get_impl_cls()
        assert backend.supports_ring_rotation()
        assert impl.forward is FlashAttentionImpl.forward
        assert impl.forward_ring_kv_chunk is FlashAttentionImpl.forward_ring_kv_chunk
    assert registry.get('diffusion_fa2')['mechanism'] == registry.MECH_SEAM


@pytest.mark.parametrize('requested', [None, 'TORCH_SDPA'])
def test_other_attention_backends_delegate_to_upstream(native_platform, monkeypatch, requested):
    import torch
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang.multimodal_gen.runtime.platforms.cuda import NonNvmlCudaPlatform

    requested = AttentionBackendEnum[requested] if requested is not None else None
    calls = []

    def resolve(cls, backend, head_size, dtype):
        calls.append((cls, backend, head_size, dtype))
        return 'upstream.Backend'

    monkeypatch.setattr(NonNvmlCudaPlatform, 'get_attn_backend_cls_str', classmethod(resolve))
    assert native_platform.get_attn_backend_cls_str(requested, 128, torch.bfloat16) == 'upstream.Backend'
    assert calls == [(native_platform, requested, 128, torch.bfloat16)]


def test_vendor_kill_switch_delegates_fa_and_rejects_fa2(native_platform, monkeypatch):
    import torch
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    from hcu_adapter.infra import kill_switch

    from sglang.multimodal_gen.runtime.platforms.cuda import NonNvmlCudaPlatform

    monkeypatch.setattr(kill_switch, 'op_disabled', lambda op: op == 'vendor_fa')
    monkeypatch.setattr(
        NonNvmlCudaPlatform,
        'get_attn_backend_cls_str',
        classmethod(lambda cls, backend, head_size, dtype: 'upstream.Backend'),
    )
    assert (
        native_platform.get_attn_backend_cls_str(AttentionBackendEnum.FA, 128, torch.bfloat16)
        == 'upstream.Backend'
    )
    with pytest.raises(ValueError, match='disabled by vendor_fa'):
        native_platform.get_attn_backend_cls_str(AttentionBackendEnum.FA2, 128, torch.bfloat16)
