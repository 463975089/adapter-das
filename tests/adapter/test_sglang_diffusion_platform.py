# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import os
from types import SimpleNamespace

import pytest

from hcu_adapter import registry, sglang_diffusion as diffusion


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


def test_activation_requires_actual_supported_hip_identity(detection, monkeypatch):
    assert diffusion.platform_plugin() == diffusion._PLATFORM_CLASS
    monkeypatch.setattr(diffusion.hcu_env, 'arch_name', lambda: 'gfx950')
    assert diffusion.platform_plugin() is None
    monkeypatch.setattr(diffusion.hcu_env, 'arch_name', lambda: 'gfx936')
    detection.version.hip = None
    assert diffusion.platform_plugin() is None


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
from hcu_adapter import sglang_diffusion
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
    monkeypatch.setattr(diffusion, 'install_collective_policy', lambda: False)
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    native_platform().init_backend()
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
