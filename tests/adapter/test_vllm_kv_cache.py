"""Framework-free registry ownership and neutral-path contracts."""
import importlib
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hcu_adapter.bridges import vllm_kv_cache as registry


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(registry, '_provider', None)


def test_empty_registry_preserves_default_and_config_identity():
    config, cache = object(), SimpleNamespace()
    assert registry.select_kv_cache_builder(config, 'native.Builder') == 'native.Builder'
    assert registry.localize_kv_cache_config(config, cache) is cache


def test_owner_registration_is_atomic_and_idempotent():
    pick = lambda config: 'extension.Builder'
    transform = lambda config, cache: cache
    registry.register_kv_cache_layout('extension', pick, transform)
    registry.register_kv_cache_layout('extension', pick, transform)
    registry.require_kv_cache_layout('extension', pick, transform)
    with pytest.raises(RuntimeError):
        registry.register_kv_cache_layout('foreign', pick, transform)
    with pytest.raises(RuntimeError):
        registry.require_kv_cache_layout('extension', pick, lambda config, cache: cache)
    assert registry.select_kv_cache_builder(object(), 'native.Builder') == 'extension.Builder'


def test_inactive_provider_falls_through():
    registry.register_kv_cache_layout('extension', lambda config: None, lambda config, cache: cache)
    assert registry.select_kv_cache_builder(object(), 'native.Builder') == 'native.Builder'


def test_invalid_callback_contracts_are_rejected():
    with pytest.raises(TypeError):
        registry.register_kv_cache_layout('extension', None, lambda config, cache: cache)
    assert registry._provider is None
    registry.register_kv_cache_layout('extension', lambda config: 42, lambda config, cache: None)
    with pytest.raises(TypeError):
        registry.select_kv_cache_builder(object(), 'native.Builder')
    with pytest.raises(TypeError):
        registry.localize_kv_cache_config(object(), SimpleNamespace())


def test_registry_import_and_dispatch_are_framework_free():
    code = '''
import sys
from importlib.abc import MetaPathFinder
class BlockFrameworks(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'vllm', 'sglang'):
            raise AssertionError('framework import: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
from hcu_adapter.bridges import vllm_kv_cache as r
cache = object()
r.register_kv_cache_layout('test', lambda cfg: 'plugin.Builder', lambda cfg, c: c)
r.require_kv_cache_layout('test', r._provider[1], r._provider[2])
assert r.select_kv_cache_builder(None, 'default') == 'plugin.Builder'
assert r.localize_kv_cache_config(None, cache) is cache
'''
    env = dict(os.environ, HCU_ADAPTER_HOST='none', VLLM_PLUGINS='__disabled__')
    subprocess.run([sys.executable, '-c', code], env=env, check=True,
                   capture_output=True, text=True, timeout=30)
