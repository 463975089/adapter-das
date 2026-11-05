# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Execute installed provider getter source, without importing/booting AITER.

Only the two real getter definitions and the real SMALL_MESSAGE RHS execute.
This proves Python input consumption, not a constructor/native/GPU execution.
"""
import ast
from importlib import metadata
import os
from pathlib import Path

import pytest

from test_vllm_allreduce import isolated, provider, enable, compat, envs, adapter_registry

INPUTS = [('AITER_AR_MAX_SIZE_MB', '96', 96 * 1024**2),
          ('AITER_AR_TRANSPORT', 'fabric', 'fabric'),
          ('AITER_AR_SMALL_MESSAGE', 'll', 'll')]


@pytest.fixture
def provider_read():
    path = Path(metadata.distribution('aiter').locate_file(
        'aiter/dist/device_communicators/custom_all_reduce.py'))
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {'_requested_ar_max_size', '_requested_ar_transport'}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names
    namespace = {'os': os, '_DEFAULT_AR_MAX_SIZE_MB': 64}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    small = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == 'requested_small_ar' for target in node.targets)]
    assert len(small) == 1
    expression = compile(ast.Expression(body=small[0]), str(path), 'eval')
    return lambda name: (namespace['_requested_ar_max_size']() if name == INPUTS[0][0]
                         else namespace['_requested_ar_transport']() if name == INPUTS[1][0]
                         else eval(expression, {'os': os}))


def configured(monkeypatch, tmp_path, name, value):
    from hcu_adapter import hcu_config
    # Current catalog exposes enable, not these provider controls. A test-only
    # compound catalog row exercises real HCU_CONFIG parsing/resolution without
    # adding a production configuration surface or monkeypatching resolve.
    spec = dict(hcu_config.BACKEND_CATALOG['vllm_allreduce'])
    spec['vars'] = {'official-aiter': {'HCU_AITER_AR': '1', name: value}}
    monkeypatch.setitem(hcu_config.BACKEND_CATALOG, 'vllm_allreduce', spec)
    path = tmp_path / 'provider-input.json'
    path.write_text('{"version":1,"ops":{"vllm_allreduce":"official-aiter"}}')
    monkeypatch.setenv('HCU_CONFIG', str(path))
    monkeypatch.setattr(hcu_config, '_STATE', dict(loaded=False, path=None, ops={}, overrides={}, hosts={}))


@pytest.mark.parametrize('name,value,expected', INPUTS)
def test_config_only_provider_input_refused_before_mutation(monkeypatch, tmp_path, provider, provider_read, name, value, expected):
    from hcu_adapter import hcu_env
    enable(monkeypatch)
    configured(monkeypatch, tmp_path, name, value)
    assert hcu_env.resolve(name, host='vllm') == (value, 'config')
    assert provider_read(name) != expected  # Actual installed provider source ignores config.
    before = dict(envs.environment_variables)
    with pytest.raises(compat.AllReduceCompatibilityError, match='provider environment'):
        compat.install()
    assert compat._installed is None and compat._commit_failure is None
    assert dict(envs.environment_variables) == before and adapter_registry.records() == {}
    assert provider[0] == []


@pytest.mark.parametrize('name,value,expected', INPUTS)
def test_explicit_environment_wins_config_and_reaches_provider_getter(monkeypatch, tmp_path, provider, provider_read, name, value, expected):
    from hcu_adapter import allreduce, hcu_env
    enable(monkeypatch)
    configured(monkeypatch, tmp_path, name, '128' if name == INPUTS[0][0] else 'auto')
    monkeypatch.setenv(name, value)
    assert hcu_env.resolve(name, host='vllm') == (value, 'explicit')
    assert provider_read(name) == expected
    policy = allreduce.requested_policy()
    field = {'AITER_AR_MAX_SIZE_MB': 'admission_bytes', 'AITER_AR_TRANSPORT': 'transport', 'AITER_AR_SMALL_MESSAGE': 'small_message'}[name]
    assert getattr(policy, field) == expected
    if name == 'AITER_AR_TRANSPORT':
        provider[1]['fabric'] = True
    compat.install()


@pytest.mark.parametrize('name,value,expected', INPUTS)
def test_empty_provider_environment_refused_before_mutation(monkeypatch, provider, name, value, expected):
    enable(monkeypatch)
    monkeypatch.setenv(name, '')  # resolve treats empty as unset; provider does not.
    with pytest.raises(compat.AllReduceCompatibilityError, match='provider environment'):
        compat.install()
    assert compat._installed is None and compat._commit_failure is None


def test_provider_defaults_match_policy_without_importing_provider(provider_read):
    from hcu_adapter import allreduce
    assert provider_read('AITER_AR_MAX_SIZE_MB') == allreduce.Policy(True).admission_bytes
    assert provider_read('AITER_AR_TRANSPORT') == allreduce.Policy(True).transport
    assert provider_read('AITER_AR_SMALL_MESSAGE') == allreduce.Policy(True).small_message


@pytest.mark.parametrize('name,value,expected', [
    ('AITER_AR_MAX_SIZE_MB', '64', 64 * 1024**2),
    ('AITER_AR_TRANSPORT', 'ipc', 'ipc'),
    ('AITER_AR_SMALL_MESSAGE', 'off', 'off'),
])
def test_config_matching_provider_default_is_safe(monkeypatch, tmp_path, provider, provider_read, name, value, expected):
    enable(monkeypatch)
    configured(monkeypatch, tmp_path, name, value)
    assert provider_read(name) == expected
    assert compat.install().selected


def test_kill_ignores_unconsumable_provider_config(monkeypatch, tmp_path, provider):
    enable(monkeypatch)
    configured(monkeypatch, tmp_path, 'AITER_AR_TRANSPORT', 'fabric')
    monkeypatch.setenv('HCU_DISABLE_OPS', 'aiter_ar')
    result = compat.install(require_available=True)
    assert result.available and not result.selected and not result.fused_ops_registered
    assert provider[0] == []


def test_late_config_drift_refuses_selection_and_stays_hashed(monkeypatch, tmp_path, provider):
    import json
    enable(monkeypatch)
    compat.install()
    configured(monkeypatch, tmp_path, 'AITER_AR_SMALL_MESSAGE', 'll')
    with pytest.raises(compat.AllReduceCompatibilityError, match='provider environment'):
        compat.state()
    assert 'provider environment' in json.loads(compat.effective_compile_factor())['implementation']['invalid']
    assert envs.compile_factors()[compat.FACTOR] == compat.effective_compile_factor()
