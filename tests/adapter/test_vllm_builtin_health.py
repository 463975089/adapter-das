# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU host-boundary contracts for combined built-in compatibility owners.

Run with HCU_ADAPTER_HOST=vllm and hidden GPU devices, with this checkout on
PYTHONPATH: python -m pytest tests/adapter/test_vllm_builtin_health.py.
Real installed public selectors/spawn/fork are verified by separate probes.
"""
from importlib import import_module

import pytest

from hcu_adapter.bridges import vllm as host


OWNERS = (
    ('vllm_mhc', 'mhc'),
    ('vllm_sparse_indexer', 'indexer'),
    ('vllm_flashmla', 'prefill'),
    ('vllm_flashmla_decode', 'decode'),
    ('vllm_pcp', 'pcp'),
)


def _owners(monkeypatch, calls, failed_owner=None):
    def check(name):
        calls.append(name)
        if name == failed_owner:
            raise RuntimeError(name + ' owner unhealthy')

    for module, name in OWNERS:
        monkeypatch.setattr(import_module('hcu_adapter.bridges.' + module),
                            'require_healthy', lambda name=name: check(name))


@pytest.mark.parametrize('completed', [False, True])
@pytest.mark.parametrize('failed_owner', [None, 'host', *(name for _, name in OWNERS)])
def test_runtime_health_checks_each_builtin_owner(monkeypatch, failed_owner, completed):
    calls = []
    actions = host.ProcessLocalActions()
    monkeypatch.setattr(host, '_RUNTIME_ACTIONS', actions)
    if failed_owner == 'host':
        def fail():
            raise RuntimeError('host owner unhealthy')
        with pytest.raises(host.LatchedVllmRuntimeError, match='host owner unhealthy'):
            actions.run_once('vllm-runtime', fail)
    elif completed:
        actions.run_once('vllm-runtime', lambda: None)

    original = actions.require_runtime_healthy
    def check_host():
        calls.append('host')
        original()

    monkeypatch.setattr(actions, 'require_runtime_healthy', check_host)
    _owners(monkeypatch, calls, failed_owner)
    if failed_owner is None:
        host.require_runtime_healthy()
    else:
        with pytest.raises(RuntimeError, match=failed_owner + ' owner unhealthy'):
            host.require_runtime_healthy()
    expected = ['host', *(name for _, name in OWNERS)]
    if failed_owner is not None:
        expected = expected[:expected.index(failed_owner) + 1]
    assert calls == expected


def test_owner_aggregation_is_deferred_only_during_runtime_action(monkeypatch):
    calls = []
    actions = host.ProcessLocalActions()
    monkeypatch.setattr(host, '_RUNTIME_ACTIONS', actions)
    _owners(monkeypatch, calls, 'decode')

    def activate():
        assert actions.runtime_action_in_progress()
        host.require_runtime_healthy()
        assert calls == []

    actions.run_once('vllm-runtime', activate)
    assert not actions.runtime_action_in_progress()
    with pytest.raises(RuntimeError, match='decode owner unhealthy'):
        host.require_runtime_healthy()
    assert calls == ['mhc', 'indexer', 'prefill', 'decode']


def test_host_failure_is_refused_even_inside_running_callback(monkeypatch):
    calls = []
    actions = host.ProcessLocalActions()
    monkeypatch.setattr(host, '_RUNTIME_ACTIONS', actions)
    _owners(monkeypatch, calls)

    rendered = []

    class Failure(RuntimeError):
        def __str__(self):
            assert actions.runtime_action_in_progress()
            with pytest.raises(host.LatchedVllmRuntimeError):
                host.require_runtime_healthy()
            rendered.append(True)
            return 'host owner unhealthy'

    def fail():
        raise Failure()

    with pytest.raises(host.LatchedVllmRuntimeError):
        actions.run_once('vllm-runtime', fail)

    with pytest.raises(host.LatchedVllmRuntimeError, match='host owner unhealthy'):
        host.require_runtime_healthy()
    assert rendered == [True] and calls == []
