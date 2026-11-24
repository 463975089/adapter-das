# SPDX-License-Identifier: Apache-2.0
"""Lazy provider ABI and failure propagation without a GPU."""
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest


def bridge():
    explicit = os.environ.get('HCU_EXPERT_GATHER_BRIDGE_SOURCE')
    if explicit is None:
        return importlib.import_module('hcu_adapter.bridges.vllm_expert_gather')
    spec = importlib.util.spec_from_file_location('expert_gather_candidate', Path(explicit))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_does_not_load_provider(monkeypatch):
    class Missing:
        def __getattr__(self, name):
            pytest.fail('Provider imported before dispatch: '+name)
    monkeypatch.setitem(sys.modules, 'lightop', Missing())
    assert callable(bridge().unpermute_and_reduce)


def test_native_named_arguments_reach_provider_in_order(monkeypatch):
    parent = ModuleType('lightop'); parent.__path__ = []
    moe = ModuleType('lightop.moe')
    calls = []
    moe.ep_gather = lambda *args: calls.append(args)
    monkeypatch.setitem(sys.modules, 'lightop', parent)
    monkeypatch.setitem(sys.modules, 'lightop.moe', moe)
    args = tuple(object() for _ in range(6))
    result = bridge().unpermute_and_reduce(a=args[0], topk_ids=args[1],
        topk_weights=args[2], inv_perm=args[3], expert_map=args[4], output=args[5])
    assert calls == [args] and result is None


def test_provider_failure_propagates_without_retry(monkeypatch):
    parent = ModuleType('lightop'); parent.__path__ = []
    moe = ModuleType('lightop.moe')
    error = RuntimeError('provider execution failed')
    calls = []
    def fail(*args):
        calls.append(args)
        raise error
    moe.ep_gather = fail
    monkeypatch.setitem(sys.modules, 'lightop', parent)
    monkeypatch.setitem(sys.modules, 'lightop.moe', moe)
    with pytest.raises(RuntimeError) as caught:
        bridge().unpermute_and_reduce(*[None]*6)
    assert caught.value is error and len(calls) == 1
