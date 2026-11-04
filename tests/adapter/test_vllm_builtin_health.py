# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU host-boundary contracts for combined built-in compatibility owners.

Run with HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES='' and this checkout on
PYTHONPATH: python -m pytest tests/adapter/test_vllm_builtin_health.py.
Real installed public selectors/spawn/fork are verified by separate probes.
"""
from types import SimpleNamespace

import pytest

from hcu_adapter.bridges import vllm as host
from hcu_adapter.bridges import vllm_mhc, vllm_sparse_indexer


@pytest.mark.parametrize('failed_owner', [None, 'mhc', 'indexer'])
def test_runtime_health_checks_each_builtin_owner(monkeypatch, failed_owner):
    calls = []

    def check(name):
        calls.append(name)
        if name == failed_owner:
            raise RuntimeError(name + ' owner unhealthy')

    monkeypatch.setattr(host, '_RUNTIME_ACTIONS',
                        SimpleNamespace(require_runtime_healthy=lambda: check('host')))
    monkeypatch.setattr(vllm_mhc, 'require_healthy', lambda: check('mhc'))
    monkeypatch.setattr(vllm_sparse_indexer, 'require_healthy', lambda: check('indexer'))
    if failed_owner is None:
        host.require_runtime_healthy()
    else:
        with pytest.raises(RuntimeError, match=failed_owner + ' owner unhealthy'):
            host.require_runtime_healthy()
    expected = ['host', 'mhc'] if failed_owner == 'mhc' else ['host', 'mhc', 'indexer']
    assert calls == expected
