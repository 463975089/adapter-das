# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real runtime health and profile owner, with unrelated providers isolated."""
import os
from types import SimpleNamespace as NS

import pytest

from hcu_adapter.bridges import vllm as bridge
from hcu_adapter.bridges import vllm_model_runner as runners
from hcu_adapter.features.hy4 import pcp_profile as profile
from hcu_adapter.infra import registry
from test_vllm_bridge import _stub_runtime_owners


@pytest.fixture
def startup(monkeypatch):
    monkeypatch.setattr(runners, '_factories', {})
    monkeypatch.setattr(runners, '_frozen', False)
    monkeypatch.setattr(runners, '_pid', os.getpid())
    monkeypatch.setattr(profile, '_registration', None)
    monkeypatch.setattr(profile, '_failure', None)
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    monkeypatch.setenv('HCU_DISABLE_OPS', '')
    monkeypatch.setattr(bridge, '_RUNTIME_ACTIONS', bridge.ProcessLocalActions())
    monkeypatch.setattr(bridge, 'detect_hcu',
        lambda: bridge.HcuDetection(True, 'gfx938', 'CPU contract'))
    monkeypatch.setattr(bridge, '_ensure_vllm_compatible', lambda: '0.32.0+cpu')
    for name in ('_ensure_hcu_platform_selected', 'configure_vllm_hip_runtime_path',
                 '_require_vllm_hip_runtime_abi'):
        monkeypatch.setattr(bridge, name, lambda: None)
    monkeypatch.setattr(bridge, '_installed_adapter_version', lambda: 'cpu')
    monkeypatch.setattr(bridge, 'activate_extensions', lambda *a, **k: ())
    _stub_runtime_owners(monkeypatch, preserve=('pcp_profile',))


def test_runtime_without_a_model_does_not_publish_hy4_factory(startup):
    bridge.hcu_runtime_plugin()
    assert runners._factories == {}
    assert profile._registration is None


def test_unselected_hy4_factory_does_not_poison_healthy_pid_change(startup, monkeypatch):
    bridge.hcu_runtime_plugin()
    child_pid = os.getpid() + 1000
    monkeypatch.setattr(os, 'getpid', lambda: child_pid)
    bridge.require_runtime_healthy()
    bridge.hcu_runtime_plugin()
    assert runners._factories == {}


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires actual fork')
def test_real_healthy_fork_keeps_unselected_profile_owner_idle(startup):
    from test_vllm_commit_fatal import child_result

    bridge.hcu_runtime_plugin()

    def child():
        bridge.require_runtime_healthy()
        bridge.hcu_runtime_plugin()
        assert runners._factories == {}
        assert profile._registration is None

    child_result(child)


def test_runtime_owner_helper_isolates_profile_across_registry_resets(startup, monkeypatch):
    _stub_runtime_owners(monkeypatch)
    for _ in range(2):
        monkeypatch.setattr(registry, '_RECORDS', {})
        monkeypatch.setattr(registry, '_BOUND', {})
        monkeypatch.setattr(bridge, '_RUNTIME_ACTIONS', bridge.ProcessLocalActions())
        bridge.hcu_runtime_plugin()
        # The helper must isolate both entry points, including after an AR
        # fixture has discarded the previous process's registration records.
        profile.install()
        profile.require_healthy()


@pytest.mark.parametrize('scope', [
    'hy4', 'other_model', 'missing_model', 'pcp1', 'dcp2', 'tp2', 'graph', 'runner1',
])
def test_real_worker_class_registers_profile_only_for_its_scope(startup, monkeypatch, scope):
    """Execute the real Worker class with only the upstream base isolated."""
    import ast
    import importlib.util
    from pathlib import Path

    source = Path(importlib.util.find_spec('hcu_adapter.bridges.vllm_worker').origin)
    parsed = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'Worker')
    module = ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
        cls,
    ], type_ignores=[]))
    official = object()

    class OfficialWorker:
        def _make_model_runner(self):
            return official

    class ProfileRunner:
        def __init__(self, config, device):
            self.config, self.device = config, device

    namespace = {'OfficialWorker': OfficialWorker}
    exec(compile(module, str(source), 'exec'), namespace)
    cfg = NS(model_config=NS(hf_config=NS(model_type='hy_v4'), enforce_eager=True),
        parallel_config=NS(prefill_context_parallel_size=16,
            decode_context_parallel_size=1, tensor_parallel_size=1),
        use_v2_model_runner=True)
    if scope == 'other_model':
        cfg.model_config.hf_config.model_type = 'other'
    elif scope == 'missing_model':
        cfg.model_config = None
    elif scope == 'pcp1':
        cfg.parallel_config.prefill_context_parallel_size = 1
    elif scope == 'dcp2':
        cfg.parallel_config.decode_context_parallel_size = 2
    elif scope == 'tp2':
        cfg.parallel_config.tensor_parallel_size = 2
    elif scope == 'graph':
        cfg.model_config.enforce_eager = False
    elif scope == 'runner1':
        cfg.use_v2_model_runner = False
    monkeypatch.setattr(profile, 'runner_class', lambda: ProfileRunner)
    worker = object.__new__(namespace['Worker'])
    worker.vllm_config, worker.device = cfg, object()
    result = worker._make_model_runner()
    if scope == 'hy4':
        assert isinstance(result, ProfileRunner)
        assert result.config is cfg and result.device is worker.device
        assert set(runners._factories) == {profile.OP}
    else:
        assert result is official
        assert runners._factories == {}
