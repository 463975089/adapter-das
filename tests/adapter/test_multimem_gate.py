import ast
import inspect
import os
from pathlib import Path
import sys
import types

import pytest

from hcu_adapter import multimem_gate as gate


@pytest.fixture
def consumer(monkeypatch):
    tree = os.environ.get('HCU_TEST_SGLANG_TREE')
    if not tree:
        pytest.skip('requires explicitly selected upstream tree')
    path = Path(tree) / 'python/sglang/srt/distributed/device_communicators/triton_symm_mem_ag.py'
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'MultimemAllGatherer')
    calls = []
    parallel = types.ModuleType('sglang.srt.distributed.parallel_state')
    parallel.get_tp_group = lambda: types.SimpleNamespace(world_size=2, device_group='group', rank_in_group=0)
    parallel.in_the_same_node_as = lambda *a, **k: True
    distributed = types.ModuleType('sglang.srt.distributed')
    distributed.tensor_model_parallel_all_gather = lambda x, dim: calls.append(('gather', x, dim)) or ('native', x)
    for name in ('sglang', 'sglang.srt'):
        module = types.ModuleType(name);module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    monkeypatch.setitem(sys.modules, parallel.__name__, parallel)
    def create_state(**kwargs):
        calls.append(('unsupported-probe', kwargs))
        raise RuntimeError('unsupported symmetric memory')
    torch = types.SimpleNamespace(bfloat16='bf16', cuda=types.SimpleNamespace(is_available=lambda: False))
    namespace = dict(torch=torch, logger=types.SimpleNamespace(warning=lambda *a: None),
                     get_parallel=lambda: types.SimpleNamespace(nnodes=1), create_state=create_state,
                     _NUMEL_PER_THREAD=8)
    exec(compile('from __future__ import annotations\n'+ast.get_source_segment(source, node), str(path), 'exec'), namespace)
    x = types.SimpleNamespace(dtype='bf16', dim=lambda: 2, shape=(4, 16))
    return namespace['MultimemAllGatherer'], calls, x


def test_real_consumer_probe_counterexample_and_fallback(consumer):
    cls, calls, x = consumer
    assert cls(128)(x) == ('native', x)
    assert any(c[0] == 'unsupported-probe' for c in calls)
    calls.clear()
    gate._install_init(cls.__init__)
    for _ in range(2):
        instance = cls(128, enabled=True, skip_entry_sync=True)
        assert instance._state is None and instance._skip_entry_sync is True
        assert instance(x) == ('native', x)
    assert [c[0] for c in calls] == ['gather', 'gather']
    assert all(c[2] == -1 for c in calls)


def test_cached_constructor_alias_and_signature(consumer):
    cls, calls, x = consumer
    alias = cls.__init__;signature = inspect.signature(alias)
    assert gate._install_init(alias) is alias
    assert cls.__init__ is alias and inspect.signature(alias) == signature
    instance = object.__new__(cls)
    alias(instance, 257, enabled=True, skip_entry_sync=False)
    assert instance._max_tokens == 257 and instance._state is None
    assert instance(x) == ('native', x)
    assert gate._install_init(alias) is alias
    assert not any(c[0] == 'unsupported-probe' for c in calls)


def test_explicit_disabled_and_empty_metadata_preserved(consumer):
    cls, calls, x = consumer
    gate._install_init(cls.__init__)
    instance = cls(0, enabled=False, skip_entry_sync=False)
    assert instance._max_tokens == 0 and instance._skip_entry_sync is False
    assert instance(x) == ('native', x)
    assert not any(c[0] == 'unsupported-probe' for c in calls)


def test_constructor_drift_rejected():
    def unknown(self, max_tokens, *, enabled=True):
        self._state = enabled
    with pytest.raises(RuntimeError, match='contract changed'):
        gate._install_init(unknown)


def test_late_mutation_rejected(consumer):
    cls, _, _ = consumer
    gate._install_init(cls.__init__)
    cls.__init__.__globals__[gate._TARGET] = lambda: None
    with pytest.raises(RuntimeError, match='changed after installation'):
        gate._install_init(cls.__init__)


def test_non_hip_does_not_import_provider(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(version=types.SimpleNamespace(hip=None)))
    assert gate.install() is False


def test_hip_install_registers_real_constructor_fallback(consumer, monkeypatch):
    cls, calls, x = consumer
    from hcu_adapter import registry
    registrations = []
    monkeypatch.setattr(registry, 'register', lambda *a, **k: registrations.append((a, k)))
    provider = types.ModuleType('sglang.srt.distributed.device_communicators')
    provider.triton_symm_mem_ag = types.SimpleNamespace(MultimemAllGatherer=cls)
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(version=types.SimpleNamespace(hip='test-hip')))
    alias = cls.__init__
    assert gate.install() is True and gate.install() is True
    assert cls.__init__ is alias
    assert cls(128)(x) == ('native', x)
    assert not any(c[0] == 'unsupported-probe' for c in calls)
    assert all(r[0] == ('platform_detect', 'multimem-nccl-fallback') for r in registrations)
