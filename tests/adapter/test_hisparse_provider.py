# SPDX-License-Identifier: Apache-2.0
"""CPU-only loader contracts. No device API is used, even when HIP is installed."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def provider(monkeypatch):
    from hcu_adapter.kernels import hisparse
    from hcu_adapter.infra import kill_switch, registry
    module = importlib.reload(hisparse)
    monkeypatch.setattr(kill_switch, 'op_disabled', lambda op: False)
    records = []
    monkeypatch.setattr(registry, 'register', lambda *args: records.append(args))
    present = set()
    implementations = set()
    schemas = dict(zip(module._NAMES, module.SCHEMAS))
    calls = []

    def canonical(name):
        return '_C_cache_ops::' + schemas[name]

    def find(name, overload):
        short = name.split('::')[1]
        if short not in present:
            raise RuntimeError('absent')
        return types.SimpleNamespace(schema=lambda: canonical(short))

    def load(path):
        calls.append(path)
        present.update(module._NAMES)
        implementations.update(module._NAMES)

    fake = types.SimpleNamespace(
        version=types.SimpleNamespace(hip='7.2'),
        _C=types.SimpleNamespace(
            _dispatch_find_schema_or_throw=find,
            parse_schema=lambda s: s,
            _dispatch_has_kernel_for_dispatch_key=lambda name, key:
                name.split('::')[1] in implementations and key == 'CUDA'),
        ops=types.SimpleNamespace(load_library=load))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    real_import = importlib.import_module
    monkeypatch.setattr(module.importlib, 'import_module',
                        lambda name: object() if name == 'vllm._C' else real_import(name))
    return module, fake, present, implementations, schemas, calls, records


def test_import_is_light_and_no_compile():
    code = '''import json,sys
from hcu_adapter.kernels import hisparse, hisparse_cpp
assert not any(n in sys.modules for n in ('torch','vllm','sgl_kernel','torch.utils.cpp_extension'))
print(json.dumps({'names':hisparse._NAMES}))'''
    p = subprocess.run([sys.executable, '-I', '-S', '-c',
                        'import sys;sys.path.insert(0,' + repr(str(ROOT)) + ');' + code],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert len(json.loads(p.stdout)['names']) == 4


def test_explicit_load_idempotent(provider, tmp_path):
    m, fake, present, implementations, schemas, calls, records = provider
    artifact = tmp_path / 'hisparse.so'
    artifact.touch()
    m.ensure_hisparse_ops(artifact)
    m.ensure_hisparse_ops()
    assert calls == [str(artifact.resolve())]
    assert len(records) == 1
    assert present == set(m._NAMES)
    assert implementations == set(m._NAMES)


def test_existing_native_set_is_not_replaced(provider):
    m, fake, present, implementations, schemas, calls, records = provider
    present.update(m._NAMES)
    implementations.update(m._NAMES)
    m.ensure_hisparse_ops()
    assert calls == [] and records == []


def test_partial_set_refused_and_failure_latched(provider):
    m, fake, present, implementations, schemas, calls, records = provider
    present.add(m._NAMES[0])
    implementations.add(m._NAMES[0])
    with pytest.raises(RuntimeError, match='Partial'):
        m.ensure_hisparse_ops()
    present.clear()
    implementations.clear()
    with pytest.raises(RuntimeError, match='previously failed'):
        m.ensure_hisparse_ops()
    assert calls == []


def test_schema_without_dispatch_refused(provider):
    m, fake, present, implementations, schemas, calls, records = provider
    present.add(m._NAMES[0])
    with pytest.raises(RuntimeError, match='without HIP/CUDA'):
        m.ensure_hisparse_ops()
    assert calls == []


def test_schema_drift_refused(provider):
    m, fake, present, implementations, schemas, calls, records = provider
    present.update(m._NAMES)
    implementations.update(m._NAMES)
    fake._C.parse_schema = lambda s: s + 'drift'
    with pytest.raises(RuntimeError, match='schema drift'):
        m.ensure_hisparse_ops()
    assert calls == []


def test_non_hip_refused(provider):
    m, fake, *rest = provider
    fake.version.hip = None
    with pytest.raises(RuntimeError, match='requires HIP'):
        m.ensure_hisparse_ops()


def test_absent_artifact_is_not_jit_compiled(provider, tmp_path):
    m, fake, present, implementations, schemas, calls, records = provider
    with pytest.raises(FileNotFoundError, match='absent'):
        m.ensure_hisparse_ops(tmp_path / 'missing.so')
    assert calls == []


def test_kill_switch_precedes_publication(provider, monkeypatch):
    from hcu_adapter.infra import kill_switch
    m, fake, *rest = provider
    monkeypatch.setattr(kill_switch, 'op_disabled', lambda name: name == 'vllm_hisparse')
    with pytest.raises(RuntimeError, match='explicitly disabled'):
        m.ensure_hisparse_ops()
    assert not m._LOADED


def test_fork_owner_checked_before_lock(provider):
    m, fake, *rest = provider
    m._PID = -1
    class PoisonLock:
        def __enter__(self):
            raise AssertionError('inherited lock acquired')
    m._LOCK = PoisonLock()
    with pytest.raises(RuntimeError, match='inherited across fork'):
        m.ensure_hisparse_ops()


def test_artifact_drift_refused(provider, tmp_path):
    m, fake, *rest = provider
    artifact = tmp_path / 'a.so'
    artifact.touch()
    m.ensure_hisparse_ops(artifact)
    with pytest.raises(RuntimeError, match='cannot change'):
        m.ensure_hisparse_ops(tmp_path / 'b.so')


def test_native_source_has_no_ptx_or_subgroup_dependency():
    source = (ROOT / 'csrc/hisparse/hisparse_cpp.cu').read_text(encoding='utf-8')
    for forbidden in ('asm volatile', '__ballot', '__shfl', 'warp_inclusive_scan',
                      '__stcg', '__ldcg', 'torch::stable', '__syncwarp'):
        assert forbidden not in source
    assert 'hipHostGetDevicePointer' in source
    assert 'hipPointerGetAttributes' in source
    assert 'mapped_host_spans(host_cache)' in source
    assert 'hipDeviceAttributeMaxSharedMemoryPerBlock' in source
    assert 'check_grid' in source and 'same_device' in source
    assert 'TORCH_LIBRARY_FRAGMENT(_C_cache_ops' in source
    assert 'TORCH_LIBRARY_IMPL(_C_cache_ops, CUDA' in source


def test_schemas_are_core033_target_only(provider):
    m, *_ = provider
    assert len(m.SCHEMAS) == 4
    resolve = m.SCHEMAS[0]
    assert 'owner' not in resolve and 'valid_counts=None' in resolve
    assert 'resident_null_block=0' in resolve
    assert all(schema.endswith('-> ()') for schema in m.SCHEMAS)


def test_concurrent_explicit_load_publishes_once(provider, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    m, fake, present, implementations, schemas, calls, records = provider
    path = tmp_path / 'hisparse.so'
    path.touch()
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: m.ensure_hisparse_ops(path), range(12)))
    assert calls == [str(path.resolve())] and len(records) == 1


def test_library_load_error_is_not_retried(provider, tmp_path):
    m, fake, *rest = provider
    path = tmp_path / 'broken.so'
    path.touch()
    def broken(path):
        raise OSError('incompatible Torch ABI')
    fake.ops.load_library = broken
    with pytest.raises(OSError, match='incompatible Torch ABI'):
        m.ensure_hisparse_ops(path)
    with pytest.raises(RuntimeError, match='previously failed'):
        m.ensure_hisparse_ops(path)


def test_incomplete_native_fragment_refused(provider, tmp_path):
    m, fake, *rest = provider
    path = tmp_path / 'empty.so'
    path.touch()
    fake.ops.load_library = lambda path: None
    with pytest.raises(RuntimeError, match='did not register all four'):
        m.ensure_hisparse_ops(path)


def test_package_discovery_does_not_import_sgl_kernel(provider, monkeypatch, tmp_path):
    m, fake, *rest = provider
    path = tmp_path / 'hisparse_ops.so'
    path.touch()
    calls = []
    def find(name, locations):
        calls.append(name)
        if name == 'sgl_kernel':
            return types.SimpleNamespace(submodule_search_locations=[str(tmp_path)])
        assert name == 'sgl_kernel.hisparse_ops'
        return types.SimpleNamespace(origin=str(path))
    monkeypatch.setattr(m.importlib.machinery.PathFinder, 'find_spec', find)
    assert m._packaged_library() == str(path.resolve())
    assert calls == ['sgl_kernel', 'sgl_kernel.hisparse_ops']
