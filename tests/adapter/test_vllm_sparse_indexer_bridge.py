"""Offline C4 publication/failure contracts, separate from GPU acceptance.

Run with HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES='' and this checkout on
PYTHONPATH: python -m pytest tests/adapter/test_vllm_sparse_indexer_bridge.py.
"""
import importlib
import os
import signal
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hcu_adapter.bridges import vllm as host
from hcu_adapter.bridges import vllm_sparse_indexer as module
from hcu_adapter import registry


def test_tools_import_without_arguments_frameworks_or_runtime_side_effects():
    root = Path(__file__).resolve().parents[2]
    code = '''
import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name.split('.')[0] in {'torch','vllm','sglang','hcu_adapter'}:
            raise AssertionError('probe import touched runtime: '+name)
sys.meta_path.insert(0,Block())
sys.argv=['no-probe-arguments']
for path in sys.argv[1:]:
    runpy.run_path(path,run_name='inert_probe_import')
'''
    # Feed paths independently of argv so argument parsing would fail if a
    # probe accidentally executed at import time.
    paths = [root/'tests/tools/_probe_vllm_sparse_indexer.py',
             root/'tests/tools/_probe_vllm_sparse_indexer_lifecycle.py']
    code = code.replace('for path in sys.argv[1:]:', f'for path in {[str(p) for p in paths]!r}:')
    result = subprocess.run([sys.executable, '-I', '-c', code], text=True,
                            capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def seams(monkeypatch):
    m = importlib.reload(module)
    monkeypatch.setenv('HCU_VLLM_SPARSE_INDEXER_IMPL', 'native')
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, 'gfx936', 'test'))
    monkeypatch.setattr(host, '_installed_vllm_version', lambda: '0.33.0+compatible')
    selection, calls, records = {}, [], []

    class Indexer:
        def __new__(cls):
            return object.__new__(selection.get(cls, cls))

        def __init__(self, k_cache, quant_block_size, scale_fmt, topk_tokens, head_dim,
                     max_model_len, max_total_seq_len, topk_indices_buffer,
                     skip_k_cache_insert=False, use_fp4_cache=False, compress_ratio=1,
                     candidate_blocks=None, candidate_block_size=0, candidate_write=False):
            raise AssertionError('offline allocation only; no model initialization')

        @classmethod
        def register_oot(cls, replacement):
            selection[cls] = replacement
            return replacement

        def forward_hip(self, hidden_states, q_quant, k, weights):
            raise AssertionError('offline math forbidden')

    class Replacement(Indexer):
        pass

    def paged(q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len):
        raise AssertionError('offline math forbidden')

    def prefill(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
        raise AssertionError('offline math forbidden')

    def gather(k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen, token_to_seq,
               block_tile_size=16, head_tile_size=16, cache_layout=None):
        raise AssertionError('offline math forbidden')

    def full(hidden_states, k_cache_prefix, kv_cache, q_fp8, k, weights,
             quant_block_size, scale_fmt, topk_tokens, head_dim, max_model_len,
             total_seq_lens, topk_indices_buffer, skip_k_cache_insert=False,
             compress_ratio=1, candidate_blocks=None, candidate_block_size=0,
             candidate_write=False):
        raise AssertionError('offline math forbidden')

    class Pack:
        def __call__(self, x_reshaped, out, lengths, *, N, D, Lmax, pad_value,
                     pad_is_uint8, block_t, block_d):
            raise AssertionError('offline math forbidden')

    class Unpack:
        def __call__(self, packed_reshaped, out, lengths, *, B, Lmax, D, block_t, block_d):
            raise AssertionError('offline math forbidden')

    def register(name, function, *, mutates_args, fake_impl, dispatch_key):
        calls.append((name, mutates_args, dispatch_key))

    envs = SimpleNamespace(environment_variables={})
    official = SimpleNamespace(rocm_fp8_paged_mqa_logits_triton=paged,
        rocm_fp8_mqa_logits=prefill, cp_gather_indexer_k_quant_cache_triton=gather,
        rocm_aiter_sparse_attn_indexer=full)
    modules = {
        'vllm.model_executor.layers.sparse_attn_indexer': SimpleNamespace(SparseAttnIndexer=Indexer),
        'vllm.v1.attention.ops.rocm_aiter_mla_sparse': official,
        'vllm.v1.attention.ops.common': SimpleNamespace(PackSeqTritonKernel=Pack, UnpackSeqTritonKernel=Unpack),
        'vllm.v1.worker.workspace': object(), 'vllm.forward_context': object(),
        'vllm.v1.attention.backends.mla.indexer': object(),
        'vllm.platforms': SimpleNamespace(current_platform=SimpleNamespace(dispatch_key='CUDA')),
        'vllm.envs': envs,
        'vllm.utils.torch_utils': SimpleNamespace(direct_register_custom_op=register),
    }
    monkeypatch.setattr(m, 'import_module', modules.__getitem__)
    monkeypatch.setattr(m.inspect, 'getfile', lambda obj: __file__)
    monkeypatch.setattr(m, '_declare', lambda *args: (full, object(), Replacement))
    monkeypatch.setattr(m, 'torch', SimpleNamespace(__version__='test',
        version=SimpleNamespace(hip='test'), ops=SimpleNamespace(vllm=SimpleNamespace())), raising=False)
    monkeypatch.setattr(registry, 'register', lambda **kw: records.append(kw))
    yield SimpleNamespace(m=m, original=Indexer, selection=selection, calls=calls,
        records=records, envs=envs, modules=modules, official=official)
    importlib.reload(module)


def test_stock_and_kill_precede_framework_import_and_invalid_mode(seams, monkeypatch):
    s = seams
    monkeypatch.setenv('HCU_VLLM_SPARSE_INDEXER_IMPL', 'stock')
    monkeypatch.setattr(s.m, 'import_module', lambda name: pytest.fail('stock imported '+name))
    assert not s.m.install()
    s.m.require_healthy()
    monkeypatch.setenv('HCU_VLLM_SPARSE_INDEXER_IMPL', 'invalid')
    monkeypatch.setenv('HCU_DISABLE_OPS', 'vllm_sparse_indexer')
    assert not s.m.install()
    s.m.require_healthy()


def test_publication_is_idempotent_and_mutates_only_output(seams):
    s = seams
    assert s.m.install() and s.m.install()
    assert s.calls == [(s.m.OP_NAME, ['output'], 'CUDA')]
    assert s.records[0]['op'] == 'vllm_sparse_indexer'
    assert s.records[0]['mechanism'] == registry.MECH_SEAM
    assert s.envs.environment_variables[s.m.COMPILE_FACTOR]().startswith('official-c4:')


@pytest.mark.parametrize('arch', ['gfx928', 'gfx92a', 'gfx938', 'gfx942', 'gfx950', None])
def test_unvalidated_arch_rejects_before_framework_import(seams, monkeypatch, arch):
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, arch, 'test'))
    with pytest.raises(seams.m.SparseIndexerCompatibilityError, match='gfx936'):
        seams.m.install()
    assert not seams.calls


@pytest.mark.parametrize('version', ['0.32.1', '1!0.33.0', '0.34.0'])
def test_release_line_does_not_widen_host_admission(seams, monkeypatch, version):
    monkeypatch.setattr(host, '_installed_vllm_version', lambda: version)
    with pytest.raises(seams.m.SparseIndexerCompatibilityError, match='0.33'):
        seams.m.install()
    assert not seams.calls


@pytest.mark.parametrize('target', ['forward', 'paged', 'prefill', 'gather', 'profile', 'pack'])
def test_changed_public_abi_refuses_before_publication(seams, monkeypatch, target):
    s = seams
    if target == 'forward':
        monkeypatch.setattr(s.original, 'forward_hip', lambda self, unknown: None)
    elif target == 'pack':
        monkeypatch.setattr(s.modules['vllm.v1.attention.ops.common'].PackSeqTritonKernel,
                            '__call__', lambda self, unknown: None)
    else:
        name = {'paged': 'rocm_fp8_paged_mqa_logits_triton', 'prefill': 'rocm_fp8_mqa_logits',
                'gather': 'cp_gather_indexer_k_quant_cache_triton',
                'profile': 'rocm_aiter_sparse_attn_indexer'}[target]
        monkeypatch.setattr(s.official, name, lambda unknown: None)
    with pytest.raises(s.m.SparseIndexerCompatibilityError, match='ABI'):
        s.m.install()
    assert not s.calls and not s.records


@pytest.mark.parametrize('owner', ['oot', 'torch', 'factor'])
def test_existing_owners_are_not_overwritten(seams, owner):
    s = seams
    if owner == 'oot':
        class Foreign(s.original):
            pass
        s.selection[s.original] = Foreign
    elif owner == 'torch':
        setattr(s.m.torch.ops.vllm, s.m.OP_NAME, object())
    else:
        s.envs.environment_variables[s.m.COMPILE_FACTOR] = lambda: 'foreign'
    with pytest.raises(s.m.SparseIndexerCompatibilityError, match='owner'):
        s.m.install()
    assert not s.calls


@pytest.mark.parametrize('mutation', ['clear', 'foreign', 'factor', 'policy', 'kill'])
def test_late_drift_refuses_health_and_retry_without_repair(seams, monkeypatch, mutation):
    s = seams
    s.m.install()
    if mutation == 'clear':
        s.selection.clear()
    elif mutation == 'foreign':
        class Foreign(s.original):
            pass
        s.selection[s.original] = Foreign
    elif mutation == 'factor':
        s.envs.environment_variables[s.m.COMPILE_FACTOR] = lambda: 'foreign'
    elif mutation == 'kill':
        monkeypatch.setenv('HCU_DISABLE_OPS', 'vllm_sparse_indexer')
    else:
        monkeypatch.setenv('HCU_VLLM_SPARSE_INDEXER_IMPL', 'stock')
    for action in (s.m.require_healthy, s.m.install):
        with pytest.raises(s.m.SparseIndexerCompatibilityError):
            action()
    assert len(s.calls) == 1


def test_direct_retry_with_invalid_policy_latches_before_restore(seams, monkeypatch):
    s = seams
    s.m.install()
    monkeypatch.setenv('HCU_VLLM_SPARSE_INDEXER_IMPL', 'invalid')
    with pytest.raises(s.m.SparseIndexerCompatibilityError, match='policy invalid'):
        s.m.install()
    monkeypatch.setenv('HCU_VLLM_SPARSE_INDEXER_IMPL', 'native')
    for action in (s.m.require_healthy, s.m.install):
        with pytest.raises(s.m.SparseIndexerCompatibilityError):
            action()
    assert len(s.calls) == 1


@pytest.mark.parametrize('format_error', [AssertionError, SystemExit])
def test_partial_publication_latches_before_exception_formatting(seams, format_error):
    s = seams
    class BadError(RuntimeError):
        def __str__(self):
            assert s.m._failure is not None
            with pytest.raises(s.m.SparseIndexerCompatibilityError):
                s.m.require_healthy()
            raise format_error('diagnostic formatting failed')
    def broken(name, function, **kwargs):
        s.calls.append(name)
        raise BadError()
    s.modules['vllm.utils.torch_utils'].direct_register_custom_op = broken
    with pytest.raises(s.m.SparseIndexerCompatibilityError, match='publication failed'):
        s.m.install()
    for action in (s.m.require_healthy, s.m.install):
        with pytest.raises(s.m.SparseIndexerCompatibilityError):
            action()
    assert len(s.calls) == 1 and not s.records


def test_interrupted_framework_import_cannot_be_retried(seams, monkeypatch):
    def interrupted(name):
        raise SystemExit('partial import')
    monkeypatch.setattr(seams.m, 'import_module', interrupted)
    with pytest.raises(seams.m.SparseIndexerCompatibilityError, match='activation failed'):
        seams.m.install()
    with pytest.raises(seams.m.SparseIndexerCompatibilityError):
        seams.m.install()
    assert not seams.calls


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='fork required')
def test_fork_requires_child_activation_without_duplicate_publication(seams):
    s = seams
    s.m.install()
    pid = os.fork()
    if pid == 0:
        signal.alarm(10)
        try:
            with pytest.raises(s.m.SparseIndexerCompatibilityError, match='another PID'):
                s.m.require_healthy()
            s.m.install()
            s.m.require_healthy()
            assert len(s.calls) == 1
        except BaseException:
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    s.m.require_healthy()


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='fork required')
def test_failure_is_inherited_during_exception_formatting(seams):
    s = seams
    children = []
    class ForkError(RuntimeError):
        def __str__(self):
            assert s.m._failure is not None
            pid = os.fork()
            if pid == 0:
                signal.alarm(10)
                try:
                    for action in (s.m.require_healthy, s.m.install):
                        with pytest.raises(s.m.SparseIndexerCompatibilityError):
                            action()
                except BaseException:
                    os._exit(1)
                os._exit(0)
            _, status = os.waitpid(pid, 0)
            children.append(os.waitstatus_to_exitcode(status))
            return 'child rejected inherited failed publication'
    def broken(name, function, **kwargs):
        s.calls.append(name)
        raise ForkError()
    s.modules['vllm.utils.torch_utils'].direct_register_custom_op = broken
    with pytest.raises(s.m.SparseIndexerCompatibilityError):
        s.m.install()
    assert children == [0] and len(s.calls) == 1
    with pytest.raises(s.m.SparseIndexerCompatibilityError):
        s.m.require_healthy()
