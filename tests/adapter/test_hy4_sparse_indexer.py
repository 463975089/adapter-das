# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real published reference base routing; device kernels tested separately."""
import sys
import os
from types import ModuleType, SimpleNamespace as NS

import torch
import pytest

from hcu_adapter.features.dsv41 import v41_fp8_indexer as reference
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer


def make_layer():
    base = reference.build()['base']
    layer = object.__new__(base)
    torch.nn.Module.__init__(layer)
    layer._hcu_owned = False
    layer._hcu_fp4 = False
    return base, layer


def test_hy4_routes_before_native_gfx938_rejection(monkeypatch):
    base, layer = make_layer()
    layer._hcu_hy4 = True
    module = ModuleType('hcu_adapter.features.dsa.hy4_sparse_indexer')
    calls = []
    module.forward = lambda *args: calls.append(args) or 'owned'
    monkeypatch.setitem(sys.modules, module.__name__, module)
    args = tuple(object() for _ in range(4))
    assert base.forward_hip(layer, *args) == 'owned'
    assert calls == [(layer, *args)]


def test_non_hy4_retains_original_rocm_path(monkeypatch):
    base, layer = make_layer()
    layer._hcu_hy4 = False
    calls = []
    monkeypatch.setattr(SparseAttnIndexer, 'forward_hip',
        lambda *args: calls.append(args) or 'native')
    args = tuple(object() for _ in range(4))
    assert base.forward_hip(layer, *args) == 'native'
    assert calls == [(layer, *args)]


def test_native_constructor_marks_hy4_without_enabling_v41(monkeypatch):
    from hcu_adapter.features.dsa import hy4_sparse_indexer as owner
    calls = []
    monkeypatch.setattr(owner, 'prepare', lambda cfg: calls.append('prepared'), raising=False)
    cfg = NS(model_config=NS(hf_config=NS(model_type='hy_v4')))
    monkeypatch.setattr(reference, '_resolvers',
        {'current_config': lambda: cfg, 'enabled': lambda: False})
    base = reference.build()['base']
    monkeypatch.setattr(SparseAttnIndexer, '__init__', lambda self: None)
    layer = object.__new__(base)
    base.__init__(layer)
    assert layer._hcu_hy4 is True
    assert layer._hcu_owned is False
    assert calls == ['prepared']


def test_prefill_splits_bounded_vendor_results_and_preserves_request_columns():
    from hcu_adapter.features.dsa.hy4_sparse_indexer import make_prefill
    q = torch.ones((5, 2, 4), dtype=torch.float8_e4m3fn)
    keys = torch.arange(32).reshape(8, 4).to(torch.float8_e4m3fn)
    scale = torch.arange(1, 9).float()
    weights = torch.ones((5, 2))
    starts = torch.tensor([0, 2, 1, 0, 4], dtype=torch.int32)
    ends = torch.tensor([2, 6, 4, 1, 8], dtype=torch.int32)
    calls = []
    def provider(query, keys, sf, w, begin, end):
        calls.append(query.shape[0])
        return torch.einsum('mhd,nd->mhn', query.float(), keys.float()).relu().mul(
            w[..., None]).sum(1).mul(sf)
    def compact(scores, output, begin, end, offset):
        cols = torch.arange(output.shape[1])[None] + begin[:, None]
        copied = scores.gather(1, cols.long().clamp(max=scores.shape[1] - 1))
        output[offset:offset + scores.shape[0]] = copied.masked_fill(cols >= end[:, None], -float('inf'))
    function = make_prefill(provider, compact, 2 * 8 * 4)
    result = function((q, None), (keys, scale), weights, starts, ends, 4)
    assert calls == [2, 2, 1]
    raw = torch.einsum('mhd,nd->mhn', q.float(), keys.float()).relu().sum(1) * scale
    cols = torch.arange(4)[None] + starts[:, None]
    expected = raw.gather(1, cols.long()).masked_fill(cols >= ends[:, None], -float('inf'))
    torch.testing.assert_close(result, expected)


def test_mx_precision_and_excess_memory_fail_before_vendor_call():
    from hcu_adapter.features.dsa.hy4_sparse_indexer import make_prefill
    function = make_prefill(lambda *args: pytest.fail('vendor called'), None, 4)
    q = torch.empty((2, 2, 4), dtype=torch.float8_e4m3fn)
    kv = torch.empty((8, 4), dtype=q.dtype), torch.ones(8)
    with pytest.raises(ValueError, match='folded Q'):
        function((q, torch.ones(2)), kv, None, None, None, 4)
    with pytest.raises(ValueError, match='temporary memory'):
        function((q, None), kv, None, None, None, 4)


def test_paged_uses_native_normal_layout_and_rejects_varlen_indices():
    from hcu_adapter.features.dsa.hy4_sparse_indexer import make_paged
    calls = []
    function = make_paged(lambda *args, **kw: calls.append((args, kw)) or 'native')
    query, weights = NS(shape=(2, 2, 32, 128)), NS(shape=(4, 32))
    assert function((query, None), 'kv', weights, 'lengths', 'table', None, 128) == 'native'
    assert calls == [((query, 'kv', weights, 'lengths', 'table', 128), {'cache_layout': 'NORMAL'})]
    with pytest.raises(ValueError, match='varlen'):
        function(('q', None), 'kv', 'w', 'lengths', 'table', None, 128, indices='remapping')


def test_disabled_route_does_not_publish_and_owner_loss_is_latched(monkeypatch):
    from hcu_adapter.features.dsa import hy4_sparse_indexer as owner
    monkeypatch.setattr(owner, '_state', None)
    monkeypatch.setattr(owner, '_failure', None)
    monkeypatch.setattr(owner, 'op_disabled', lambda name: True)
    monkeypatch.setattr(owner, '_build', lambda: pytest.fail('publication'))
    with pytest.raises(RuntimeError, match='disabled'):
        owner.forward(None, None, None, None, None)
    monkeypatch.setattr(owner, 'op_disabled', lambda name: False)
    provider = object()
    module = NS(fp8_mqa_logits=object())
    monkeypatch.setattr(owner, '_state', (os.getpid(), None, None, module, provider))
    with pytest.raises(RuntimeError, match='owner'):
        owner._check()
    module.fp8_mqa_logits = provider
    with pytest.raises(RuntimeError, match='owner'):
        owner._check()


def test_compiled_forward_does_not_trace_activation_lock(monkeypatch):
    from hcu_adapter.features.dsa import hy4_sparse_indexer as owner
    layer = NS(use_fp4_cache=False, compress_ratio=1, dcp_world_size=1,
               candidate_blocks=None)
    monkeypatch.setattr(owner, '_state', None)
    monkeypatch.setattr(owner, '_failure', None)
    monkeypatch.setattr(owner, '_check', lambda: None)
    def route(layer, hidden_states, q, k, weights):
        return hidden_states + weights
    monkeypatch.setattr(owner, '_build', lambda: (os.getpid(), route, None, None, None))
    # The published layer constructor must prepare the op before model tracing.
    owner.prepare()
    def run(x, weights):
        return owner.forward(layer, x, x, x, weights)
    compiled = torch.compile(run, backend='eager', fullgraph=True)
    for value in (1.0, 3.0):
        x = torch.full((4,), value)
        torch.testing.assert_close(compiled(x, x), x + x)


def test_opaque_execution_retains_health_checks(monkeypatch):
    from hcu_adapter.features.dsa import hy4_sparse_indexer as owner
    calls = []
    def check():
        calls.append('checked')
    monkeypatch.setattr(owner, '_check', check)
    guarded = owner._guard_execution(lambda value: value + 1)
    assert guarded(2) == 3
    assert calls == ['checked']
    def fail():
        raise RuntimeError('publication changed owner')
    monkeypatch.setattr(owner, '_check', fail)
    with pytest.raises(RuntimeError, match='owner'):
        guarded(2)


def test_hy4_constructor_preserves_core_piecewise_partition_boundary(monkeypatch):
    from hcu_adapter.features.dsa import hy4_sparse_indexer as owner
    from vllm.config import CompilationMode, CUDAGraphMode
    compilation = NS(mode=CompilationMode.VLLM_COMPILE,
        use_inductor_graph_partition=False,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        splitting_ops=['vllm::sparse_attn_indexer'])
    cfg = NS(model_config=NS(hf_config=NS(model_type='hy_v4')),
             compilation_config=compilation)
    monkeypatch.setattr(reference, '_resolvers',
        {'current_config': lambda: cfg, 'enabled': lambda: False})
    monkeypatch.setattr(owner, '_check', lambda: None)
    monkeypatch.setattr(owner, '_state', (os.getpid(), None, None, None, None))
    base = reference.build()['base']
    monkeypatch.setattr(SparseAttnIndexer, '__init__', lambda self: None)
    layer = object.__new__(base)
    base.__init__(layer)
    owner.prepare(cfg)
    assert compilation.splitting_ops == [
        'vllm::sparse_attn_indexer', 'vllm::' + owner.OP_NAME]
    assert reference.OP_NAME == 'vllm::hcu_v41_fp8_indexer_reference'


@pytest.mark.parametrize('partition,mode,expected', [
    (False, 'NONE', []), (False, 'FULL', []),
    (True, 'FULL_AND_PIECEWISE', []),
    (False, 'PIECEWISE', ['vllm::hcu_hy4_sparse_attn_indexer']),
])
def test_partition_registration_preserves_other_compile_modes(monkeypatch, partition, mode, expected):
    from hcu_adapter.features.dsa import hy4_sparse_indexer as owner
    from vllm.config import CompilationMode, CUDAGraphMode
    compilation = NS(mode=CompilationMode.VLLM_COMPILE,
        use_inductor_graph_partition=partition,
        cudagraph_mode=getattr(CUDAGraphMode, mode), splitting_ops=[])
    monkeypatch.setattr(owner, '_check', lambda: None)
    monkeypatch.setattr(owner, '_state', (os.getpid(), None, None, None, None))
    owner.prepare(NS(compilation_config=compilation))
    assert compilation.splitting_ops == expected
