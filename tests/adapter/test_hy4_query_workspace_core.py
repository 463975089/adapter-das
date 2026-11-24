# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Actual Core constructor and partition bounds; allocation/transport are doubles."""
from math import prod
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_adapter.kernels import vllm_sparse_flashmla_dtk as kernel


def config():
    return NS(model_config=NS(hf_config=NS(model_type='hy_v4'), enforce_eager=True),
        parallel_config=NS(prefill_context_parallel_size=16,
            decode_context_parallel_size=1, tensor_parallel_size=1),
        scheduler_config=NS(max_num_batched_tokens=98304, max_num_seqs=256),
        speculative_config=NS(num_speculative_tokens=3, parallel_drafting=False),
        use_v2_model_runner=True, attention_config=NS(hisparse_config=None))


@pytest.mark.parametrize('owner,field,value', [
    ('parallel_config', 'prefill_context_parallel_size', 1),
    ('parallel_config', 'tensor_parallel_size', 2),
    ('parallel_config', 'decode_context_parallel_size', 2),
    ('model_config', 'enforce_eager', False),
    ('model_config.hf_config', 'model_type', 'other'),
    ('', 'use_v2_model_runner', False),
])
def test_unqualified_paths_keep_native_capacity(owner, field, value):
    cfg = config()
    target = cfg
    for name in owner.split('.') if owner else []:
        target = getattr(target, name)
    setattr(target, field, value)
    assert kernel._pcp_query_rows(cfg) is None


def test_query_bound_reuses_native_decode_rows():
    cfg = config()
    assert kernel._pcp_query_rows(cfg) == 7680
    assert cfg.scheduler_config.max_num_batched_tokens == 98304


def test_original_flashmla_constructor_keeps_global_kv_reservation(monkeypatch):
    from vllm.model_executor.layers.attention.sparse_mla_attention import SparseMLACommonImpl
    from vllm.v1.attention.backends.mla import flashmla_sparse as native
    cfg, allocations, kv_configs = config(), [], []

    def base_init(self, *args, **kwargs):
        self.kv_lora_rank = 512
        self.qk_rope_head_dim = 64
        self.dcp_world_size = 1
        self.pcp_world_size = 16

    monkeypatch.setattr(SparseMLACommonImpl, '__init__', base_init)
    monkeypatch.setattr(native, 'get_current_vllm_config', lambda: cfg)
    monkeypatch.setattr(native, 'get_prefill_workspace_size',
        lambda c: kv_configs.append(c) or 131072)
    monkeypatch.setattr(native, 'current_workspace_manager', lambda: NS(
        get_simultaneous=lambda *specs: allocations.append(specs)))
    ctor = native.FlashMLASparseImpl.__init__
    kwargs = dict(num_heads=64, head_size=576, scale=.125, num_kv_heads=1,
        alibi_slopes=None, sliding_window=None, kv_cache_dtype='fp8_ds_mla',
        logits_soft_cap=None, attn_type='decoder', kv_sharing_target_layer_name=None)
    original = object.__new__(native.FlashMLASparseImpl)
    ctor(original, **kwargs)
    bounded = object.__new__(native.FlashMLASparseImpl)
    kernel.bind_query_workspace_constructor(ctor, kernel._pcp_query_rows)(bounded, **kwargs)
    assert original.workspace_specs[1] == bounded.workspace_specs[1] == ((131072, 576), torch.bfloat16)
    assert kv_configs == [cfg, cfg]
    for i in (0, 2, 3, 4):
        assert original.workspace_specs[i][0][0] == 98304
        assert bounded.workspace_specs[i][0][0] == 7680
        assert original.workspace_specs[i][0][1:] == bounded.workspace_specs[i][0][1:]
    def size(specs):
        return sum(prod(shape) * torch.empty((), dtype=dtype).element_size()
                   for shape, dtype in specs)
    assert size(bounded.workspace_specs) < size(original.workspace_specs)
    assert allocations == [tuple(original.workspace_specs), tuple(bounded.workspace_specs)]
    assert cfg.scheduler_config.max_num_batched_tokens == 98304


def test_owned_mro_preserves_native_constructor_forward_and_sink():
    from vllm.models.hy_v4.nvidia.flashmla_sparse import HYV4FlashMLASparseImpl as Native
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl
    vendor = NS(flash_mla_sparse_fwd=lambda *a, **kw: None,
        flash_mla_with_kvcache=lambda *a, **kw: None,
        get_mla_metadata=lambda *a, **kw: None)
    impl = kernel.build_backend(vendor).get_impl_cls()
    middle = impl.__mro__[impl.__mro__.index(FlashMLASparseImpl) - 1]
    assert middle is not Native
    assert middle.__init__.__code__ is FlashMLASparseImpl.__init__.__code__
    assert middle.__init__.__globals__ is not FlashMLASparseImpl.__init__.__globals__
    assert impl.__init__ is Native.__init__
    assert impl.forward_mqa is Native.forward_mqa
    assert impl._sinks_for_query is Native._sinks_for_query
