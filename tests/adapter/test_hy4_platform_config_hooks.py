# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Compose platform CLI admission with HY4 cache canonicalization; CPU doubles."""
import sys
from types import ModuleType,SimpleNamespace as NS

import pytest

from hcu_adapter.bridges import vllm as bridge
from hcu_adapter.bridges import vllm_cli as cli
from hcu_adapter.bridges import vllm_sparse_flashmla as sparse


@pytest.fixture
def platform(monkeypatch):
    seen=[]
    class RocmPlatform:
        @classmethod
        def check_and_update_config(cls,cfg):
            seen.append('core')
            assert cfg.cache_config.cache_dtype=='fp8_ds_mla'
    for name in ['vllm','vllm.platforms','vllm.platforms.rocm']:
        module=ModuleType(name);module.__path__=[]
        if name.endswith('rocm'):module.RocmPlatform=RocmPlatform
        monkeypatch.setitem(sys.modules,name,module)
    for name,attrs in {
        'hcu_adapter.bridges.vllm_moe_aiter':{'configure_padding_policy':lambda value:seen.append('padding')},
        'hcu_adapter.bridges.vllm_model_load':{'OFFICIAL_MODEL_LOAD_WORKER':'official.Worker','POST_MODEL_LOAD_WORKER':'post.Worker'},
    }.items():
        module=ModuleType(name);module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setitem(bridge.__dict__,'HcuPlatform',None)
    monkeypatch.setattr(bridge,'require_runtime_healthy',lambda:None)
    monkeypatch.setattr(bridge,'_configure_indexer_kv_dtype',lambda cfg:None)
    monkeypatch.setattr(cli,'_switches',{})
    def validate(cfg):
        seen.append('cli')
        if cfg.additional_config.get('invalid'):raise ValueError('CLI configuration rejected')
    cli.register_config_switch('--test-admission','test_admission',help='test',validate=validate)
    monkeypatch.setattr(sparse,'require_available',lambda:None)
    backend=object()
    monkeypatch.setattr(sparse,'_backend_registry',lambda:NS(AttentionBackendEnum=NS(
        FLASHMLA_SPARSE=NS(get_class=lambda:backend))))
    canonical=ModuleType('vllm.model_executor.layers.attention.mla_attention')
    def canonicalize(owner,dtype):
        assert owner is backend and dtype=='fp8_e4m3'
        seen.append('canonical');return 'fp8_ds_mla'
    canonical._canonicalize_sparse_mla_kv_cache_dtype=canonicalize
    monkeypatch.setitem(sys.modules,canonical.__name__,canonical)
    cfg=NS(attention_config=NS(backend='FLASHMLA_SPARSE'),cache_config=NS(cache_dtype='fp8_e4m3'),
        kernel_config=NS(moe_backend='deep_gemm'),additional_config={},
        parallel_config=NS(worker_cls='custom.Worker',distributed_executor_backend='mp'))
    return bridge._hcu_platform_class(),cfg,seen


def test_cli_admission_and_cache_alias_both_precede_core(platform):
    cls,cfg,seen=platform
    cls.check_and_update_config(cfg)
    assert seen==['cli','canonical','padding','core']


def test_rejected_cli_never_changes_cache_or_reaches_core(platform):
    cls,cfg,seen=platform;cfg.additional_config={'invalid':True}
    with pytest.raises(ValueError,match='CLI configuration rejected'):
        cls.check_and_update_config(cfg)
    assert seen==['cli'] and cfg.cache_config.cache_dtype=='fp8_e4m3'


def test_both_upstream_and_hy4_kill_switches_remain_registered(monkeypatch):
    from hcu_adapter.infra import kill_switch
    names=['hy4_prefill_workspace','vllm_paged_mqa_logits']
    monkeypatch.setenv('HCU_DISABLE_OPS',','.join(names))
    assert all(name in kill_switch.SWITCHES and kill_switch.op_disabled(name) for name in names)
