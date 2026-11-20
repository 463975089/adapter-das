# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Profile bound checked against actual Core PCP partitioning."""
from types import SimpleNamespace as NS
import numpy as np
import pytest

from hcu_adapter.features.hy4 import pcp_profile as owned


def config(pcp=16, batch=98304, seqs=256):
    return NS(model_config=NS(hf_config=NS(model_type='hy_v4'), enforce_eager=True),
        parallel_config=NS(prefill_context_parallel_size=pcp,
            decode_context_parallel_size=1, tensor_parallel_size=1),
        scheduler_config=NS(max_num_batched_tokens=batch, max_num_seqs=seqs),
        speculative_config=NS(num_speculative_tokens=3, parallel_drafting=False),
        use_v2_model_runner=True, attention_config=NS(hisparse_config=None))


@pytest.mark.parametrize('pcp', [2, 4, 16])
@pytest.mark.parametrize('shard_decode', [False, True])
def test_bound_covers_real_core_mixed_request_partition(pcp, shard_decode):
    from vllm.v1.worker.gpu.pcp_manager import PCPManager
    cfg = config(pcp)
    bound = owned.local_profile_tokens(cfg)
    manager = object.__new__(PCPManager)
    manager.pcp_world_size = pcp
    manager.dcp_world_size = 1
    manager.shard_decode_requests = shard_decode
    rng = np.random.default_rng(52)
    for _ in range(64):
        size = int(rng.integers(1, 257))
        prefill = rng.random(size) > .5
        lengths = rng.integers(1, 5, size=size)
        room = cfg.scheduler_config.max_num_batched_tokens - int(lengths.sum())
        for i in np.flatnonzero(prefill):
            extra = int(rng.integers(0, room + 1))
            lengths[i] += extra
            room -= extra
        actual = max(sum(n for _, _, n in manager._iter_rank_chunks(rank, lengths, prefill))
                     for rank in range(pcp))
        assert actual <= bound


def test_profile_only_and_native_runner_algorithms_are_retained(monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as Native
    calls = []
    monkeypatch.setattr(Native, '_dummy_run',
        lambda self, tokens, *args, **kwargs: calls.append((tokens, kwargs)) or 'native')
    cls = owned.runner_class()
    assert cls.profile_run is Native.profile_run
    assert cls.execute_model is Native.execute_model
    runner = object.__new__(cls)
    runner.vllm_config = config()
    assert runner._dummy_run(98304, is_profile=True) == 'native'
    assert calls[-1][0] == 7680
    runner._dummy_run(98304, is_profile=False)
    assert calls[-1][0] == 98304
    runner._dummy_run(512, is_profile=True)
    assert calls[-1][0] == 512
    assert runner.vllm_config.scheduler_config.max_num_batched_tokens == 98304


def test_factory_scope_and_small_budget():
    cfg = config(batch=32, seqs=8)
    assert owned.local_profile_tokens(cfg) == 32
    assert owned.matches(cfg)
    cfg.parallel_config.prefill_context_parallel_size = 1
    assert not owned.matches(cfg)
    cfg.parallel_config.prefill_context_parallel_size = 16
    cfg.model_config.hf_config.model_type = 'deepseek_v4'
    assert not owned.matches(cfg)


def test_factory_health_is_latched_after_owner_restore(monkeypatch):
    entry = NS(create=owned.create, matches=owned.matches)
    monkeypatch.setattr(owned, '_registration', entry)
    monkeypatch.setattr(owned, '_failure', None)
    monkeypatch.setattr(owned, 'op_disabled', lambda name: False)
    monkeypatch.setattr(owned.factory, 'require_model_runner_factory', lambda entry: None)
    record = dict(implementation=owned.__name__ + '.create',
        target=owned.__name__ + '.create', mechanism=owned.registry.MECH_SEAM)
    monkeypatch.setattr(owned.registry, 'get', lambda op: record)
    owned.require_healthy()
    record['implementation'] = 'foreign'
    with pytest.raises(RuntimeError, match='owner'):
        owned.require_healthy()
    record['implementation'] = owned.__name__ + '.create'
    with pytest.raises(RuntimeError, match='owner'):
        owned.require_healthy()
