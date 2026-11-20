# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Bound MRV2's local dummy profile without changing the global scheduler budget."""
from hcu_adapter.bridges import vllm_model_runner as factory
from hcu_adapter.infra import registry
from hcu_adapter.infra.kill_switch import op_disabled

OP = 'vllm_hy4_pcp_profile'
_registration = None
_failure = None
_runner_cls = None


def matches(cfg):
    p = cfg.parallel_config
    return (getattr(cfg.model_config.hf_config, 'model_type', None) == 'hy_v4'
        and cfg.use_v2_model_runner and cfg.model_config.enforce_eager
        and p.prefill_context_parallel_size > 1
        and p.decode_context_parallel_size == 1 and p.tensor_parallel_size == 1)


def local_profile_tokens(cfg):
    from vllm.v1.attention.backends.mla.index_group import get_sparse_mla_index_group_max_rows
    scheduler = cfg.scheduler_config
    width = cfg.parallel_config.prefill_context_parallel_size
    # Each prefill contributes at most two rounded zig-zag chunks per rank.
    # Include all possible decode rows even if PCP currently shards requests.
    prefill = (scheduler.max_num_batched_tokens + width - 1) // width
    return min(scheduler.max_num_batched_tokens,
        prefill + 2 * scheduler.max_num_seqs + get_sparse_mla_index_group_max_rows(cfg))


def runner_class():
    global _runner_cls
    if _runner_cls is None:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        class HcuHY4PCPModelRunner(GPUModelRunner):
            def _dummy_run(self, num_tokens, *args, **kwargs):
                if kwargs.get('is_profile', False):
                    num_tokens = min(num_tokens, local_profile_tokens(self.vllm_config))
                return super()._dummy_run(num_tokens, *args, **kwargs)
        _runner_cls = HcuHY4PCPModelRunner
    return _runner_cls


def create(cfg, device):
    require_healthy()
    if not matches(cfg):
        raise ValueError('HY4 local profile factory requires eager MRV2/TP1/PCP>1/DCP1')
    return runner_class()(cfg, device)


def require_healthy():
    global _failure
    if _failure is not None:
        raise RuntimeError(_failure)
    if _registration is not None:
        try:
            if op_disabled(OP):
                raise RuntimeError('HY4 PCP profile was disabled after publication')
            factory.require_model_runner_factory(_registration)
            record = registry.get(OP)
            if (_registration.create is not create or _registration.matches is not matches
                    or record is None or record['implementation'] != __name__ + '.create'
                    or record['target'] != __name__ + '.create'
                    or record['mechanism'] != registry.MECH_SEAM):
                raise RuntimeError('HY4 PCP profile registry changed owner')
        except BaseException as error:
            _failure = 'HY4 PCP profile publication changed owner or policy; restart the process'
            raise RuntimeError(_failure) from error


def install():
    global _registration, _failure
    require_healthy()
    if _registration is not None:
        return
    if op_disabled(OP):
        return
    try:
        entry = factory.register_model_runner_factory(OP, matches=matches, create=create)
        registry.register(op=OP, impl=__name__ + '.create', target=__name__ + '.create',
                          mechanism=registry.MECH_SEAM)
        _registration = entry
    except BaseException as error:
        _failure = 'HY4 PCP profile publication failed; restart the process'
        raise RuntimeError(_failure) from error
