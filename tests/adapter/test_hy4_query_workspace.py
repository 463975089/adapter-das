# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Portable constructor-boundary tests; no GPU performance qualification."""
from types import SimpleNamespace as NS

import pytest

from hcu_adapter.kernels import vllm_sparse_flashmla_dtk as kernel


def config():
    return NS(scheduler_config=NS(max_num_batched_tokens=98304, max_num_seqs=256),
              model_config=NS(max_model_len=88000), marker=object())


def constructor_fixture(cfg, kv_calls):
    namespace = {
        'get_current_vllm_config': lambda: cfg,
        'get_prefill_workspace_size': lambda c: kv_calls.append(c) or c.scheduler_config.max_num_batched_tokens,
    }
    exec('def init(self):\n'
         ' cfg = get_current_vllm_config()\n'
         ' self.query_rows = cfg.scheduler_config.max_num_batched_tokens\n'
         ' self.kv_rows = get_prefill_workspace_size(cfg)\n'
         ' self.marker = cfg.marker\n', namespace)
    return namespace['init']


def test_query_rows_are_local_while_kv_and_scheduler_remain_global():
    cfg, calls, result = config(), [], NS()
    native = constructor_fixture(cfg, calls)
    owned = kernel.bind_query_workspace_constructor(native, lambda c: 7680)
    owned(result)
    assert result.query_rows == 7680
    assert result.kv_rows == 98304
    assert calls == [cfg]
    assert cfg.scheduler_config.max_num_batched_tokens == 98304
    assert result.marker is cfg.marker


def test_native_function_and_cached_alias_are_unchanged():
    cfg, calls = config(), []
    native = constructor_fixture(cfg, calls)
    cached_alias = native
    owned = kernel.bind_query_workspace_constructor(native, lambda c: 7680)
    assert owned.__code__ is native.__code__
    assert owned.__globals__ is not native.__globals__
    original = NS()
    cached_alias(original)
    assert original.query_rows == original.kv_rows == 98304
    assert native.__globals__['get_current_vllm_config']() is cfg


def test_unaffected_configuration_keeps_its_original_identity():
    cfg, calls, result = config(), [], NS()
    native = constructor_fixture(cfg, calls)
    owned = kernel.bind_query_workspace_constructor(native, lambda c: None)
    owned(result)
    assert result.query_rows == result.kv_rows == 98304
    assert calls == [cfg]


@pytest.mark.parametrize('rows', [0, -1, 98305, 1.5, True])
def test_invalid_or_underqualified_capacity_is_rejected(rows):
    cfg, calls = config(), []
    owned = kernel.bind_query_workspace_constructor(
        constructor_fixture(cfg, calls), lambda c: rows)
    with pytest.raises((ValueError, TypeError), match='query|capacity|rows'):
        owned(NS())
    assert not calls


def test_constructor_drift_is_rejected_before_execution():
    with pytest.raises(RuntimeError, match='anchor'):
        kernel.bind_query_workspace_constructor(lambda self: None, lambda c: 7680)
