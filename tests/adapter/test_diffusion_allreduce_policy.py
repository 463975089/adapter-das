# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import inspect
from unittest.mock import create_autospec

import pytest

from hcu_adapter import hcu_env, kill_switch, registry, sglang_diffusion


@pytest.mark.parametrize('value,killed,expected', [('0', False, False), ('1', False, True), ('1', True, False), ('refused(old-aiter)', False, False)])
def test_native_tp_factory_honors_existing_custom_ar_policy(monkeypatch, value, killed, expected):
    from sglang.multimodal_gen.runtime.distributed import parallel_state as native
    cls = native.GroupCoordinator
    original = cls.__init__
    construct = create_autospec(original, return_value=None)
    monkeypatch.setattr(cls, '__init__', construct)
    monkeypatch.setattr(hcu_env, 'resolve', lambda name: (value, 'explicit'))
    monkeypatch.setattr(kill_switch, 'op_disabled', lambda name: killed)
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    install = getattr(sglang_diffusion, 'install_collective_policy', lambda: False)
    install()
    current = cls.__init__
    install()
    assert cls.__init__ is current
    group = native.init_parallel_group_coordinator([[0, 1]], 0, 'nccl', 'tensor')
    assert isinstance(group, cls)
    bound = inspect.signature(original).bind(*construct.call_args.args, **construct.call_args.kwargs)
    assert bound.arguments['use_srt_custom_allreduce'] is expected
    assert bound.arguments['group_ranks'] == [[0, 1]]
    assert bound.arguments['local_rank'] == 0
    assert bound.arguments['torch_distributed_backend'] == 'nccl'
    assert bound.arguments['use_device_communicator'] is False
    assert bound.arguments['group_name'] == 'tp_group'
    if not expected:
        assert registry.get('allreduce')['implementation'] == 'nccl'


def test_policy_registration_failure_does_not_publish_constructor(monkeypatch):
    from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator
    original = GroupCoordinator.__init__
    monkeypatch.setattr(hcu_env, 'resolve', lambda name: ('0', 'explicit'))
    monkeypatch.setattr(kill_switch, 'op_disabled', lambda name: False)
    def reject(*args, **kwargs):
        raise ValueError('policy registration rejected')
    monkeypatch.setattr(registry, 'register', reject)
    install = getattr(sglang_diffusion, 'install_collective_policy', lambda: False)
    with pytest.raises(ValueError, match='policy registration rejected'):
        install()
    assert GroupCoordinator.__init__ is original
