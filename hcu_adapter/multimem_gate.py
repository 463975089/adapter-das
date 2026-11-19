# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging
import threading
import types

_LOCK = threading.RLock()
_TARGET = '_hcu_multimem_original_init'
_MARKER = '_hcu_multimem_init_gate'


def _init(self, max_tokens, *, enabled=True, skip_entry_sync=False):
    return _hcu_multimem_original_init(
        self, max_tokens, enabled=False, skip_entry_sync=skip_entry_sync,
    )


def _install_init(function):
    with _LOCK:
        if not isinstance(function, types.FunctionType) or function.__closure__:
            raise RuntimeError('multimem_gate: expected closure-free constructor')
        if hasattr(function, _MARKER):
            if (function.__code__ is not _init.__code__
                    or function.__globals__.get(_TARGET) is not getattr(function, _MARKER)):
                raise RuntimeError('multimem_gate: constructor changed after installation')
            return function
        signature = inspect.signature(function)
        params = signature.parameters
        if (tuple(params) != ('self', 'max_tokens', 'enabled', 'skip_entry_sync')
                or params['enabled'].kind is not inspect.Parameter.KEYWORD_ONLY
                or params['skip_entry_sync'].kind is not inspect.Parameter.KEYWORD_ONLY
                or '_UNINIT' not in function.__code__.co_names):
            raise RuntimeError('multimem_gate: upstream constructor contract changed')
        if _TARGET in function.__globals__:
            raise RuntimeError('multimem_gate: private binding already exists')
        original = types.FunctionType(
            function.__code__, dict(function.__globals__), function.__name__,
            function.__defaults__, function.__closure__,
        )
        original.__kwdefaults__ = function.__kwdefaults__
        function.__globals__[_TARGET] = original
        function.__signature__ = signature
        setattr(function, _MARKER, original)
        function.__code__ = _init.__code__
        return function


def install():
    import torch

    if getattr(torch.version, 'hip', None) is None:
        return False
    from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
    from hcu_adapter import registry

    _install_init(triton_symm_mem_ag.MultimemAllGatherer.__init__)
    registry.register(
        'platform_detect', 'multimem-nccl-fallback',
        target='sglang.srt.distributed.device_communicators.triton_symm_mem_ag.MultimemAllGatherer.__init__',
        mechanism=registry.MECH_REBIND,
        detail='real HIP: preserve native all-gather fallback without unsupported symmetric-memory rendezvous',
    )
    logging.getLogger('hcu.adapter').warning(
        'HCU adapter: multimem all-gather probe disabled on real HIP; native all-gather fallback retained'
    )
    return True
