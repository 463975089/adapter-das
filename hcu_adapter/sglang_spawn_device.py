# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Select a scheduler's GPU before adapter imports create HIP queues.

SGLang's parent-side maybe_reindex_device_id context already encloses spawn.
Carry its yielded logical device to the fresh interpreter through a one-use,
parent-scoped hint. Keep visibility, rank arguments and operator policy intact.
"""

from contextlib import contextmanager
from functools import partial
import inspect
import json
import logging
import operator
import os
import sys
import threading
import types

LOG = logging.getLogger("hcu.adapter")
_DEVICE_ENV = "_HCU_ADAPTER_SGLANG_SPAWN_DEVICE"
_DISPATCH = "_hcu_sglang_spawn_device_context"
_MARKER = "_hcu_sglang_spawn_device_original"
_LOCK = threading.RLock()


def select_inherited_device():
    """Consume the hint before common/host bridges import GPU libraries."""
    hint = os.environ.pop(_DEVICE_ENV, None)
    if hint is None or "--multiprocessing-fork" not in sys.argv:
        return
    try:
        hint = json.loads(hint)
        parent, device = hint["parent_pid"], hint["device"]
        if type(parent) is not int or type(device) is not int or device < 0:
            raise ValueError("expected a parent PID and non-negative logical device")
    except (ValueError, TypeError, KeyError) as exc:
        raise RuntimeError("HCU adapter: invalid scheduler device hint") from exc
    if parent != os.getppid():
        raise RuntimeError("HCU adapter: scheduler device hint has a different parent")

    import torch

    torch.cuda.set_device(device)
    LOG.warning("HCU adapter: selected scheduler GPU %s before bootstrap", device)


@contextmanager
def _device_context(gpu_id, *, original):
    with _LOCK, original(gpu_id) as selected:
        device = operator.index(selected)
        if isinstance(selected, bool) or device < 0:
            raise RuntimeError("HCU adapter: invalid reindexed scheduler device")
        previous = os.environ.get(_DEVICE_ENV)
        os.environ[_DEVICE_ENV] = json.dumps(
            {"parent_pid": os.getpid(), "device": device}
        )
        try:
            yield selected
        finally:
            if previous is None:
                os.environ.pop(_DEVICE_ENV, None)
            else:
                os.environ[_DEVICE_ENV] = previous


def _reindex_device_id(gpu_id):
    # The original generator retains its identity, including cached aliases
    # held by contextlib's decorated entry and upstream re-export modules.
    with _hcu_sglang_spawn_device_context(gpu_id) as selected:
        yield selected


def _install_function(entry):
    with _LOCK:
        function = getattr(entry, "__wrapped__", None)
        if (not isinstance(function, types.FunctionType)
                or function.__closure__ or not inspect.isgeneratorfunction(function)
                or list(inspect.signature(function).parameters) != ["gpu_id"]):
            raise RuntimeError("HCU adapter: maybe_reindex_device_id contract changed")
        if hasattr(function, _MARKER):
            original = getattr(function, _MARKER)
            dispatch = function.__globals__.get(_DISPATCH)
            if (function.__code__ is not _reindex_device_id.__code__
                    or not isinstance(dispatch, partial)
                    or dispatch.func is not _device_context
                    or dispatch.keywords.get("original") is not original):
                raise RuntimeError("HCU adapter: scheduler device binding changed")
            return
        if _DISPATCH in function.__globals__:
            raise RuntimeError("HCU adapter: scheduler device binding already exists")
        # Keep later upstream/provider global rebindings visible to the original.
        original = types.FunctionType(
            function.__code__, function.__globals__, function.__name__,
            function.__defaults__, function.__closure__,
        )
        original.__kwdefaults__ = function.__kwdefaults__
        original = contextmanager(original)
        function.__globals__[_DISPATCH] = partial(_device_context, original=original)
        setattr(function, _MARKER, original)
        function.__code__ = _reindex_device_id.__code__


def install():
    from hcu_adapter import registry
    from sglang.srt.utils.common import maybe_reindex_device_id

    _install_function(maybe_reindex_device_id)
    registry.register(
        "platform_detect", _device_context,
        target="sglang.srt.utils.common:maybe_reindex_device_id",
        mechanism=registry.MECH_REBIND,
        detail="parent-scoped spawn hint; logical GPU selected before bootstrap",
    )
    LOG.warning("HCU adapter: scheduler GPU selection before bootstrap installed")
