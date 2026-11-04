# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Writable TP broadcast buffers without changing community source files.

Torch's numpy/DLPack path cannot consume numpy's read-only view of pickle
bytes. Give only broadcast_pyobj a private numpy facade which copies such
views. Keep the community function body and distributed protocol unchanged.

The function object itself is retained: common's package re-export and any
earlier ``from ... import broadcast_pyobj`` aliases see the repair as well.
Each spawned interpreter installs its own bridge during normal bootstrap.
"""

import inspect
import logging
import threading
import types

LOG = logging.getLogger("hcu.adapter")
_LOCK = threading.RLock()
_DISPATCH = "_hcu_dist_broadcast_impl"
_MARKER = "_hcu_dist_broadcast_original"


class _WritableBuffers:
    def __init__(self, numpy):
        self._numpy = numpy

    def __getattr__(self, name):
        return getattr(self._numpy, name)

    def frombuffer(self, *args, **kwargs):
        array = self._numpy.frombuffer(*args, **kwargs)
        return array if array.flags.writeable else array.copy()


def _broadcast(*args, **kwargs):
    # Installed on the original function, whose globals hold this dispatch
    # binding. No process-wide numpy/Torch patching or source recompilation.
    return _hcu_dist_broadcast_impl(*args, **kwargs)


def _install_function(function):
    """Retain cached aliases, defaults and introspection of a Python entry."""
    with _LOCK:
        if not isinstance(function, types.FunctionType) or function.__closure__:
            raise RuntimeError("dist_bc: expected a closure-free Python broadcast_pyobj")
        if hasattr(function, _MARKER):
            if (function.__code__ is not _broadcast.__code__
                    or function.__globals__.get(_DISPATCH) is not getattr(function, _MARKER)):
                raise RuntimeError("dist_bc: broadcast binding changed after installation")
            return function
        names = function.__code__.co_names
        numpy = function.__globals__.get("np")
        if "np" not in names or "frombuffer" not in names or numpy is None:
            raise RuntimeError("dist_bc: broadcast numpy buffer contract changed upstream")
        if _DISPATCH in function.__globals__:
            raise RuntimeError("dist_bc: private broadcast dispatch binding already exists")
        signature = inspect.signature(function)
        namespace = dict(function.__globals__, np=_WritableBuffers(numpy))
        original = types.FunctionType(
            function.__code__, namespace, function.__name__,
            function.__defaults__, function.__closure__,
        )
        original.__kwdefaults__ = function.__kwdefaults__
        function.__globals__[_DISPATCH] = original
        function.__signature__ = signature
        setattr(function, _MARKER, original)
        function.__code__ = _broadcast.__code__
        return function


def install():
    from hcu_adapter import registry
    import sglang.srt.utils.common as common

    _install_function(common.broadcast_pyobj)
    registry.register(
        "dist_broadcast",
        "writable-numpy-broadcast-buffer",
        target="sglang.srt.utils.common.broadcast_pyobj",
        mechanism=registry.MECH_REBIND,
        detail="private numpy facade; existing function aliases retained; no source writes",
    )
    LOG.warning("HCU adapter: broadcast_pyobj writable buffer runtime bridge installed")
