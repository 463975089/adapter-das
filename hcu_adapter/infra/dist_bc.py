# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Writable TP broadcast buffers without changing community source files.

Torch's numpy/DLPack path cannot consume numpy's read-only view of pickle
bytes. Give only broadcast_pyobj a private numpy facade which copies such
views. Keep the community function body and the default wire format.
An opt-in pickle facade compresses large payloads between upgraded ranks.

The function object itself is retained: common's package re-export and any
earlier ``from ... import broadcast_pyobj`` aliases see the repair as well.
Each spawned interpreter installs its own bridge during normal bootstrap.
"""

import inspect
import logging
import threading
import types
import zlib

LOG = logging.getLogger("hcu.adapter")
_LOCK = threading.RLock()
_DISPATCH = "_hcu_dist_broadcast_impl"
_MARKER = "_hcu_dist_broadcast_original"
_COMPRESSED = b"\x00HCUZ1"
_MIN_COMPRESS_BYTES = 1024 * 1024


class _WritableBuffers:
    def __init__(self, numpy):
        self._numpy = numpy

    def __getattr__(self, name):
        return getattr(self._numpy, name)

    def frombuffer(self, *args, **kwargs):
        array = self._numpy.frombuffer(*args, **kwargs)
        return array if array.flags.writeable else array.copy()



class _BroadcastPickle:
    """Private wire encoding for the broadcast function's bytes payload."""

    def __init__(self, pickle, enabled):
        self._pickle = pickle
        self._enabled = enabled

    def dumps(self, *args, **kwargs):
        payload = self._pickle.dumps(*args, **kwargs)
        if not self._enabled or len(payload) < _MIN_COMPRESS_BYTES:
            return payload
        compressed = zlib.compress(payload, level=1)
        if len(compressed) + len(_COMPRESSED) >= len(payload):
            return payload
        return _COMPRESSED + compressed

    def loads(self, payload, *args, **kwargs):
        if payload.startswith(_COMPRESSED):
            payload = zlib.decompress(payload[len(_COMPRESSED):])
        return self._pickle.loads(payload, *args, **kwargs)


def _broadcast(*args, **kwargs):
    # Installed on the original function, whose globals hold this dispatch
    # binding. No process-wide numpy/Torch patching or source recompilation.
    return _hcu_dist_broadcast_impl(*args, **kwargs)


def _install_function(function, *, compress=False):
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
        if "pickle" in names and namespace.get("pickle") is not None:
            namespace["pickle"] = _BroadcastPickle(namespace["pickle"], compress)
        elif compress:
            raise RuntimeError("dist_bc: broadcast pickle contract changed upstream")
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
    from hcu_adapter.infra import hcu_env, registry
    import sglang.srt.utils.common as common

    if hasattr(common, "broadcast_pyobj"):
        owner = common
    else:
        from sglang.srt.utils import process_group as owner

    value, source = hcu_env.resolve("HCU_DIST_BROADCAST_COMPRESS")
    compress = value == "1"
    _install_function(owner.broadcast_pyobj, compress=compress)
    registry.register(
        "dist_broadcast",
        "writable-numpy-broadcast-buffer",
        target=f"{owner.__name__}.broadcast_pyobj",
        mechanism=registry.MECH_REBIND,
        detail="private numpy facade; existing function aliases retained; no source writes",
    )
    LOG.warning("HCU adapter: broadcast_pyobj writable buffer runtime bridge installed")
    if compress:
        LOG.info("HCU adapter: lossless large broadcast payload compression enabled (%s)", source)
