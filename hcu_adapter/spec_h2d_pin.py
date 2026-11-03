"""Pinned speculative-decode metadata without changing community source.

Only eagle_prepare_for_decode receives a private Torch facade. Its original
body, CPU tensor values, allocation checks and non_blocking transfers remain
in use. Pinned allocations belong to Torch's allocator, which tracks pending
H2D copies; no fixed-depth ring may overwrite a still-in-flight source.
"""

from __future__ import annotations

import dis
import inspect
import logging
import threading
import types

from . import registry as _reg

logger = logging.getLogger("hcu.adapter")
_LOCK = threading.RLock()
_DISPATCH = "_hcu_spec_prepare_impl"
_MARKER = "_hcu_spec_prepare_original"


class _PinnedMetadata:
    def __init__(self, torch):
        self._torch = torch
        self._unavailable = False

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def tensor(self, *args, **kwargs):
        # Guard the runtime ABI too: never pin unrelated/device/float tensors.
        if (kwargs.get("device") != "cpu"
                or kwargs.get("dtype") != self._torch.int32
                or "pin_memory" in kwargs):
            raise RuntimeError("spec_h2d_pin: KV-length tensor contract changed")
        if not self._unavailable:
            try:
                return self._torch.tensor(*args, **kwargs, pin_memory=True)
            except RuntimeError:
                # Verify the stock allocation itself succeeds before treating
                # the failure as an unavailable pinned allocator.
                value = self._torch.tensor(*args, **kwargs)
                self._unavailable = True
                logger.warning("HCU spec_h2d_pin: pinned allocation unavailable; "
                               "using stock pageable KV-length tensors")
                return value
        return self._torch.tensor(*args, **kwargs)


def _prepare(*args, **kwargs):
    return _hcu_spec_prepare_impl(*args, **kwargs)


def _install_function(function):
    with _LOCK:
        if not isinstance(function, types.FunctionType) or function.__closure__:
            raise RuntimeError("spec_h2d_pin: expected closure-free prepare function")
        if hasattr(function, _MARKER):
            if (function.__code__ is not _prepare.__code__
                    or function.__globals__.get(_DISPATCH) is not getattr(function, _MARKER)):
                raise RuntimeError("spec_h2d_pin: binding changed after installation")
            return function
        # Every Torch access must be one of the original two metadata tensor
        # constructions. Reject new Torch consumers instead of changing them.
        instructions = list(dis.get_instructions(function))
        accesses = []
        for i, item in enumerate(instructions):
            if item.opname == "LOAD_GLOBAL" and item.argval == "torch":
                following = instructions[i + 1]
                if following.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                    raise RuntimeError("spec_h2d_pin: upstream Torch access changed")
                accesses.append(following.argval)
        if sorted(accesses) != ["int32", "int32", "tensor", "tensor"]:
            raise RuntimeError("spec_h2d_pin: upstream metadata construction changed")
        if not {"cur_kv_lens_cpu", "nxt_kv_lens_cpu"}.issubset(function.__code__.co_varnames):
            raise RuntimeError("spec_h2d_pin: upstream KV-length locals changed")
        torch = function.__globals__.get("torch")
        if torch is None or _DISPATCH in function.__globals__:
            raise RuntimeError("spec_h2d_pin: missing Torch or foreign dispatch binding")
        signature = inspect.signature(function)
        namespace = dict(function.__globals__, torch=_PinnedMetadata(torch))
        original = types.FunctionType(function.__code__, namespace, function.__name__,
                                      function.__defaults__, function.__closure__)
        original.__kwdefaults__ = function.__kwdefaults__
        function.__globals__[_DISPATCH] = original
        function.__signature__ = signature
        setattr(function, _MARKER, original)
        function.__code__ = _prepare.__code__
        return function


def install():
    from . import hcu_env

    if not hcu_env.enabled("HCU_SPEC_H2D_PIN"):
        return
    import torch

    if torch.version.hip is None:
        return
    import sglang.srt.speculative.eagle_utils as eagle

    _install_function(eagle.eagle_prepare_for_decode)
    _reg.register(
        "platform_detect", "spec-h2d-pinned-metadata",
        target="sglang.srt.speculative.eagle_utils.eagle_prepare_for_decode",
        mechanism=_reg.MECH_REBIND,
        detail="private Torch metadata allocator; aliases retained; no source writes or ring reuse",
    )
    logger.warning("HCU adapter: speculative KV-length pinned allocation installed at runtime")


def pinned_h2d(cpu_t, device, key="d"):
    """Compatibility for previously imported helpers; never reuse a live slot."""
    flat = cpu_t.reshape(-1)
    try:
        pinned = flat.pin_memory()
    except RuntimeError:
        return flat.to(device=device, non_blocking=True)
    return pinned.to(device=device, non_blocking=True)
