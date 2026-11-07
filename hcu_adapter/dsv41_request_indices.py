"""Keep V4.1 request metadata valid when EP pads the live token count."""

import dis
import inspect
import logging
import threading
import types

from . import registry

_LOCK = threading.RLock()
_DISPATCH = "_hcu_dsv41_request_indices_impl"
_MARKER = "_hcu_dsv41_request_indices_original"


class _RequestTorch:
    def __init__(self, torch):
        self._torch = torch

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def repeat_interleave(self, requests, repeats, *, output_size=None):
        if isinstance(repeats, self._torch.Tensor) and output_size is not None:
            from .kernels.request_indices_emu import padded_request_indices

            return padded_request_indices(requests, repeats, output_size)
        return self._torch.repeat_interleave(requests, repeats, output_size=output_size)


def _request_indices(*args, **kwargs):
    return _hcu_dsv41_request_indices_impl(*args, **kwargs)


def _install_function(function):
    with _LOCK:
        if not isinstance(function, types.FunctionType) or function.__closure__:
            raise RuntimeError("dsv41_request_indices: expected closure-free function")
        if hasattr(function, _MARKER):
            if (function.__code__ is not _request_indices.__code__
                    or function.__globals__.get(_DISPATCH) is not getattr(function, _MARKER)):
                raise RuntimeError("dsv41_request_indices: binding changed after installation")
            return function
        signature = inspect.signature(function)
        parameters = list(signature.parameters.values())
        if (len(parameters) != 2 or parameters[0].name != "forward_batch"
                or parameters[0].kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
                or parameters[1].name != "num_tokens"
                or parameters[1].kind != inspect.Parameter.KEYWORD_ONLY
                or parameters[1].default is not None):
            raise RuntimeError("dsv41_request_indices: upstream signature changed")
        instructions = list(dis.get_instructions(function))
        accesses = []
        for i, item in enumerate(instructions):
            if item.opname == "LOAD_GLOBAL" and item.argval == "torch":
                following = instructions[i + 1]
                if following.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                    raise RuntimeError("dsv41_request_indices: upstream Torch access changed")
                accesses.append(following.argval)
        if (sorted(accesses) != ["int64", "int64", "repeat_interleave", "repeat_interleave"]
                or not {"is_decode", "is_target_verify", "is_extend", "extend_seq_lens",
                        "req_pool_indices", "draft_token_num"}.issubset(function.__code__.co_names)):
            raise RuntimeError("dsv41_request_indices: upstream expansion contract changed")
        torch = function.__globals__.get("torch")
        if torch is None or _DISPATCH in function.__globals__:
            raise RuntimeError("dsv41_request_indices: missing Torch or foreign dispatch")
        namespace = dict(function.__globals__, torch=_RequestTorch(torch))
        original = types.FunctionType(function.__code__, namespace, function.__name__,
                                      function.__defaults__, function.__closure__)
        original.__kwdefaults__ = function.__kwdefaults__
        function.__globals__[_DISPATCH] = original
        function.__signature__ = signature
        setattr(function, _MARKER, original)
        function.__code__ = _request_indices.__code__
        return function


def install():
    import torch

    if torch.version.hip is None:
        return
    from sglang.srt.layers.attention.dsv4 import dsv41_sparse

    _install_function(dsv41_sparse.token_req_indices)
    registry.register(
        "platform_detect", "dsv41-padded-request-indices",
        target="sglang.srt.layers.attention.dsv4.dsv41_sparse.token_req_indices",
        mechanism=registry.MECH_REBIND,
        detail="private Torch expansion maps padded tokens to reserved request zero; aliases retained; no source writes",
    )
    logging.getLogger("hcu.adapter").warning(
        "HCU adapter: V4.1 padded request indices adapted at runtime"
    )
