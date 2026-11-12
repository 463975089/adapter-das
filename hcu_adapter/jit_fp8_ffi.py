# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Zero-copy FP8 output storage at the narrow tvm-ffi kernel boundary.

Community kernels use uint8 storage for OCP e4m3 output. DTK DLPack exposes
Torch float8_e4m3fn as its nominal dtype, so only that output is viewed as
uint8 when entering FFI. Public Python/custom-op interfaces, dtype selection,
scales, input tensors and cache_once factory aliases retain their semantics.
"""

import functools
import importlib
import inspect
import logging
import threading
import types

import torch

LOG = logging.getLogger("hcu.adapter")
_LOCK = threading.RLock()
_MARKER = "_hcu_fp8_output_cache"
_TARGETS = (
    ("sglang.kernels.ops.quantization.per_tensor_quant_fp8", "per_tensor_quant_fp8_module",
     "per_tensor_quant_fp8"),
    ("sglang.kernels.ops.quantization.per_token_quant_fp8", "_jit_per_token_quant_fp8_module",
     "per_token_quant_fp8"),
    ("sglang.kernels.ops.quantization.per_token_group_quant", "_jit_module",
     "per_token_group_quant"),
    ("sglang.kernels.ops.quantization.per_token_group_quant_8bit_v2", "_jit_module",
     "per_token_group_quant_8bit_v2"),
    ("sglang.kernels.ops.moe.dsv4", "_jit_silu_mul_quant_varlen_module", "run"),
    ("sglang.kernels.ops.moe.dsv4", "_jit_silu_mul_quant_contig_module", "run"),
)


class _OutputStorage:
    def __init__(self, module, entry):
        self._module = module
        self._entry = entry
        function = getattr(module, entry)

        def call(*args, **kwargs):
            # FFI exports use positional arguments; accepting a named output
            # also keeps the view boundary explicit in protocol tests.
            if len(args) > 1:
                output = args[1]
                if torch.is_tensor(output) and output.dtype == torch.float8_e4m3fn:
                    args = (args[0], output.view(torch.uint8), *args[2:])
            else:
                for name in ("output_q", "output"):
                    output = kwargs.get(name)
                    if torch.is_tensor(output) and output.dtype == torch.float8_e4m3fn:
                        kwargs[name] = output.view(torch.uint8)
            return function(*args, **kwargs)

        setattr(self, entry, call)

    def __getattr__(self, name):
        return getattr(self._module, name)


def _storage(module, entry):
    if isinstance(module, _OutputStorage):
        if module._entry != entry:
            raise RuntimeError("jit_fp8_ffi: cached export changed")
        return module
    return _OutputStorage(module, entry)


def _install_factory(factory, entry):
    """Keep the original community cache_once wrapper and all its aliases.

    Its validated fn/result_map cells are the cache interface. Existing cached
    modules are adapted, so early cache hits cannot bypass the storage view.
    No public function or registered Torch op is replaced or re-registered.
    """
    with _LOCK:
        if not isinstance(factory, types.FunctionType):
            raise RuntimeError("jit_fp8_ffi: expected cache_once factory")
        cells = dict(zip(factory.__code__.co_freevars, factory.__closure__ or ()))
        if set(cells) != {"fn", "result_map"}:
            raise RuntimeError("jit_fp8_ffi: community cache_once layout changed")
        cached = cells["result_map"].cell_contents
        fn = cells["fn"].cell_contents
        marker = getattr(factory, _MARKER, None)
        if marker:
            if (marker[0] != entry or marker[1] is not fn
                    or marker[2] is not cached or not isinstance(cached, dict)):
                raise RuntimeError("jit_fp8_ffi: factory changed after installation")
            return factory
        if (not isinstance(cached, dict) or not isinstance(fn, types.FunctionType)
                or inspect.unwrap(factory) is not fn or fn.__closure__
                or "load_jit" not in fn.__code__.co_names):
            raise RuntimeError("jit_fp8_ffi: community factory contract changed")
        # Build every existing view before changing either cell. A missing
        # export must not leave a partially adapted cache.
        adapted = {key: _storage(module, entry) for key, module in cached.items()}

        @functools.wraps(fn)
        def load(*args, **kwargs):
            return _storage(fn(*args, **kwargs), entry)

        cells["fn"].cell_contents = load
        cells["result_map"].cell_contents = adapted
        setattr(factory, _MARKER, (entry, load, adapted))
        return factory


def install():
    from . import hcu_env, registry

    if not hcu_env.enabled("HCU_JIT_FP8_FFI") or torch.version.hip is None:
        return
    for module_name, name, entry in _TARGETS:
        module = importlib.import_module(module_name)
        _install_factory(getattr(module, name), entry)
        registry.register(
            "per_token_quant", "jit-fp8-output-storage",
            target=module_name + "." + name, mechanism=registry.MECH_REBIND,
            detail="retain factory/cache aliases; only e4m3fn output viewed as uint8 at FFI",
        )
    LOG.warning("HCU adapter: JIT FP8 output storage views installed (per-token, generic, v2, DSV4)")
