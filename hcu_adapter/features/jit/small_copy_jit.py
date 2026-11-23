# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

_MISSING = object()
_MARKER = "_hcu_small_copy_standard_jit"


def _jit_options(kernel):
    options = getattr(kernel, "_init_kwargs", _MISSING)
    if options is not _MISSING:
        if not isinstance(options, dict):
            raise TypeError("cannot preserve Triton JIT options: invalid _init_kwargs")
        return dict(options)
    # DTK builds without _init_kwargs retain these constructor options on
    # the actual upstream JIT object. Do not silently substitute defaults.
    attributes = {
        "version": "version",
        "do_not_specialize": "do_not_specialize",
        "do_not_specialize_on_alignment": "do_not_specialize_on_alignment",
        "debug": "debug",
        "noinline": "noinline",
        "repr": "_repr",
        "launch_metadata": "launch_metadata",
    }
    try:
        return {option: getattr(kernel, name) for option, name in attributes.items()}
    except AttributeError as error:
        raise TypeError("cannot preserve Triton JIT options: incomplete vendor ABI") from error


def adapt_kernel(kernel, jit_type, *, register=None):
    if getattr(kernel, _MARKER, None) is not None:
        return False
    if (type(kernel).__module__, type(kernel).__name__) != (
        "triton.backends.hcu.jit", "FastJITFunction"
    ):
        return False
    namespace = vars(kernel)
    if type(namespace) is not dict:
        raise TypeError("small_copy requires a writable kernel namespace")
    previous = {name: namespace.get(name, _MISSING) for name in ("run", _MARKER)}
    standard = jit_type(kernel.fn, **_jit_options(kernel))
    replacements = {"run": standard.run, _MARKER: standard}

    def check_owner():
        if any(namespace.get(name, _MISSING) is not value
               for name, value in previous.items()):
            raise RuntimeError("small_copy kernel changed before publication")

    check_owner()
    if register is not None:
        register()
    check_owner()
    try:
        # Plain namespace writes preserve aliases and avoid host setters.
        namespace.update(replacements)
    except BaseException:
        for name, value in replacements.items():
            if namespace.get(name, _MISSING) is value:
                if previous[name] is _MISSING:
                    namespace.pop(name, None)
                else:
                    namespace[name] = previous[name]
        raise
    return True


def install():
    import torch

    if torch.version.hip is None:
        return
    from sglang.kernels.ops.memory import small_copy
    from triton.runtime.jit import JITFunction
    from hcu_adapter.infra import registry

    kernel = small_copy._small_copy_kernel

    def register_binding():
        registry.register(
            "small_copy", "standard-jit",
            target="sglang.kernels.ops.memory.small_copy._small_copy_kernel.run",
            mechanism=registry.MECH_REBIND,
            detail="retain upstream kernel and aliases; use standard Triton tuple argument specialization",
        )

    with registry.registration_transaction():
        if not adapt_kernel(kernel, JITFunction, register=register_binding):
            return
    logging.getLogger("hcu.adapter").warning(
        "HCU adapter: small_copy uses standard Triton JIT for dynamic tuple arguments"
    )
