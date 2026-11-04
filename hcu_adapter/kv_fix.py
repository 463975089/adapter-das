# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""KV-pool stride tensor fix — text-patch (governance §3 tier-3, allowlisted).

`memory_pool.py::_init_data_ptrs_and_strides` builds

    torch.tensor([np.prod(x.shape[1:]) * x.dtype.itemsize ...], dtype默认,
                 device=cuda)

`np.prod` returns numpy.int64; torch 2.11 (dlpack-constructor route) cannot
infer a dtype from numpy scalars -> RuntimeError. Wrapping with int() restores
the upstream-on-NVIDIA behavior where python ints infer an integer dtype.
Same class of wound as dist_bc (np payload) but in a private module with its
own `import numpy`, hence a text patch rather than a namespace rebind.
"""

import logging

LOG = logging.getLogger("hcu.adapter")

_NEEDLE = "np.prod(x.shape[1:]) * x.dtype.itemsize"
_REPLACE = "int(np.prod(x.shape[1:])) * x.dtype.itemsize"
# v0.5.22 rewrote _init_data_ptrs_and_strides: strides now come from
# x.stride(0) * x.dtype.itemsize (plain python ints, no numpy scalars in
# the torch.tensor constructors) -- the wound this patch treated is gone
# upstream. Detect the healed shape and retire the patch instead of
# raising, so one adapter tree serves both upstream lines.
_HEALED_MARK = "slot_move_pointer_buffers"


def install():
    import importlib

    import hcu_adapter.registry as _reg

    mod = importlib.import_module(
        "sglang.srt.mem_cache.memory_pool"
    )
    src_path = mod.__file__
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    if _REPLACE in src:
        return  # already patched (double install in spawn'd children)
    if _NEEDLE not in src:
        if _HEALED_MARK in src:
            # No patch happens, so no registry binding: the op is served by
            # upstream itself on this tree ("登记即存在" applies to what WE
            # install). The text-patch allowlist entry stays valid for
            # <=v0.5.21 trees.
            LOG.info(
                "HCU adapter: kv_fix no-op -- upstream rewrote "
                "_init_data_ptrs_and_strides (np.prod wound gone, v0.5.22)"
            )
            return
        raise RuntimeError(
            "kv_fix: needle not found in memory_pool.py -- upstream changed; "
            "re-derive the patch point (governance §3)"
        )
    patched = src.replace(_NEEDLE, _REPLACE, 1)

    import hashlib
    import types

    code = compile(patched, src_path, "exec")
    fresh = types.ModuleType(mod.__name__)
    fresh.__file__ = src_path
    fresh.__dict__.update(mod.__dict__)  # keep identity of shared objects
    exec(code, fresh.__dict__)
    import sys

    sys.modules[mod.__name__] = fresh
    _reg.register(
        "kv_pool_strides",
        _REPLACE,
        target=f"text:{src_path}::_init_data_ptrs_and_strides",
        mechanism=_reg.MECH_TEXT,
        detail=_NEEDLE,
    )
    LOG.warning("HCU adapter: kv-pool stride text patch installed")
