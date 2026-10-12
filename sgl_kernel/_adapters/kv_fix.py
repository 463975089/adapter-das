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


def install():
    import importlib

    import sgl_kernel._adapters.registry as _reg

    mod = importlib.import_module(
        "sglang.srt.mem_cache.memory_pool"
    )
    src_path = mod.__file__
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    if _REPLACE in src:
        return  # already patched (double install in spawn'd children)
    if _NEEDLE not in src:
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
