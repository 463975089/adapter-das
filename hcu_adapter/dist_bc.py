"""TP init-broadcast payload fix — on-disk text patch (governance §3 tier-3,
allowlisted entry "dist_broadcast").

`common.broadcast_pyobj` wraps `np.frombuffer(serialized_data, ...)` in
`torch.ByteTensor(...)`. torch 2.11 routes numpy inputs through DLPack;
numpy 1.25's buffers are read-only and cannot signal that flag ->
`BufferError: Cannot export readonly array...`. In-memory module rebinds
(exec'd fresh module objects) proved racy against spawn'd scheduler
children whose tp_worker binds the function before install runs, so — like
the JIT kernel-source patches — this edits the source file on disk once and
lets every subsequent import (main process + all children) compile the
patched body. The writable `.copy()` is exactly what NVIDIA's torch/numpy
combos do implicitly (their numpy build hands torch a copy).

Idempotent: re-running finds the patched marker and skips.
"""

import logging
import pathlib
import py_compile

LOG = logging.getLogger("hcu.adapter")

_NEEDLE = (
    "torch.ByteTensor(\n"
    "                np.frombuffer(serialized_data, dtype=np.uint8)\n"
    "            )"
)
_REPLACE = (
    "torch.ByteTensor(\n"
    "                np.frombuffer(serialized_data, dtype=np.uint8).copy()\n"
    "            )"
)


def install():
    import hcu_adapter.registry as _reg

    import sglang.srt.utils.common as _probe  # locate the source file

    src_path = pathlib.Path(_probe.__file__)
    src = src_path.read_text(encoding="utf-8")
    if "np.frombuffer(serialized_data, dtype=np.uint8).copy()" in src:
        pass  # already patched
    else:
        if _NEEDLE not in src:
            raise RuntimeError(
                "dist_bc: needle not found in common.py -- upstream changed; "
                "re-derive the patch point (governance §3)"
            )
        from hcu_adapter.atomic_io import atomic_write

        atomic_write(src_path, src.replace(_NEEDLE, _REPLACE, 1))
        # invalidate the cached bytecode so children recompile
        pyc = src_path.with_suffix(".pyc")
        cache_dir = src_path.parent / "__pycache__"
        if cache_dir.exists():
            for p in cache_dir.glob(src_path.stem + ".*.pyc"):
                p.unlink(missing_ok=True)
        try:
            py_compile.compile(str(src_path), doraise=True)
        except py_compile.PyCompileError:
            pass  # compile-at-import will surface real errors if any

    _reg.register(
        "dist_broadcast",
        _REPLACE,
        target=f"text:{src_path}::broadcast_pyobj",
        mechanism=_reg.MECH_TEXT,
        detail=_NEEDLE,
    )
    LOG.warning(
        "HCU adapter: broadcast_pyobj frombuffer copy on-disk patch ensured"
    )
