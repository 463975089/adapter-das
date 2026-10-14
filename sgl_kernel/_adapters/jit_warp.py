"""JIT sgl_kernel/warp.cuh portability patch — HIP arm reduce<T> widening.

`warp::reduce` (warp.cuh) declares `__hip_bfloat162 _v;` unconditionally in
both shuffle loops of its HIP branch, then feeds `_v` through a
`std::is_same_v<T, __hip_bfloat162>` constexpr split. The bf16x2 packed-bit
path is fine, but the generic `else` assigns the shuffle result of a *non*
-bf16 `T` (e.g. float — every fp32 warp max/sum/min reduce) into a
`__hip_bfloat162` and converts back: DTK hipcc rejects both the assignment
("no viable overloaded =") and the final `Trait::reduce(value, _v)`
("no viable conversion from '__hip_bfloat162' to 'const float'"). On HIP
this made reduce<T> un-compilable for every T except __hip_bfloat162.

Fix: type the shuffle receiver as `T` (behavior identical for bf16x2 —
that instantiation's packed-bit branch reinterprets into __hip_bfloat162
which is exactly T there — and correct-by-construction for every other
4-byte type). Same text-patch mechanics as jit_rope/jit_store_cache.
"""

import logging
import pathlib

LOG = logging.getLogger("hcu.adapter")

_RELPATH = "include/sgl_kernel/warp.cuh"

_NEEDLE = "__hip_bfloat162 _v;"
_REPL = "T _v;  // HCU adapter: receiver type follows T (bf16x2 keeps its packed-bit branch)"


def install():
    import sglang.kernels.jit as _jit_root

    p = pathlib.Path(_jit_root.__file__).parent / _RELPATH
    text = p.read_text(encoding="utf-8")
    if _REPL in text:
        LOG.warning(
            "HCU adapter: JIT warp reduce patch already installed (%s)",
            _RELPATH,
        )
        return
    n = text.count(_NEEDLE)
    if n == 0:
        raise RuntimeError(
            f"jit_warp: needle missing in {_RELPATH}: {_NEEDLE!r} -- "
            "upstream changed; re-derive (governance §3)"
        )
    p.write_text(text.replace(_NEEDLE, _REPL), encoding="utf-8")

    import sgl_kernel._adapters.registry as _reg

    # One allowlist record per needle occurrence (registry fingerprints by
    # (op, detail); the needle is identical for both loops, so a single
    # (op, needle) pair is registered once -- the second occurrence is the
    # same immutable text edit of the same file).
    _reg.register(
        "per_token_quant",  # the failing instantiation: fp32 warp max in the
        # indexer k-store; catalog has no warp-primitive entry
        "gfx93x-hipcc-portability",
        target=f"text:sglang/kernels/jit/{_RELPATH}",
        mechanism=_reg.MECH_TEXT,
        detail=_NEEDLE,
    )
    LOG.warning(
        "HCU adapter: JIT warp reduce portability patch installed (%d edits)",
        n,
    )
