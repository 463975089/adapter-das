"""JIT per_token_group_quant portability — DTK hipcc has no __all_sync.

Upstream v0.5.22 JIT-compiles the per-token-group quantization family from
`sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh`. DTK's clang-based
hipcc declares the plain warp vote `__all(pred)` (and 1-arg `__ballot`) but
NOT the mask-form `__all_sync(mask, pred)` (nmz18 gfx938 probe, 2026-10-20),
so every instantiation of the family dies at name lookup -- glm53f serving
hits it in decode graph capture (fp8 block-quant MoE gate_up path).

The upstream call site votes with a full mask (`warp::kFullMask`) from
kernel entry where the whole wave is active, so the plain `__all(pred)`
(warp-wide AND over active lanes) is semantics-equal: on USE_ROCM warp.cuh
defines kFullMask as the full wave64 mask and the sorted-rows guard
documents whole-warp participation.

(The same `__all_sync` needle also exists in kvcacheio/hicache_tma.cuh and
the sm100a diffusion kernel; those paths are not exercised on HCU today --
hicache off in serving configs, sm100a never built on gfx938 -- so they
stay unpatched pending their own governance entries if ever needed.)

Text-patch discipline per jit_rope: needle missing in an existing file
raises loud (upstream changed); file absent (older trees without the
module) skips quietly.
"""

import logging
import pathlib

LOG = logging.getLogger("hcu.adapter")

_RELPATH = "gemm/per_token_group_quant.cuh"
_NEEDLE = "if (__all_sync(warp::kFullMask, skip)) return;"
_REPL = "if (__all(skip)) return;  // HCU: DTK hipcc has no __all_sync"


def install():
    import sglang.kernels.jit as _jit_root
    import hcu_adapter.registry as _reg

    csrc = pathlib.Path(_jit_root.__file__).parent / "csrc"
    p = csrc / _RELPATH
    if not p.exists():
        LOG.info("jit_ptgq: %s absent (tree predates the module)", _RELPATH)
        return
    text = p.read_text(encoding="utf-8")
    if _REPL not in text:
        if _NEEDLE not in text:
            raise RuntimeError(
                f"jit_ptgq: needle missing in {_RELPATH}: {_NEEDLE[:60]!r} -- "
                "upstream changed; re-derive (governance §3)"
            )
        from hcu_adapter.atomic_io import atomic_write

        atomic_write(p, text.replace(_NEEDLE, _REPL))

    _reg.register(
        "per_token_quant",
        "dtk-hipcc-no-all-sync",
        target=f"text:sglang/kernels/jit/csrc/{_RELPATH}",
        mechanism=_reg.MECH_TEXT,
        detail=_NEEDLE,
    )
    LOG.warning(
        "HCU adapter: JIT per_token_group_quant portability patch installed"
        " (__all_sync(kFullMask, skip) -> __all(skip))"
    )
