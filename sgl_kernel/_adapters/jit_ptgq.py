"""JIT per_token_group_quant portability — DTK hipcc has no __all_sync.

Upstream v0.5.22 JIT-compiles the per-token-group quantization family from
`sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh`. DTK's clang-based
hipcc declares the plain warp vote `__all(pred)` (and 1-arg `__ballot`) but
NOT the mask-form `__all_sync(mask, pred)` (nmz18 gfx938 probe, 2026-10-20),
so every instantiation of the family dies at name lookup -- glm53f serving
hits it in decode graph capture (fp8 block-quant MoE gate_up path).

All upstream call sites vote with a full mask (`warp::kFullMask` /
`0xffffffffu`) from kernel entry where the whole wave is active, so the
plain `__all(pred)` (warp-wide AND over active lanes) is semantics-equal:
on USE_ROCM warp.cuh defines kFullMask as the full wave64 mask and the
sorted-rows guard documents whole-warp participation.

Same needle in two more files patched for uniformity: hicache_tma.cuh
(two mid-kernel full-mask votes) and the sm100a diffusion kernel (never
built on gfx938; inert there, correct on CUDA where __all also exists).

Text-patch discipline per jit_rope: needle missing in an existing file
raises loud (upstream changed); file absent (older trees without the
module) skips quietly.
"""

import logging
import pathlib

LOG = logging.getLogger("hcu.adapter")

# (relative to sglang/kernels/jit/csrc, needle, replacement)
_PATCHES = (
    (
        "gemm/per_token_group_quant.cuh",
        "if (__all_sync(warp::kFullMask, skip)) return;",
        "if (__all(skip)) return;  // HCU: DTK hipcc has no __all_sync",
    ),
    (
        "kvcacheio/hicache_tma.cuh",
        "run = __all_sync(warp::kFullMask, run);",
        "run = __all(run);  // HCU: DTK hipcc has no __all_sync",
    ),
    (
        "kvcacheio/hicache_tma.cuh",
        "run_dst = __all_sync(warp::kFullMask, run_dst);",
        "run_dst = __all(run_dst);  // HCU: DTK hipcc has no __all_sync",
    ),
    (
        "diffusion/vsa_block_sparse_sm100/block_sparse_kernel_sm100a.cuh",
        "bool skip = __all_sync(0xffffffffu, alpha == 1.0f);",
        "bool skip = __all(alpha == 1.0f);  // HCU: DTK hipcc has no __all_sync",
    ),
)


def install():
    import sglang.kernels.jit as _jit_root

    csrc = pathlib.Path(_jit_root.__file__).parent / "csrc"
    by_file = {}
    for rel, needle, repl in _PATCHES:
        p = csrc / rel
        if not p.exists():
            continue  # tree predates the module (e.g. v0.5.21) -- not ours
        text = p.read_text(encoding="utf-8")
        if repl in text:
            continue  # already patched (spawn'd child / second install)
        if needle not in text:
            raise RuntimeError(
                f"jit_ptgq: needle missing in {rel}: {needle[:60]!r} -- "
                "upstream changed; re-derive (governance §3)"
            )
        p.write_text(text.replace(needle, repl), encoding="utf-8")
        by_file[rel] = by_file.get(rel, 0) + 1

    import sgl_kernel._adapters.registry as _reg

    _reg.register(
        "per_token_quant",  # catalog op: per-token/per-group quant kernels
        "dtk-hipcc-no-all-sync",
        target="text:sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh",
        mechanism=_reg.MECH_TEXT,
        detail="__all_sync(kFullMask, skip) -> __all(skip)",
    )
    if by_file:
        LOG.warning(
            "HCU adapter: JIT per_token_group_quant portability patch "
            "installed (%s)",
            ", ".join(f"{k}:{v}" for k, v in sorted(by_file.items())),
        )
