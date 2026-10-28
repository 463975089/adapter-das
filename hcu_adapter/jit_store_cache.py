"""JIT dsa/fused_store_index_cache.cuh portability patch — HIP arm.

GLM-5.2-DSA indexer k-cache store (unfused path) JIT-compiles
`sglang/kernels/jit/csrc/dsa/fused_store_index_cache.cuh`. On HCU
(USE_ROCM) the type aliases make `fp8x2_e4m3_t` a bare `uint16_t`
(utils.cuh), so three nvcc-friendly constructs fail DTK hipcc:

1. `pack_fp8` brace-constructs `fp8x2_e4m3_t{fp32x2_t{...}}` — uint16_t
   has no such constructor. We delegate to the proven packer from
   `sgl_kernel/deepseek_v4/fp8_utils.cuh` (software branch on gfx938,
   E4M3FN / 448-clip flavour — exactly what this file's own
   `fp8_e4m3_clip` assumes via kFP8E4M3Max).
2. Two structured bindings into `fp32x2_t` (= HIP `float2`, a class with
   an anonymous union member — same ill-formed-decompose as rope.cuh).
   Replaced with explicit `.x/.y` member reads.

Same text-patch mechanics as jit_rope: sources rewritten pre-hipcc on
this host only; upstream trees untouched; loader hash isolates the cache
entry.
"""

import logging
import pathlib

LOG = logging.getLogger("hcu.adapter")

_RELPATH = "dsa/fused_store_index_cache.cuh"

# (needle, replacement)
_PATCHES = (
    (
        "#include <sgl_kernel/vec.cuh>\n#include <sgl_kernel/warp.cuh>",
        "#include <sgl_kernel/vec.cuh>\n#include <sgl_kernel/warp.cuh>\n"
        "#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>  // HCU adapter: HIP pack_fp8 delegate",
    ),
    (
        "[[maybe_unused]]\n"
        "SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {\n"
        "  return fp8x2_e4m3_t{fp32x2_t{fp8_e4m3_clip(x), fp8_e4m3_clip(y)}};\n"
        "}",
        "[[maybe_unused]]\n"
        "SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {\n"
        "#if defined(USE_ROCM)\n"
        "  // HCU adapter: HIP fp8x2_e4m3_t is a bare uint16_t (no brace ctor);\n"
        "  // delegate to the proven E4M3FN packer (same 448 clip as kFP8E4M3Max here)\n"
        "  return sglang::deepseek_v4::fp8::pack_fp8(x, y);\n"
        "#else\n"
        "  return fp8x2_e4m3_t{fp32x2_t{fp8_e4m3_clip(x), fp8_e4m3_clip(y)}};\n"
        "#endif\n"
        "}",
    ),
    (
        "  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);",
        "  fp32x2_t _v0 = cast<fp32x2_t>(elems[0]);"
        " const auto x0 = _v0.x, x1 = _v0.y;",
    ),
    (
        "  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);",
        "  fp32x2_t _v1 = cast<fp32x2_t>(elems[1]);"
        " const auto y0 = _v1.x, y1 = _v1.y;",
    ),
)


def install():
    import sglang.kernels.jit as _jit_root

    p = pathlib.Path(_jit_root.__file__).parent / "csrc" / _RELPATH
    text = p.read_text(encoding="utf-8")
    applied = 0
    registered = []
    for needle, repl in _PATCHES:
        if repl in text:
            applied += 1  # already patched (spawn'd child / second install)
        elif needle not in text:
            raise RuntimeError(
                f"jit_store_cache: needle missing in {_RELPATH}: {needle[:70]!r} -- "
                "upstream changed; re-derive (governance §3)"
            )
        else:
            text = text.replace(needle, repl)
            applied += 1
        registered.append(needle)
    from hcu_adapter.atomic_io import atomic_write

    atomic_write(p, text)

    import hcu_adapter.registry as _reg

    # TEXT_PATCH_ALLOWLIST keys are (op, exact needle) — one record per
    # needle (same-op/same-target re-registration of an immutable text
    # patch is the jit_quant_fp8 pattern).
    for needle in registered:
        _reg.register(
            "per_token_quant",  # catalog op: per-token fp8 quant+store kernel
            "gfx93x-hipcc-portability",
            target=f"text:sglang/kernels/jit/csrc/{_RELPATH}",
            mechanism=_reg.MECH_TEXT,
            detail=needle,
        )
    LOG.warning(
        "HCU adapter: JIT dsa store-cache portability patch installed (%d edits)",
        applied,
    )
