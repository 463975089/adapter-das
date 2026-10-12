"""JIT rope portability patch — gfx936 DTK hipcc (clang) vs nvcc-isms.

Upstream sglang 0.5.21 JIT-compiles its fused rope kernel from
`sglang/kernels/jit/csrc/elementwise/rope.cuh` at first call. gfx936's DTK
clang-based hipcc rejects two nvcc-only constructs:

1. bare `__syncwarp()` — clang/HIP wants the mask form; our CUDA shims
   header (see below) defines `__syncwarp(mask=0xffffffff)` for both worlds.
2. `const auto [a, b] = HIP_vector_type<float,2>` — structured binding into
   a class with an anonymous union member is ill-formed here; replace with
   explicit `.x/.y` member reads.

This is a source-level compatibility patch (behavior identical on both
platforms), applied by reading the module's .cuh files at import time and
rewriting the needle strings before ninja launches hipcc. The loader hashes
the generated build text, so patched sources land in their own cache entry —
upstream trees stay untouched and stock NVIDIA hosts never see this bridge's
output (they don't take the HIP route at all).
"""

import logging
import pathlib

LOG = logging.getLogger("hcu.adapter")

# (relative to sglang/kernels/jit/csrc, needle, replacement)
_PATCHES = (
    (
        "elementwise/rope.cuh",
        "  __syncwarp();  // to avoid warp divergence",
        "  __syncwarp_full_mask();  // to avoid warp divergence (HCU portability shim)",
    ),
    (
        "elementwise/rope.cuh",
        "      const auto [x0, x1] = cast<fp32x2_t>(input_vec_x[j]);",
        "      fp32x2_t _vx = cast<fp32x2_t>(input_vec_x[j]);"
        " const auto x0 = _vx.x, x1 = _vx.y;",
    ),
    (
        "elementwise/rope.cuh",
        "      const auto [y0, y1] = cast<fp32x2_t>(input_vec_y[j]);",
        "      fp32x2_t _vy = cast<fp32x2_t>(input_vec_y[j]);"
        " const auto y0 = _vy.x, y1 = _vy.y;",
    ),
    (
        "elementwise/rope.cuh",
        "      const auto [cos_0, cos_1] = cos_pair[j];",
        "      const auto cos_0 = cos_pair[j].x, cos_1 = cos_pair[j].y;",
    ),
    (
        "elementwise/rope.cuh",
        "      const auto [sin_0, sin_1] = sin_pair[j];",
        "      const auto sin_0 = sin_pair[j].x, sin_1 = sin_pair[j].y;",
    ),
)

# shim appended to rope.cuh: mask-form syncwarp for HIP; identity on CUDA
# (nvcc's __builtin has the mask overload; define only under __HIP__).
_SHIM = """/*
 HCU adapter portability shim (appended by sgl_kernel._adapters.jit_rope)
 */
#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIP_PLATFORM_AMD__)
__attribute__((device, always_inline)) inline void __syncwarp_full_mask(unsigned mask = 0xffffffffu) {
  __builtin_amdgcn_wave_barrier();  // gfx93x wavefront barrier (mask ignored)
}
#elif defined(__CUDA_ARCH__)
__attribute__((device, always_inline)) inline void __syncwarp_full_mask(unsigned mask = 0xffffffffu) {
  __syncwarp(mask);
}
#endif
"""


def install():
    import sglang.kernels.jit as _jit_root

    csrc = pathlib.Path(_jit_root.__file__).parent / "csrc"
    applied = 0
    for rel, needle, repl in _PATCHES:
        p = csrc / rel
        text = p.read_text(encoding="utf-8")
        if repl in text:
            applied += 1  # already patched (spawn'd child / second install)
            continue
        if needle not in text:
            raise RuntimeError(
                f"jit_rope: needle missing in {rel}: {needle[:60]!r} -- "
                "upstream changed; re-derive (governance §3)"
            )
        p.write_text(text.replace(needle, repl), encoding="utf-8")
        applied += 1
    shim_marker = "HCU adapter portability shim"
    rope_p = csrc / "elementwise/rope.cuh"
    rope_text = rope_p.read_text(encoding="utf-8")
    if shim_marker not in rope_text:
        # prepend: C/C++ need the declaration before first use (call sites
        # sit mid-file at ~line 230)
        rope_p.write_text(_SHIM + rope_text, encoding="utf-8")

    import sgl_kernel._adapters.registry as _reg

    _reg.register(
        "rotary_embedding",
        "gfx936-hipcc-portability",
        target="text:sglang/kernels/jit/csrc/elementwise/rope.cuh",
        mechanism=_reg.MECH_TEXT,
        detail="  __syncwarp();  // to avoid warp divergence",
    )
    LOG.warning(
        "HCU adapter: JIT rope portability patch installed (%d edits + shim)",
        applied,
    )
