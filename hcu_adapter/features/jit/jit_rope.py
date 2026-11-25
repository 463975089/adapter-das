# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""JIT rope portability patch — gfx936 DTK hipcc (clang) vs nvcc-isms.

Compiler-input transformation: _apply operates only on private adapter staging
assets via jit_sources. install() delegates to that complete bundle pipeline;
no community source or module origin is rewritten.


Upstream sglang 0.5.21 JIT-compiles its fused rope kernel from
`sglang/kernels/jit/csrc/elementwise/rope.cuh` at first call. gfx936's DTK
clang-based hipcc rejects two nvcc-only constructs:

1. bare `__syncwarp()` — clang/HIP wants the mask form; our CUDA shims
   header (see below) defines `__syncwarp(mask=0xffffffff)` for both worlds.
2. `const auto [a, b] = HIP_vector_type<float,2>` — structured binding into
   a class with an anonymous union member is ill-formed here; replace with
   explicit `.x/.y` member reads.

This is a source-level compatibility patch (behavior identical on both
platforms), applied to adapter-owned .cuh assets before publication and before
ninja launches hipcc. The loader hashes
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
    # rope_rotate_head non-neox arm (hit by GLM-5.2-DSA indexer rope on
    # gfx938 DTK hipcc -- same anonymous-union ABI as the bindings above)
    (
        "elementwise/rope.cuh",
        "      const auto [x, y] = cast<fp32x2_t>(input_vec[j]);",
        "      fp32x2_t _vxy = cast<fp32x2_t>(input_vec[j]);"
        " const auto x = _vxy.x, y = _vxy.y;",
    ),
    # pos_enc.cuh DISPATCH macro names nv_bfloat16 (CUDA-only builtin) and
    # lacks the bf16 header entirely. Include the shim's HIP-alias header and
    # alias to __nv_bfloat16 (which DTK hipcc knows via cuda_bf16.h pull-in
    # from sgl_kernel headers); keep one patch per needle.
    (
        "elementwise/pos_enc.cuh",
        "#include <cuda_fp16.h>",
        '#include <cuda_fp16.h>\n#include <cuda_bf16.h>  // HCU: nv_bfloat16 alias source on HIP',
    ),
    (
        "elementwise/pos_enc.cuh",
        "launch_kernel<nv_bfloat16, IS_NEOX>(__VA_ARGS__);",
        "launch_kernel<__hip_bfloat16, IS_NEOX>(__VA_ARGS__);",
    ),
)

# MiniMax's 0.5.23 fused Q/K path has the same HIP float2 ABI issue.
_MINIMAX_SOURCE = "minimax/fused_gemma_qknorm_rope.cuh"
_MINIMAX_PATCHES = tuple(
    (
        f"    const auto [{kind}{x}, {kind}{y}] = cast<fp32x2_t>"
        f"(bf16x2_t{{{kind}{x}_bf16, {kind}{y}_bf16}});",
        f"    const auto {kind}{x}{y} = cast<fp32x2_t>"
        f"(bf16x2_t{{{kind}{x}_bf16, {kind}{y}_bf16}});\n"
        f"    const auto {kind}{x} = {kind}{x}{y}.x, {kind}{y} = {kind}{x}{y}.y;",
    )
    for kind in ("i", "w")
    for x, y in ((0, 1), (2, 3))
)


def _apply_minimax(csrc, register):
    p = csrc / _MINIMAX_SOURCE
    if not p.exists():  # older SGLang releases do not ship this kernel
        return 0
    from hcu_adapter.infra.atomic_io import atomic_write
    from hcu_adapter.infra.registry import MECH_TEXT

    original = p.read_text(encoding="utf-8")
    text = original
    for needle, repl in _MINIMAX_PATCHES:
        if repl in text:
            continue
        if text.count(needle) != 1:
            raise RuntimeError(f"jit_rope: expected one {_MINIMAX_SOURCE} needle: {needle!r}")
        text = text.replace(needle, repl)
    if text != original:
        atomic_write(p, text)
    register(
        "rotary_embedding",
        "minimax-qknorm-hipcc-portability",
        target=f"text:sglang/kernels/jit/csrc/{_MINIMAX_SOURCE}",
        mechanism=MECH_TEXT,
        detail=_MINIMAX_PATCHES[0][0],
    )
    return len(_MINIMAX_PATCHES)


_DIFFUSION_SOURCE = "diffusion/qknorm_rope.cuh"
_DIFFUSION_PATCHES = (
    (
        "const auto [x0, x1] = cast<fp32x2_t>(input_vec[j]);",
        "const auto xy = cast<fp32x2_t>(input_vec[j]);\n"
        "      const auto x0 = xy.x, x1 = xy.y;",
        1,
    ),
    (
        "\n      const auto [w0, w1] = cast<fp32x2_t>(weight_vec[j]);",
        "\n      const auto weights = cast<fp32x2_t>(weight_vec[j]);\n"
        "      const auto w0 = weights.x, w1 = weights.y;",
        1,
    ),
    (
        "__shfl_sync(kActiveMask, partner_bits, partner_lane)",
        "__shfl(partner_bits, partner_lane, kWarpThreads)",
        1,
    ),
    (
        "__shfl_sync(kActiveMask, elems[i], partner_lane)",
        "__shfl(elems[i], partner_lane, kWarpThreads)",
        1,
    ),
    ("device.set_options<kDLCUDA>();", "device.set_options<kDLGPU>();", 3),
)


def _apply_diffusion(csrc, register):
    path = csrc / _DIFFUSION_SOURCE
    if not path.exists():
        return 0
    original = path.read_text(encoding="utf-8")
    if not all(
        original.count(needle) == count for needle, _, count in _DIFFUSION_PATCHES
    ):
        return 0

    from hcu_adapter.infra.atomic_io import atomic_write
    from hcu_adapter.infra.registry import MECH_TEXT

    text = original
    for needle, replacement, _ in _DIFFUSION_PATCHES:
        text = text.replace(needle, replacement)
    atomic_write(path, text)
    register(
        "rotary_embedding",
        "diffusion-qknorm-hipcc-portability",
        target=f"text:sglang/kernels/jit/csrc/{_DIFFUSION_SOURCE}",
        mechanism=MECH_TEXT,
        detail=_DIFFUSION_PATCHES[0][0],
    )
    return sum(count for _, _, count in _DIFFUSION_PATCHES)


# shim appended to rope.cuh: mask-form syncwarp for HIP; identity on CUDA
# (nvcc's __builtin has the mask overload; define only under __HIP__).
_SHIM = """/*
 HCU adapter portability shim (appended by hcu_adapter.features.jit.jit_rope)
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


def _apply(root, register):
    """Transform adapter-owned staging files; never pass a community root."""

    csrc = root / "csrc"
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
        from hcu_adapter.infra.atomic_io import atomic_write

        atomic_write(p, text.replace(needle, repl))
        applied += 1
    shim_marker = "HCU adapter portability shim"
    rope_p = csrc / "elementwise/rope.cuh"
    rope_text = rope_p.read_text(encoding="utf-8")
    if shim_marker not in rope_text:
        # prepend: C/C++ need the declaration before first use (call sites
        # sit mid-file at ~line 230)
        from hcu_adapter.infra.atomic_io import atomic_write

        atomic_write(rope_p, _SHIM + rope_text)

    import hcu_adapter.infra.registry as _reg

    register(
        "rotary_embedding",
        "gfx936-hipcc-portability",
        target="text:sglang/kernels/jit/csrc/elementwise/rope.cuh",
        mechanism=_reg.MECH_TEXT,
        detail="  __syncwarp();  // to avoid warp divergence",
    )
    applied += _apply_minimax(csrc, register)
    applied += _apply_diffusion(csrc, register)
    LOG.warning(
        "HCU adapter: JIT rope portability patch installed (%d edits + shim)",
        applied,
    )


def install():
    """Compatibility entry: prepare the complete immutable JIT input bundle."""
    from .jit_sources import install as install_inputs

    return install_inputs()
