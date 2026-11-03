"""JIT elementwise/concat_mla.cuh portability patch — HIP arm.

Compiler-input transformation: _apply operates only on private adapter staging
assets via jit_sources. install() delegates to that complete bundle pipeline;
no community source or module origin is rewritten.


MLA-absorb concat (GLM-5.2/DSA models) JIT-compiles
`sglang/kernels/jit/csrc/elementwise/concat_mla.cuh`. The file's memory
helpers are hand-written PTX inline asm:

- `get_lane_id` reads `%laneid` (NVIDIA-only register);
- `st_na_global_*` / `ld_na_global_*` use `st.global.L1::no_allocate.*`
  with PTX cache-hint syntax and the `"l"` register constraint;
- `prefetch_L2` uses `prefetch.global.L2`.

DTK hipcc rejects all of these (invalid % escape / invalid input
constraint 'l'). Portable HIP replacements, same semantics:

- lane id: delegate to `sglang::device::get_lane_id()` from
  sgl_kernel/utils.cuh (HIP arm: `threadIdx.x % kNumThreads`, the
  non-asm identity upstream itself uses);
- non-atomic 32-bit global load/store: plain C++ dereference through
  the original pointer types (same addresses, same sizes; the
  L1::no_allocate PTX hint is a cache-policy optimization, not a
  semantic order constraint — these are non-atomic streaming accesses
  ordered by the surrounding __syncthreads/stream semantics);
- L2 prefetch: AMDGCN `s_prefetch_data` via builtin under
  `__HIP_DEVICE_COMPILE__` (guard: only gfx950+ actually implements
  the s-memory prefetch; keep it enabled off — a missing optimize hint
  must never be a correctness dependency). Actually emitted is a no-op;
  upstream enables it via ENABLE_L2_PREFETCH which we never define.

Same text-patch mechanics as jit_rope/jit_store_cache/jit_warp.
"""

import logging
import pathlib

LOG = logging.getLogger("hcu.adapter")

_RELPATH = "elementwise/concat_mla.cuh"

# (needle, replacement) — exact upstream strings (0.5.21)
_PATCHES = (
    (
        "SGL_DEVICE int get_lane_id() {\n"
        "  int lane_id;\n"
        '  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));\n'
        "  return lane_id;\n"
        "}",
        "SGL_DEVICE int get_lane_id() {\n"
        "  // HCU adapter: %laneid is NVIDIA-only; the canonical utils.cuh\n"
        "  // helper has the HIP identity (threadIdx.x %% kNumThreads)\n"
        "  return static_cast<int>(::sglang::device::get_lane_id());\n"
        "}",
    ),
    (
        "SGL_DEVICE void st_na_global_v1(const int* ptr, int v) {\n"
        '  asm volatile("st.global.L1::no_allocate.s32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");\n'
        "}",
        "SGL_DEVICE void st_na_global_v1(const int* ptr, int v) {\n"
        "  // HCU adapter: PTX store + cache hint -> plain volatile store\n"
        '  *const_cast<int*>(ptr) = v;\n'
        "}",
    ),
    (
        "SGL_DEVICE void st_na_global_v2(const int2* ptr, const int2& v) {\n"
        '  asm volatile("st.global.L1::no_allocate.v2.s32 [%0], {%1, %2};" ::"l"(ptr), "r"(v.x), "r"(v.y) : "memory");\n'
        "}",
        "SGL_DEVICE void st_na_global_v2(const int2* ptr, const int2& v) {\n"
        '  int2* p = const_cast<int2*>(ptr);\n'
        "  p->x = v.x;\n"
        "  p->y = v.y;\n"
        "}",
    ),
    (
        "SGL_DEVICE int ld_na_global_v1(const int* ptr) {\n"
        "  int r;\n"
        '  asm volatile("ld.global.nc.L1::no_allocate.s32 %0, [%1];" : "=r"(r) : "l"(ptr));\n'
        "  return r;\n"
        "}",
        "SGL_DEVICE int ld_na_global_v1(const int* ptr) {\n"
        '  return *ptr;\n'
        "}",
    ),
    (
        "SGL_DEVICE int2 ld_na_global_v2(const int2* ptr) {\n"
        "  int2 r;\n"
        '  asm volatile("ld.global.nc.L1::no_allocate.v2.s32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(ptr));\n'
        "  return r;\n"
        "}",
        "SGL_DEVICE int2 ld_na_global_v2(const int2* ptr) {\n"
        '  return *ptr;\n'
        "}",
    ),
)


# Host-side TensorMatcher hardcodes the CUDA dldevice enum; on ROCm the
# runtime hands it kDLROCM tensors and match fails ("Device value [rocm:N]
# not in the allowed options: [cuda]"). Same site upstream already runs the
# kDLGPU = kDLROCM indirection for this exact reason (utils.cuh). Six
# identical sites (3x concat_mla_k + 3x concat_mla_absorb_q); replace them
# wholesale -- counting occurrences, not first-match-only, so a partially
# patched file (e.g. a previous install raced) converges to fully patched.
_DLDEVICE_NEEDLE = "        .with_device<kDLCUDA>(device)"
_DLDEVICE_REPL = "        .with_device<kDLROCM>(device)"


def _apply(root, register):
    """Transform adapter-owned staging files; never pass a community root."""

    p = root / "csrc" / _RELPATH
    text = p.read_text(encoding="utf-8")
    registered = []
    for needle, repl in _PATCHES:
        if repl in text:
            continue  # already patched (spawn'd child / second install)
        if needle not in text:
            raise RuntimeError(
                f"jit_concat_mla: needle missing in {_RELPATH}: {needle[:70]!r} -- "
                "upstream changed; re-derive (governance §3)"
            )
        text = text.replace(needle, repl)
        registered.append(needle)
    n_dev = text.count(_DLDEVICE_NEEDLE)
    if n_dev:
        text = text.replace(_DLDEVICE_NEEDLE, _DLDEVICE_REPL)
        registered.extend([_DLDEVICE_NEEDLE] * n_dev)
    from hcu_adapter.atomic_io import atomic_write

    atomic_write(p, text)
    applied = len(registered)

    import hcu_adapter.registry as _reg

    # TEXT_PATCH_ALLOWLIST keys are (op, exact needle) — one record per
    # needle, jit_quant_fp8 pattern.
    for needle in registered:
        register(
            "platform_detect",  # ISA-portability routing patch; nearest catalog op
            "gfx93x-hipcc-portability",
            target=f"text:sglang/kernels/jit/csrc/{_RELPATH}",
            mechanism=_reg.MECH_TEXT,
            detail=needle,
        )
    LOG.warning(
        "HCU adapter: JIT concat_mla portability patch installed (%d edits)",
        applied,
    )


def install():
    """Compatibility entry: prepare the complete immutable JIT input bundle."""
    from .jit_sources import install as install_inputs

    return install_inputs()
