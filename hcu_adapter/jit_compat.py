"""JIT CUDA-header shim roots — gfx936 hipcc compatibility (behavior
restoration via the toolchain's own include path).

Upstream 0.5.21 JIT kernels include CUDA-only headers directly:

    #include <cuda_bf16.h>   -> HIP: <hip/hip_bf16.h>
    #include <cuda_fp8.h>    -> HIP: <hip/hip_fp8.h> (DTK)
    #include <cuda.h>,
    #include <cuda_runtime.h> -> HIP: <hip/hip_runtime.h>

The DTK toolchain already sits on the JIT compile line via
`toolchain.base_include_paths()` (`-I$ROCM_HOME/include`), so writing
forwarding headers into that tree makes every current and future JIT kernel
compile without touching a single upstream .cuh. Each forwarder maps its CUDA
name to the HIP equivalent and defers entirely to DTK's headers (types,
intrinsics, `__HIP_PLATFORM_AMD__` dispatch all stay vendor-defined).

Beyond the include, third-party JIT csrc is often written against CUDA
*tokens* (`__nv_fp8x2_e4m3`, `__nv_cvt_*`, team-image sgl-kernel JIT on
box16 dtk2604 — all 43 modules compiled only with these aliases, and the
same set originated on nmz26). The forwarders therefore also carry the
empirically-required `__nv_*` type/intrinsic aliases for the HIP
equivalents (proven on dtk2604/dtk26041; a future DTK that renames the
__hip_* side breaks loudly at include time — extend _FORWARDS then).

This mirrors kernel-space header shims the ROCm ecosystem ships (clang's
cuda_wrappers act identically for `#include <cuda_bf16.h>` under nvcc
compat mode) — DTK's llvm just doesn't enable that wrapper set on this
install, so the adapter provides the mapping explicitly.
"""

import logging
import pathlib

LOG = logging.getLogger("hcu.adapter")

# CUDA header name -> lines forwarding to the HIP/DTK equivalent
# (double-include safe: HIP headers have their own guards; repeated
# identical `using` aliases are likewise legal C++).
_FORWARDS = {
    "cuda_bf16.h": [
        "#pragma once",
        "#include <hip/hip_bf16.h>",
        # csrc written for CUDA gets __nv_bfloat16 only via this header,
        # so the alias belongs here (same argument as the fp8 set below).
        "using __nv_bfloat16 = __hip_bfloat16;",
    ],
    "cuda_fp16.h": [
        "#pragma once",
        "#include <hip/hip_fp16.h>",
    ],
    "cuda_fp8.h": [
        "#pragma once",
        "#include <hip/hip_fp8.h>",
        # __nv_* token aliases (empirically-required set, box16 2026-10-25;
        # note the _fnuz suffix on the HIP fp8 type name).
        "using __nv_fp8x2_storage_t = __hip_fp8x2_storage_t;",
        "using __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3_fnuz;",
        "#define __NV_E4M3 __HIP_E4M3",
        "#define __NV_SATFINITE __HIP_SATFINITE",
        "#define __nv_cvt_float2_to_fp8x2 __hip_cvt_float2_to_fp8x2",
        "#define __nv_cvt_fp8x2_to_halfraw2 __hip_cvt_fp8x2_to_halfraw2",
    ],
    "cuda.h": [
        "#pragma once",
        "#include <hip/hip_runtime.h>",
    ],
    "cuda_runtime.h": [
        "#pragma once",
        "#include <hip/hip_runtime.h>",
    ],
    "vector_types.h": [
        "#pragma once",
        "#include <hip/hip_runtime.h>",
    ],
}

_MARK = "// HCU adapter cuda->hip forwarder"


def install():
    from hcu_adapter import registry

    hip_include = _find_hip_include_root()
    if hip_include is None:
        LOG.warning(
            "HCU adapter: no HIP include root found; JIT cuda-header "
            "forwarders not installed"
        )
        return

    written = 0
    for name, lines in _FORWARDS.items():
        target = hip_include / name
        body = _MARK + "\n" + "\n".join(lines) + "\n"
        try:
            existing = target.read_text(encoding="utf-8") if target.exists() else ""
        except OSError:
            existing = ""
        if _MARK in existing:
            continue  # ours, already installed
        if existing.strip():
            continue  # a real file lives there (non-adapter toolchain): don't touch
        try:
            from hcu_adapter.atomic_io import atomic_write

            atomic_write(target, body)
            written += 1
        except OSError:
            LOG.warning("HCU adapter: cannot write %s (read-only root?)", target)
            return

    registry.register(
        "platform_detect",
        "jit-cuda-headers",
        target=f"fs:{hip_include}:cuda_*.h forwards",
        mechanism=registry.MECH_REBIND,
        detail=f"{written} cuda->hip forwarding headers in DTK include root",
    )
    LOG.warning(
        "HCU adapter: JIT cuda-header forwarders installed in %s (%d new)",
        hip_include,
        written,
    )


def _find_hip_include_root():
    """The same root `toolchain.base_include_paths()` puts on the compile
    line: $ROCM_HOME/include (DTK: /opt/dtk[-*]/include). Verify hip_bf16.h
    actually lives beneath it before returning."""
    import os

    cands = []
    env = os.environ.get("ROCM_HOME")
    if env:
        cands.append(pathlib.Path(env) / "include")
    for pat in ("/opt/dtk/include", "/opt/dtk*/include", "/opt/rocm/include"):
        root = pathlib.Path(pat)
        if "*" in pat:
            import glob

            cands.extend(pathlib.Path(p) / "include" for p in glob.glob(str(root.parent)))
        else:
            cands.append(root)
    for c in cands:
        if (c / "hip" / "hip_bf16.h").exists():
            return c
    return None
