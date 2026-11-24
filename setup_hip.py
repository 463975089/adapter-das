# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Optional native (hipcc) extension build for the sgl_kernel shadow package.

Adapted from sglang-das python/sglang/kernels/aot/setup_hip.py (itself a
setup_rocm.py derivative): hipcc/hipify toolchain, TORCH_LIBRARY(sgl_kernel)
registrations copied verbatim from upstream so `torch.ops.sgl_kernel.*`
resolves identically to the official wheel.

The pure-python shadow wheel (direct `pip wheel` / setup.py) remains for
tests and dev installs, where the python tiers (triton/torch) auto-activate.
packaging/build.sh builds THIS native variant for nightly/production: the
wheel version carries its ABI label
(<public>+dtk<DTK>.torch<TORCH>[.<stamp>]), and AMDGPU_TARGETS selects every
--offload-arch target. Building this extension produces a wheel whose
`sgl_kernel.native_ops` .so registers the native implementations; at import
time sgl_kernel/_native.py prefers them when present (BACKEND_CATALOG "cpp"
tier) and silently falls back otherwise.

Build (DTK container, hipcc on PATH, HIP torch installed):
    AMDGPU_TARGETS="gfx936;gfx938" python setup_hip.py bdist_wheel  # -> dist/sgl_kernel-*.whl
    # or in place for a live tree:
    python setup_hip.py build_ext --inplace

Ops covered (see csrc/ sources for the per-op porting notes):
  merge_state_v2 / concat_mla_k / concat_mla_absorb_q / copy_to_gpu_no_ce /
  sgl_per_token_group_quant_8bit / sgl_per_token_group_quant_8bit_v2 /
  silu_and_mul / gelu_tanh_and_mul / gelu_and_mul / gelu_quick
  (the four activations are vendored VERBATIM from upstream aot
  csrc/elementwise/activation.cu + its include/hip chain, and remove the
  most frequent serve-time ninja JIT compiles once installed)
"""

import os
import platform
import re
import sys
from pathlib import Path

import torch
from setuptools import Extension, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
arch = platform.machine().lower()

if torch.version.hip is None:
    raise SystemExit(
        "setup_hip.py: the installed torch is not a HIP/DTK build "
        "(torch.version.hip is None)"
    )


def _pin_module():
    # shared renderer: keeps setup_hip.py and hcu-adapter's interlock pin
    # byte-for-byte identical (same source of truth as ci/l0_gate).
    import importlib.util

    path = root / "packaging" / "hcu-adapter" / "_pin.py"
    spec = importlib.util.spec_from_file_location("_das_build_version", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _platform_tag():
    tag = os.environ.get("DAS_PLATFORM_TAG", "").strip().strip(".")
    if tag:
        return tag
    dtk = (os.environ.get("DAS_DTK_TAG")
           or os.environ.get("DTK_VERSION")
           or (f"dtk{os.environ['DTK_VER']}" if os.environ.get("DTK_VER") else ""))
    if not dtk:
        raise SystemExit(
            "setup_hip.py: cannot determine the DTK version; set "
            "DAS_PLATFORM_TAG or DTK_VERSION/DTK_VER so the wheel records "
            "which DTK/torch ABI it was built against"
        )
    dtk = re.sub(r"[^a-z0-9]", "", dtk.lower().removeprefix("dtk"))
    torch_nums = re.findall(r"\d+", torch.__version__.split("+", 1)[0])[:3]
    return f"dtk{dtk}.torch{''.join(torch_nums)}"


def _version():
    return _pin_module().kernel_version(platform=_platform_tag())


def _rocm_archs():
    archs = [a for a in re.split(r"[;,\s]+",
                                 os.environ.get("AMDGPU_TARGETS", "")) if a]
    if not archs and torch.cuda.is_available():
        archs = [torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]]
    if not archs:
        raise SystemExit(
            "setup_hip.py: AMDGPU_TARGETS is empty and no GPU is visible; "
            "set it to the target arch list (e.g. gfx936;gfx938)"
        )
    return list(dict.fromkeys(archs))


# torch.utils.cpp_extension assembles --offload-arch from PYTORCH_ROCM_ARCH
# (it does not read AMDGPU_TARGETS); bridge every requested target here.
os.environ["PYTORCH_ROCM_ARCH"] = ";".join(_rocm_archs())


operator_namespace = "sgl_kernel"
include_dirs = [
    root / "include",
    root / "csrc",
]

sources = [
    "csrc/common_extension_hcu.cc",
    "csrc/attention/merge_attn_states.cu",
    "csrc/elementwise/activation.cu",
    "csrc/elementwise/concat_mla_absorb_q_hcu.cu",
    "csrc/elementwise/concat_mla_k_hcu.cu",
    "csrc/elementwise/copy.cu",
    "csrc/gemm/per_token_group_quant_8bit_hcu.cu",
]

cxx_flags = ["-O3", "-w"]
libraries = ["hiprtc", "amdhip64", "c10", "torch", "torch_python"]
extra_link_args = [
    "-Wl,-rpath,$ORIGIN/../../torch/lib",
    f"-L/usr/lib/{arch}-linux-gnu",
]

hipcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    # mirrors the upstream ROCm build: activation.cu's vendored ROCm branch
    # (sgl_hip:: kernels) is selected by USE_ROCM, inert for the other TUs
    "-DUSE_ROCM=1",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
    "-w",
]

ext_modules = [] if "--hisparse-only" in sys.argv else [
    # Loaded via ctypes, deliberately separate from Torch op registration.
    Extension(
        "sgl_kernel.cpu_metadata",
        sources=["csrc/distributed/cpu_metadata.cc"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
    CUDAExtension(
        # sgl_kernel.native_ops, NOT common_ops: same TORCH_LIBRARY
        # namespace, distinct module file — no clash if an official wheel
        # is ever co-installed.
        name="sgl_kernel.native_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={"nvcc": hipcc_flags, "cxx": cxx_flags},
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    ),
]

# Shared native build toolchain, with an explicit small isolation lane.
# --hisparse-only is a build command option, not a runtime environment switch.
def _hisparse_build_module():
    # An installed sitecustomize may have cached an older hcu_adapter package.
    # Like _pin_module(), bind build metadata to this checkout, not that cache.
    import importlib.util

    path = root / "hcu_adapter" / "kernels" / "hisparse_cpp.py"
    spec = importlib.util.spec_from_file_location("_das_hisparse_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hisparse_ext = _hisparse_build_module().extension(
    root, libraries=libraries, extra_link_args=extra_link_args)
if "--hisparse-only" in sys.argv:
    sys.argv.remove("--hisparse-only")
    ext_modules = [hisparse_ext]
else:
    ext_modules.append(hisparse_ext)

setup(
    name="sgl-kernel",
    version=_version(),
    packages=["sgl_kernel"],
    package_dir={"": "."},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
