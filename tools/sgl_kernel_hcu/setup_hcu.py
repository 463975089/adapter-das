"""setup_hcu.py — build sglang-kernel (sgl_kernel) for HCU/DTK.

Derived from the das tree's setup_hip.py (the proven DTK recipe: no ROCm
arch gate, DTK flag set) applied to the OFFICIAL v0.5.22 aot tree, plus the
L1 tier sources the stock recipes never wired:

  - elementwise/fused_add_rms_norm_kernel.cu   (fused_add_rmsnorm)
  - gemm/per_token_group_quant_8bit.cu / _v2   (sgl_per_token_group_quant_8bit*)
  - gemm/per_token_quant_fp8.cu                (rowwise fp8 quant)

Gates the upstream setup_rocm.py refuses (gfx942/950/1250 only, sys.exit on
gfx938) are replaced with the HCU family allowlist. The fp8 macro is E4M3:
HCU serves torch.float8_e4m3fn (see DSATokenToKVPool fp8 layout).

Env knobs: HCU_GPU_TARGET (default: detect, else gfx938),
HCU_KERNEL_VER_SUFFIX (dist version suffix).

NOTE the companion patch_load_utils.py: the stock python wrapper only globs
common_ops from sm90//sm100/ subdirectories, which is exactly how the dtk2604
images shipped an empty-op wheel. Our build puts the .so at the package
top level; the loader patch makes that loadable.
"""

import os
import platform
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
arch = platform.machine().lower()

HCU_TARGETS = {"gfx906", "gfx926", "gfx928", "gfx936", "gfx938"}


def _get_version():
    v = "0.0.0"
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                v = line.split("=")[1].strip().strip('"')
                break
    return v + os.environ.get("HCU_KERNEL_VER_SUFFIX", "+hcu.dtk2610")


operator_namespace = "sgl_kernel"
include_dirs = [
    root / "include",
    root / "include" / "impl",
    root / "csrc",
]
# flashinfer headers, fetched by build.sh into the workspace copy
# (upstream's CMakeLists FetchContent-pins commit bc29697b; we reuse the
# include/ + csrc/ trees for norm.cuh / vec_dtypes.cuh /
# pytorch_extension_utils.h). Absent dirs are skipped silently so the
# flashinfer-dependent sources below can be trimmed instead.
for _fi_sub in ("include", "csrc"):
    _fi_dir = root / "3rdparty" / "flashinfer" / _fi_sub
    if _fi_dir.is_dir():
        include_dirs.append(_fi_dir)
# Minimal CUDA-compat shim headers (populated by build.sh from DTK's
# cuda-12 compat tree). The FULL compat dir cannot go on the include path —
# it clashes with hipcc's own thrust/vector_types ("HIP compilation cannot
# use cuda/include/thrust") — so only the closure flashinfer's _hip headers
# actually include is copied: cuda_fp8.{h,hpp}, cuda_device_runtime_api.h.
_shim = root / "3rdparty" / "cuda_compat"
if _shim.is_dir():
    include_dirs.append(_shim)

sources = [
    # das setup_hip.py list minus the three das-private files
    # (attention/decode_metadata.cu, elementwise/concat_mla_absorb_q_hcu.cu,
    #  elementwise/l2norm_kernel.cu are not in the official tree)
    "csrc/allreduce/custom_all_reduce.hip",
    "csrc/allreduce/deterministic_all_reduce.hip",
    "csrc/allreduce/quick_all_reduce.cu",
    "csrc/elementwise/activation.cu",
    "csrc/elementwise/deepseek_v4_topk.cu",
    "csrc/elementwise/dsv4_norm_rope.cu",
    "csrc/elementwise/topk.cu",
    "csrc/grammar/apply_token_bitmask_inplace_cuda.cu",
    "csrc/moe/moe_align_kernel.cu",
    "csrc/moe/moe_topk_sigmoid_kernels.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
    # speculative/eagle_utils.cu excluded: its eagle.cuh lives only in the
    # das fork's tree, not the official one
    # attention/merge_attn_states.cu excluded: official version's bf16
    # spellings (__hip_bfloat16) do not compile under dtk2610's hip headers
    # (the das fork carries a patched variant); not on the L1 tier list
    "csrc/kvcacheio/transfer.cu",
    "csrc/memory/weak_ref_tensor.cpp",
    "csrc/elementwise/pos_enc.cu",
    "csrc/common_extension_rocm.cc",
    # L1 cpp-tier additions (present in tree, never wired by stock recipes).
    # per_token_group_quant_8bit.cu (v1) is excluded: it includes cuda_fp8.h
    # directly, which DTK does not ship — the _v2 kernel (flashinfer
    # vec_dtypes path) is the ported entry we route to.
    "csrc/elementwise/fused_add_rms_norm_kernel.cu",
    "csrc/gemm/per_token_group_quant_8bit_v2.cu",
    "csrc/gemm/per_token_quant_fp8.cu",
]

target = os.environ.get("HCU_GPU_TARGET", "")
if not target:
    if torch.cuda.is_available():
        try:
            target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        except Exception:
            pass
    target = target or "gfx938"
if target not in HCU_TARGETS:
    sys.exit(f"setup_hcu: unsupported HCU target '{target}' (allowed: {sorted(HCU_TARGETS)})")

cxx_flags = ["-O3", "-w"]
libraries = ["hiprtc", "amdhip64", "c10", "torch", "torch_python"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", f"-L/usr/lib/{arch}-linux-gnu"]

hipcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    f"--amdgpu-target={target}",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
    "-DHIP_FP8_TYPE_E4M3",  # HCU fp8 = e4m3fn (das recipe omits: their subset
    #                       # never compiled the quant kernels)
    "-w",
]

print(f"setup_hcu: target={target} sources={len(sources)} version={_get_version()}")

ext_modules = [
    CUDAExtension(
        name="sgl_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": hipcc_flags,
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    ),
]

setup(
    name="sglang-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
