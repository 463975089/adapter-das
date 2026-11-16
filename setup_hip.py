# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Optional native (hipcc) extension build for the sgl_kernel shadow package.

Adapted from sglang-das python/sglang/kernels/aot/setup_hip.py (itself a
setup_rocm.py derivative): hipcc/hipify toolchain, TORCH_LIBRARY(sgl_kernel)
registrations copied verbatim from upstream so `torch.ops.sgl_kernel.*`
resolves identically to the official wheel.

The pure-python shadow wheel (pyproject.toml) is UNCHANGED and remains the
default: pip install keeps working everywhere, and the python tiers
(triton/torch) auto-activate.  Building this extension produces a wheel whose
`sgl_kernel.native_ops` .so registers the native implementations; at import
time sgl_kernel/_native.py prefers them when present (BACKEND_CATALOG "cpp"
tier) and silently falls back otherwise.

Build (DTK container):
    python setup_hip.py bdist_wheel          # -> dist/sgl_kernel-*.whl
    # or in place for a live tree:
    python setup_hip.py build_ext --inplace

Ops covered (see csrc/ sources for the per-op porting notes):
  merge_state_v2 / concat_mla_k / concat_mla_absorb_q / copy_to_gpu_no_ce /
  sgl_per_token_group_quant_8bit / sgl_per_token_group_quant_8bit_v2
"""

import platform
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
arch = platform.machine().lower()


def _version():
    src = (root / "sgl_kernel" / "version.py").read_text(encoding="utf-8")
    import re

    m = re.search(r'^__version__ = "(.+?)"$', src, re.M)
    if not m:
        raise RuntimeError("cannot read __version__ from sgl_kernel/version.py")
    return m.group(1)


operator_namespace = "sgl_kernel"
include_dirs = [
    root / "include",
    root / "csrc",
]

sources = [
    "csrc/common_extension_hcu.cc",
    "csrc/attention/merge_attn_states.cu",
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
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
    "-w",
]

ext_modules = [
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

setup(
    name="sgl-kernel",
    version=_version(),
    packages=["sgl_kernel"],
    package_dir={"": "."},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
