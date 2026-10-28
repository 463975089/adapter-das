#!/usr/bin/env python3
"""Build the pinned upstream C++ extension in an isolated HIP staging tree."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


UPSTREAM_PIN = "774ce244b87cc0eb8d4b94ab9f5de2ab22049f01"
COMMON_SOURCES = (
    "pybind.cpp", "engine/pybind.cpp", "engine/pybind_sequence.cpp",
    "engine/pybind_scheduler.cpp", "engine/pybind_output.cpp", "engine/zos.cpp",
    "engine/input_metadata/block_load_plan.cpp",
    "engine/input_metadata/prepare_tensor_input_base.cpp",
    "engine/input_metadata/prepare_tensor_input_v4.cpp", "engine/prepare_sample.cpp",
    "graph_timer/pybind.cpp", "cuda_utils.cpp", "graph_timer/graph_timer_cuda.cpp",
)


def git(root, *arguments):
    return subprocess.check_output(
        ["git", "-c", "safe.directory=" + str(root), "-C", str(root), *arguments],
        text=True, timeout=30,
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--toolchain", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    output = args.output.resolve()
    workdir = args.workdir.resolve()
    toolchain = args.toolchain.resolve()
    if git(upstream, "rev-parse", "HEAD") != UPSTREAM_PIN:
        raise RuntimeError("Upstream pin differs; reconcile the build recipe first")
    initial_status = git(upstream, "status", "--porcelain")
    if initial_status:
        raise RuntimeError("Upstream source tree must be clean before staging")
    if output.is_relative_to(upstream) or workdir.is_relative_to(upstream):
        raise ValueError("Build and output directories must be outside upstream")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    compiler = toolchain / "usr/bin/g++-13"
    if not compiler.is_file():
        raise FileNotFoundError(compiler)
    runtime = toolchain / "usr/lib/x86_64-linux-gnu"
    environment = dict(os.environ)
    environment.update({"CC": str(toolchain / "usr/bin/gcc-13"),
                        "CXX": str(compiler), "MAX_JOBS": str(args.jobs)})
    environment["LD_LIBRARY_PATH"] = str(runtime) + ":" + environment.get("LD_LIBRARY_PATH", "")
    os.environ.update(environment)
    subprocess.run([str(compiler), "--version"], check=True, timeout=15)
    workdir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="hcu-csrc-", dir=workdir))
    shutil.copytree(upstream / "csrc", stage / "csrc")

    import setuptools
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if not torch.version.hip or not torch._C._GLIBCXX_USE_CXX11_ABI:
        raise RuntimeError("The recipe requires DTK HIP torch with CXX11 ABI")
    previous_cwd = Path.cwd()
    try:
        os.chdir(stage)
        extension = CUDAExtension(
            name="deepinfer._C",
            sources=["csrc/" + source for source in COMMON_SOURCES],
            include_dirs=[str(stage / "csrc")],
            libraries=["pthread"],
            runtime_library_dirs=[str(runtime)],
            # These HIP APIs exist in DTK2610 but are absent from its hipify
            # token table. Compile-time aliases preserve their real semantics.
            define_macros=[("cudaEventRecordWithFlags", "hipEventRecordWithFlags"),
                           ("cudaEventRecordExternal", "hipEventRecordExternal"),
                           ("cudaEventRecordDefault", "hipEventRecordDefault")],
            extra_compile_args={"cxx": ["-O2", "-g1", "-std=c++20",
                                        "-Wno-deprecated-declarations",
                                        "-D_GLIBCXX_USE_CXX11_ABI=1",
                                        "-D__HIP_PLATFORM_AMD__=1"]},
        )
        setuptools.setup(
            name="deepinfer-hcu-native", version="0.1.0",
            script_args=["build_ext", "--build-lib", str(output),
                         "--build-temp", str(stage / "objects")],
            ext_modules=[extension], cmdclass={"build_ext": BuildExtension},
        )
        manifest = {"upstream_pin": UPSTREAM_PIN, "torch": torch.__version__,
                    "hip": torch.version.hip, "compiler": str(compiler),
                    "runtime_library_path": str(runtime), "stage": str(stage),
                    "extensions": [str(p) for p in (output / "deepinfer").glob("_C*.so")]}
        (output / "hcu-build.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest), flush=True)
    finally:
        os.chdir(previous_cwd)
        if git(upstream, "status", "--porcelain") != initial_status:
            raise RuntimeError("Upstream source tree changed during the build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
