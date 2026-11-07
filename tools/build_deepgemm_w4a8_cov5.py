#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Build the pinned native HIPC1002 kernel in an isolated COV5 extension."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig
import tempfile


MODULE = "hcu_deepgemm_w4a8_cov5"
SOURCE_COMMIT = "b061ec70272a001fca622ea814d3ee163e24bb05"
SOURCE_HASHES = {
    "csrc/arch/gfx936_gfx938_shared/m_grouped_w4a8_gemm_nt_contiguous_hipc.cu":
        "e82dc9bd4fb71ee5009f83871c779dae2ac145c6ee801dcba28daedba7158da8",
    "csrc/include/hip_common.h":
        "69b4794d804fc6fe270abaacac327391c83a919269c80de87e3f4d583ea8d5c2",
    "csrc/include/hipc_w4a8/w4a8_intrinsics.h":
        "e0fa3081e20513eb46ef117d9f8ab76ea55b9ba9e979b102c286d64df43ff1c6",
    "csrc/include/m_grouped_w4a8_gemm_nt_contiguous.h":
        "51d6d7c0204bbbf3acc76b573a296feb1de4f413b85fe102631b8107bdcbc570",
}


def check_sources(source):
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Pinned DeepGEMM source changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("--output must be a new directory outside the source tree")
    check_sources(source)
    import torch
    from torch.utils.cpp_extension import CUDAExtension, IS_HIP_EXTENSION

    if not IS_HIP_EXTENSION or torch.__version__ != "2.11.0":
        raise RuntimeError("This recipe requires the fixed image's HIP Torch 2.11.0")
    compiler = shutil.which("hipcc")
    if compiler is None:
        raise FileNotFoundError("hipcc")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.build-", dir=output.parent))
    original = source / next(iter(SOURCE_HASHES))
    text = original.read_text()
    include = '#include "m_grouped_w4a8_gemm_nt_contiguous_wasp/bm128_bn512_bk64.inc.cuh"\n'
    branch = (
        "  if (mode == 1006) {\n"
        "    return m_grouped_w4a8_gemm_nt_contiguous_wasp_bm128_bn512_bk64::run(\n"
        "        input, b_qweight, output, a_scale, b_scale, m_indices);\n"
        "  }\n"
    )
    if text.count(include) != 1 or text.count(branch) != 1:
        raise RuntimeError("Pinned WASP compile boundary changed")
    # Isolate the existing 1002 body. gfx936 compilation of this commit's
    # WASP1006 branch fails register allocation; never silently select it.
    text = text.replace(include, "").replace(branch, "")
    text = text.replace("mode == 1002 || mode == 1006", "mode == 1002")
    generated = stage / "contiguous1002.cu"
    generated.write_text(text)
    binding = stage / "binding.cpp"
    binding.write_text(
        '#include <torch/extension.h>\n'
        '#include "m_grouped_w4a8_gemm_nt_contiguous.h"\n'
        'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'
        '  m.def("run", &deepgemm::m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc);\n'
        f'  m.attr("SOURCE_COMMIT") = "{SOURCE_COMMIT}";\n'
        '  m.attr("CODE_OBJECT_VERSION") = 5;\n'
        '  m.attr("SUPPORTED_MODES") = pybind11::make_tuple(1002);\n'
        '  m.attr("ARCHITECTURE") = "gfx936";\n'
        '}\n'
    )
    previous_cwd = Path.cwd()
    try:
        os.chdir(stage)
        extension = CUDAExtension(name=MODULE, sources=[str(generated)])
        torch_root = Path(torch.__file__).parent
        binary = stage / (MODULE + sysconfig.get_config_var("EXT_SUFFIX"))
        command = [
            compiler, "-mcode-object-version=5", "--offload-arch=gfx936",
            "-shared", "-fPIC", "-O3", "-std=c++17", "-fno-gpu-rdc",
            "--gpu-max-threads-per-block=1024", "-mllvm",
            "-support-768-vgprs=true", "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
            "-DUSE_ROCM=1", "-D__HIP_PLATFORM_HCC__=1", "-DTORCH_VERSION_OVER_110",
            f"-DTORCH_EXTENSION_NAME={MODULE}", "-w",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            "-I" + str(source / "csrc/include"),
            "-I" + str(source / "csrc/arch/gfx936_gfx938_shared"),
            "-I" + str(torch_root / "include"),
            "-I" + str(torch_root / "include/torch/csrc/api/include"),
            "-I" + sysconfig.get_paths()["include"],
            str(binding), extension.sources[0], "-L" + str(torch_root / "lib"),
            "-Wl,-rpath," + str(torch_root / "lib"), "-ltorch", "-ltorch_cpu",
            "-ltorch_hip", "-ltorch_python", "-lc10", "-lc10_hip",
            "-o", str(binary),
        ]
        (stage / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        subprocess.run(command, check=True)
        check_sources(source)
        report = {
            "source_commit": SOURCE_COMMIT,
            "source_hashes": SOURCE_HASHES,
            "generated_sha256": hashlib.sha256(generated.read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "torch": torch.__version__, "code_object_version": 5,
            "architecture": "gfx936", "mode": 1002,
            "scope": "Native1002 bodies unchanged; WASP1006 excluded and rejected",
        }
        (stage / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        os.chdir(previous_cwd)
    stage.rename(output)
    print(json.dumps({"output": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
