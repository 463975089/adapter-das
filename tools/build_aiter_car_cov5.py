#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Build the pinned image's Aiter custom all-reduce through its public JIT API."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import sys


VERSION = "0.1.8+dtk2610.torch2110.2611031115.ga1a2bc"
SOURCE_DIGEST = "990a3f5480e42e2cee5e2c57b9b797aca4ed534712d34802c1b28220c0f4d62d"
ORIGINAL_BINARY = "5fd195371e731dc52998617b75e335f05686efc226d2d66c18c3bbd7bcbc5cc5"
MODULE = "module_custom_all_reduce.so"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(package):
    files = [package / "jit/core.py", package / "jit/optCompilerConfig.json",
             package / "ops/custom_all_reduce.py"]
    files.extend(path for path in (package.parent / "aiter_meta/csrc").rglob("*")
                 if path.is_file() and path.suffix in {".h", ".hpp", ".cuh", ".cu", ".cpp"})
    return {str(path.relative_to(package.parent)): sha256(path) for path in sorted(files)}


def source_digest(manifest):
    data = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def code_object_abis(binary):
    data = binary.read_bytes()
    result = []
    start = 0
    while True:
        offset = data.find(b"\x7fELF", start)
        if offset < 0:
            return result
        start = offset + 4
        if (offset + 64 <= len(data) and data[offset + 4:offset + 6] == b"\x02\x01"
                and struct.unpack_from("<H", data, offset + 18)[0] == 224):
            result.append(data[offset + 8])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if "aiter" in sys.modules:
        raise RuntimeError("Use a fresh HCU_ADAPTER_HOST=none process; Aiter is already loaded")
    distribution = importlib.metadata.distribution("aiter")
    if distribution.version != VERSION:
        raise RuntimeError(f"Expected Aiter {VERSION}, found {distribution.version}")
    package = Path(distribution.locate_file("aiter")).resolve()
    original = package / "jit" / MODULE
    manifest = sources(package)
    if source_digest(manifest) != SOURCE_DIGEST or sha256(original) != ORIGINAL_BINARY:
        raise RuntimeError("Installed Aiter source/binary differs from the pinned image")
    output = args.output_dir.expanduser().resolve()
    stage = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(output)
    jit = stage / "jit"
    stage.mkdir(parents=True, exist_ok=False)
    jit.mkdir()
    # Reuse unrelated image modules; only CAR is absent and therefore rebuilt.
    for binary in (package / "jit").glob("*.so"):
        if binary.name != MODULE:
            (jit / binary.name).symlink_to(binary)
    build_env = dict(AITER_JIT_DIR=str(jit), AITER_CODE_OBJECT_VERSION="5",
                     CK_CODE_OBJECT_VERSION="5", GPU_ARCHS="gfx936",
                     MAX_JOBS=str(args.jobs), AITER_REBUILD="0")
    child = (
        "import os; from pathlib import Path; import aiter; "
        "from aiter.jit.core import get_module; aiter.meta_size(); "
        "assert Path(get_module('module_custom_all_reduce').__file__).resolve().parent "
        "== Path(os.environ['AITER_JIT_DIR']).resolve()"
    )
    subprocess.run([sys.executable, "-c", child], check=True,
                   env=dict(os.environ, HCU_ADAPTER_HOST="none", **build_env))
    binary = jit / MODULE
    if binary.is_symlink() or code_object_abis(binary) != [3]:
        raise RuntimeError("Isolated CAR binary must contain gfx936 COV5 device code")
    if sources(package) != manifest or sha256(original) != ORIGINAL_BINARY:
        raise RuntimeError("Installed Aiter changed during the build")
    report = {
        "aiter_version": VERSION, "source_digest": SOURCE_DIGEST,
        "source_files": manifest, "original_binary_sha256": ORIGINAL_BINARY,
        "binary": "jit/" + MODULE, "binary_sha256": sha256(binary),
        "build_env": dict(build_env, AITER_JIT_DIR="<output-dir>/jit"),
        "scope": "Same image sources, COV5/gfx936 CAR only; numerical/graph/model validation required",
    }
    (stage / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    if output.exists():
        raise FileExistsError(output)
    stage.rename(output)
    print(json.dumps({"output": str(output), "binary_sha256": report["binary_sha256"]}))


if __name__ == "__main__":
    main()
