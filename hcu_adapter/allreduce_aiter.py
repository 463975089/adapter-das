# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Read-only provenance for the public AITER AR ABI; no framework imports.

Caller supplies already-admitted public package/provider objects. File hashes
require immutable artifacts; they do not attest overwritten loaded mappings,
transitive HIP libraries, kernel launches or numerical results.
"""
import hashlib
from pathlib import Path

from .allreduce import AllReduceCompatibilityError

PYTHON_FILES = (
    "__init__.py", "ops/__init__.py", "ops/custom_all_reduce.py", "ops/enum.py",
    "dist/__init__.py", "dist/parallel_state.py",
    "dist/device_communicators/custom_all_reduce.py",
    "dist/device_communicators/topology.py", "utility/__init__.py",
    "utility/dtypes.py", "utility/aiter_types.py", "jit/__init__.py", "jit/core.py",
    "jit/utils/__init__.py", "jit/utils/torch_guard.py", "jit/utils/chip_info.py",
    "jit/utils/cpp_extension.py", "jit/utils/file_baton.py",
)
OPTIONAL_NAMESPACE_FILES = ("dist/device_communicators/__init__.py",)
DTYPE_RESOURCES = ("csrc/include/aiter_enum.h", "aiter_meta/csrc/include/aiter_enum.h")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provider_identity(package, provider, modules):
    root = Path(package.__file__).resolve().parent
    if Path(provider.__file__).resolve() != root / "dist/device_communicators/custom_all_reduce.py":
        raise AllReduceCompatibilityError("AR provider origin differs from public package")
    records = {name: digest(root / name) for name in PYTHON_FILES}
    for name in OPTIONAL_NAMESPACE_FILES:
        records[name] = digest(root / name) if (root / name).is_file() else "namespace"
    dtype_root = (root / "utility/aiter_types.py").resolve().parents[2]
    resource = next((name for name in DTYPE_RESOURCES if (dtype_root / name).is_file()), None)
    if resource is None:
        raise AllReduceCompatibilityError("AR provider dtype ABI resource unavailable")
    natives = set()
    for name in ("aiter.jit.module_custom_all_reduce", "module_custom_all_reduce"):
        filename = getattr(modules.get(name), "__file__", None)
        if filename and Path(filename).suffix == ".so":
            natives.add(digest(filename))
    if not natives:
        raise AllReduceCompatibilityError("public provider has not loaded native custom-AR module")
    if len(natives) != 1:
        raise AllReduceCompatibilityError("ambiguous loaded custom-AR native providers")
    return {"python": records, "resources": {resource: digest(dtype_root / resource)},
            "native": {"module_custom_all_reduce": natives.pop()},
            "dtypes": {"fp8": str(provider.fp8),
                       "FP8_DTYPES": sorted(map(str, provider.FP8_DTYPES))}}
