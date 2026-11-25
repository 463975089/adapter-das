# SPDX-License-Identifier: Apache-2.0
"""Cache identities follow ABI bytes, independent of filesystem timestamps."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hcu_adapter.kernels import fp8_scaled_mm_cpp as kernel


@pytest.fixture
def inputs(tmp_path):
    paths = [tmp_path / name for name in ("kernel.cpp", "header.h", "library.so")]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index]) * 129)
    return paths


def test_library_timestamps_do_not_invalidate_cache(inputs):
    before = kernel._library_signature(*inputs)
    stat = inputs[-1].stat()
    os.utime(inputs[-1], ns=(stat.st_atime_ns + 1000000000, stat.st_mtime_ns + 1000000000))
    assert kernel._library_signature(*inputs) == before


@pytest.mark.parametrize("changed", [0, 1, 2])
def test_changed_source_header_or_vendor_bytes_invalidate_cache(inputs, changed):
    before = kernel._library_signature(*inputs)
    stat = inputs[changed].stat()
    data = inputs[changed].read_bytes()
    inputs[changed].write_bytes(bytes([data[0] ^ 1]) + data[1:])
    os.utime(inputs[changed], ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert kernel._library_signature(*inputs) != before


def test_library_reuses_compiled_artifact_after_metadata_only_change(monkeypatch, tmp_path):
    # Exercise the loader and its on-disk cache without invoking a compiler/GPU.
    source = tmp_path / "kernel.cpp"
    source.write_bytes(b"source")
    vendor = tmp_path / "dtk"
    header = vendor / "include/hipblaslt/hipblaslt.h"
    library = vendor / "lib/libhipblaslt.so"
    for path in (header, library):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"vendor")

    def local_path(value):
        return vendor if value == "/opt/dtk" else Path(value)

    local_path.home = lambda: tmp_path / "home"
    monkeypatch.setattr(kernel, "Path", local_path)
    monkeypatch.setattr(kernel, "__file__", str(source.with_suffix(".py")))
    compiled = []

    def compile_library(command, **kwargs):
        compiled.append(command)
        Path(command[-1]).write_bytes(b"compiled")
        return SimpleNamespace(returncode=0, stdout=b"")

    monkeypatch.setattr(kernel.subprocess, "run", compile_library)
    monkeypatch.setattr(kernel.ctypes, "CDLL", lambda path:
                        SimpleNamespace(hcu_fp8_lt_run=SimpleNamespace()))
    kernel._library.cache_clear()
    try:
        assert kernel._library() is not None
        stat = library.stat()
        os.utime(library, ns=(stat.st_atime_ns + 1000000000, stat.st_mtime_ns + 1000000000))
        kernel._library.cache_clear()  # Simulate a fresh worker.
        assert kernel._library() is not None
        assert len(compiled) == 1
        library.write_bytes(b"new vendor")
        kernel._library.cache_clear()
        assert kernel._library() is not None
        assert len(compiled) == 2
    finally:
        kernel._library.cache_clear()
