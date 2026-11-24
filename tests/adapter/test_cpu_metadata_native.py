# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the shipped C ABI, independent of Torch/HIP imports."""
import ctypes
import mmap
import multiprocessing as mp
import os
from pathlib import Path
import random
import subprocess
import threading
import time

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "csrc/distributed/cpu_metadata.cc"


def library(path):
    lib = ctypes.CDLL(str(path))  # CDLL calls must release the GIL while waiting.
    lib.hcu_meta_size_v1.restype = ctypes.c_uint64
    lib.hcu_meta_init_v1.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                   ctypes.c_int, ctypes.c_int]
    lib.hcu_meta_sum_v1.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int,
                                  ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int,
                                  ctypes.c_uint64]
    lib.hcu_meta_abort_v1.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    return lib


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    assert SOURCE.is_file(), "Adapter CPU metadata transport is not implemented"
    path = tmp_path_factory.mktemp("cpu-metadata") / "metadata.so"
    subprocess.run(["g++", "-std=c++17", "-O3", "-fPIC", "-shared", "-Wall",
                    "-Wextra", "-Werror", str(SOURCE), "-o", str(path)], check=True)
    return str(path)


def mapping(path, lib):
    fd = os.open(path, os.O_RDWR)
    try:
        mem = mmap.mmap(fd, lib.hcu_meta_size_v1())
    finally:
        os.close(fd)
    return mem, ctypes.addressof(ctypes.c_char.from_buffer(mem))


def init_file(path, lib, world, size):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.ftruncate(fd, lib.hcu_meta_size_v1())
    os.close(fd)
    mem, ptr = mapping(path, lib)
    assert lib.hcu_meta_init_v1(ptr, lib.hcu_meta_size_v1(), world, size) == 0
    return mem, ptr


def payload(rank, epoch, size):
    rng = random.Random(rank * 104729 + epoch)
    return [rng.randrange(-2**31, 2**31) for _ in range(size)]


def run_rank(binary, path, rank, world, size):
    lib = library(binary)
    mem, ptr = mapping(path, lib)
    try:
        for epoch in range(1, 121):
            data = (ctypes.c_int32 * size)(*payload(rank, epoch, size))
            if (rank + epoch) % 7 == 0:
                time.sleep((rank % 3) / 1000)
            code = lib.hcu_meta_sum_v1(ptr, lib.hcu_meta_size_v1(), rank,
                                      epoch, data, size, 10_000_000_000)
            assert code == 0, (rank, epoch, code)
            expected = [ctypes.c_int32(sum(column)).value for column in
                        zip(*(payload(r, epoch, size) for r in range(world)))]
            assert list(data) == expected, (rank, epoch)
    finally:
        mem.close()


@pytest.mark.parametrize("world", [3, 5, 16])
def test_full_int32_sum_pipelined_epochs_with_skew(binary, tmp_path, world):
    lib = library(binary)
    path = str(tmp_path / "shared")
    mem, _ = init_file(path, lib, world, 7 * world)
    ctx = mp.get_context("spawn")
    workers = [ctx.Process(target=run_rank, args=(binary, path, r, world, 7 * world))
               for r in range(world)]
    try:
        for p in workers:
            p.start()
        for p in workers:
            p.join(90)
        assert all(p.exitcode == 0 for p in workers), [p.exitcode for p in workers]
    finally:
        for p in workers:
            if p.is_alive():
                p.terminate()
                p.join(10)
        mem.close()


@pytest.mark.parametrize("bad_epoch", [1, 3, 0, 2**64 - 1])
def test_duplicate_skipped_or_invalid_epoch_poison_group(binary, tmp_path, bad_epoch):
    lib = library(binary)
    mem, ptr = init_file(str(tmp_path / "shared"), lib, 1, 1)
    data = (ctypes.c_int32 * 1)(-7)
    try:
        assert lib.hcu_meta_sum_v1(ptr, lib.hcu_meta_size_v1(), 0, 1, data, 1, 0) == 0
        assert lib.hcu_meta_sum_v1(ptr, lib.hcu_meta_size_v1(), 0, bad_epoch,
                                   data, 1, 100_000_000) < 0
        assert lib.hcu_meta_sum_v1(ptr, lib.hcu_meta_size_v1(), 0, 2,
                                   data, 1, 100_000_000) == -3
    finally:
        mem.close()


def test_missing_peer_timeout_releases_gil_and_propagates_abort(binary, tmp_path):
    lib = library(binary)
    mem, ptr = init_file(str(tmp_path / "shared"), lib, 2, 14)
    data = (ctypes.c_int32 * 14)()
    progress = threading.Event()
    observer = threading.Thread(target=lambda: (time.sleep(.005), progress.set()))
    observer.start()
    try:
        start = time.monotonic()
        assert lib.hcu_meta_sum_v1(ptr, lib.hcu_meta_size_v1(), 0, 1,
                                   data, 14, 30_000_000) == -2
        assert time.monotonic() - start < 1
        assert progress.is_set()
        assert lib.hcu_meta_sum_v1(ptr, lib.hcu_meta_size_v1(), 1, 1,
                                   data, 14, 30_000_000) == -3
    finally:
        observer.join()
        mem.close()


def test_unbounded_timeout_waits_for_delayed_peer(binary, tmp_path):
    lib = library(binary)
    mem, ptr = init_file(str(tmp_path / "shared"), lib, 2, 1)
    a, b = (ctypes.c_int32 * 1)(10), (ctypes.c_int32 * 1)(20)
    results = []
    def delayed():
        time.sleep(.05)
        results.append(lib.hcu_meta_sum_v1(ptr, lib.hcu_meta_size_v1(), 1, 1, b, 1, 0))
    peer = threading.Thread(target=delayed)
    peer.start()
    try:
        assert lib.hcu_meta_sum_v1(ptr, lib.hcu_meta_size_v1(), 0, 1, a, 1, 0) == 0
        peer.join(5)
        assert results == [0] and a[0] == b[0] == 30
    finally:
        peer.join(5)
        mem.close()


def test_header_shape_and_alignment_guards(binary, tmp_path):
    lib = library(binary)
    mem, ptr = init_file(str(tmp_path / "shared"), lib, 1, 7)
    data = (ctypes.c_int32 * 7)()
    try:
        assert lib.hcu_meta_init_v1(ptr + 1, lib.hcu_meta_size_v1(), 1, 7) == -1
        assert lib.hcu_meta_init_v1(ptr, lib.hcu_meta_size_v1(), 33, 7) == -1
        assert lib.hcu_meta_sum_v1(ptr, 1, 0, 1, data, 7, 1) == -1
        assert lib.hcu_meta_sum_v1(ptr, lib.hcu_meta_size_v1(), 0, 1, data, 6, 1) == -1
    finally:
        mem.close()
