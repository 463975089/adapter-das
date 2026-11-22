# SPDX-License-Identifier: Apache-2.0
"""Real multiprocess bootstrap, ownership and full-packet transport checks."""
import ctypes
from datetime import timedelta
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path

import pytest

from test_cpu_metadata_native import binary, library

SOURCE = Path(__file__).resolve().parents[2] / "hcu_adapter/features/cpu_metadata.py"


def module():
    assert SOURCE.is_file(), "Adapter metadata bootstrap is not implemented"
    spec = importlib.util.spec_from_file_location("cpu_metadata_under_test", SOURCE)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def worker(rank, world, binary, root, mode):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(Path(root) / "pg"),
                            rank=rank, world_size=world, timeout=timedelta(seconds=30))
    class Group:
        world_size = world
        rank_in_group = rank
        cpu_timeout = timedelta(seconds=5)
        def cpu_all_reduce_(self, x):
            dist.all_reduce(x)
            return x
    owned = module()
    owned._load_library = lambda: library(binary)
    shm = Path(root) / "shm"
    if mode == "missing-binary" and rank == world - 1:
        owned._load_library = lambda: None
    if mode == "different-mount" and rank == world - 1:
        original = owned._shared_identity
        owned._shared_identity = lambda p: b"X" * len(original(p))
    if mode == "unavailable-file" and rank == world - 1:
        shm = Path(root) / "missing-shm"
    if ((mode == 'map-failure' and rank == world - 1)
            or (mode == 'root-failure' and rank == 0)):
        original_open = owned.os.open
        def failing_open(path, *args, **kwargs):
            if 'hcu-meta-' in str(path):
                raise OSError('injected allocation/open failure')
            return original_open(path, *args, **kwargs)
        owned.os.open = failing_open
    if mode == 'allocation-enospc' and rank == 0:
        def no_space(*args):
            raise OSError(28, 'injected tmpfs ENOSPC')
        owned.os.posix_fallocate = no_space
    if mode == 'unlink-failure' and rank == 0:
        original_unlink = owned.Path.unlink
        attempts = [0]
        def unlink_once(path, *args, **kwargs):
            if 'hcu-meta-' in str(path) and attempts[0] == 0:
                attempts[0] += 1
                raise OSError('injected unlink failure')
            return original_unlink(path, *args, **kwargs)
        owned.Path.unlink = unlink_once
    transport = None
    if mode == 'unlink-failure':
        with pytest.raises(RuntimeError, match='unlink'):
            owned.create_transport(Group(), 7 * world, shm_dir=shm)
    else:
        transport = owned.create_transport(Group(), 7 * world, shm_dir=shm)
    dist.barrier()
    try:
        if mode != "normal":
            assert transport is None
        else:
            assert transport is not None
            assert not list(shm.iterdir()), "bootstrap must unlink after all mappings exist"
            for step in range(1, 101):
                x = (torch.arange(7 * world, dtype=torch.int64)
                     + 2147483640 + rank + step).to(torch.int32).reshape(7, world)
                want = torch.stack([(torch.arange(7 * world, dtype=torch.int64)
                        + 2147483640 + r + step).to(torch.int32)
                        for r in range(world)]).sum(0, dtype=torch.int32).reshape(7, world)
                ptr = x.data_ptr()
                assert transport.all_reduce_(x) is x
                assert x.data_ptr() == ptr and torch.equal(x, want)
            transport._pid += 1
            with pytest.raises(RuntimeError, match='fork'):
                transport.all_reduce_(x)
            transport._pid -= 1
            dist.barrier()
            for x in [torch.zeros(7 * world, dtype=torch.float32),
                      torch.zeros(7 * world + 1, dtype=torch.int32),
                      torch.zeros(world, 7, dtype=torch.int32).t()]:
                with pytest.raises(ValueError):
                    transport.all_reduce_(x)
    finally:
        if transport is not None:
            transport.close()
            transport.close()
            with pytest.raises(RuntimeError, match="closed"):
                transport.all_reduce_(torch.zeros(7, world, dtype=torch.int32))
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["normal", "missing-binary", "different-mount", "unavailable-file",
                                  "map-failure", "root-failure", "allocation-enospc", "unlink-failure"])
def test_collective_bootstrap_and_uniform_fallback(binary, tmp_path, mode):
    module()  # RED checks the product module before starting processes.
    (tmp_path / "shm").mkdir()
    ctx = mp.get_context("spawn")
    children = [ctx.Process(target=worker, args=(r, 3, binary, str(tmp_path), mode))
                for r in range(3)]
    try:
        for p in children:
            p.start()
        for p in children:
            p.join(90)
        assert all(p.exitcode == 0 for p in children), [p.exitcode for p in children]
        assert not list((tmp_path / "shm").iterdir())
    finally:
        for p in children:
            if p.is_alive():
                p.terminate()
                p.join(10)


def test_timeout_conversion(binary, tmp_path):
    owned = module()
    assert owned._timeout_ns(None) == owned._timeout_ns(timedelta(0)) == 0
    assert owned._timeout_ns(timedelta(seconds=.1)) == 100_000_000
    with pytest.raises(ValueError):
        owned._timeout_ns(timedelta(seconds=-1))
