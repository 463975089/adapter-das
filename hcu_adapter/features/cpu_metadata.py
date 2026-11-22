# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Private same-node CPU int32 consensus transport; no framework policy.

Bootstrap uses the caller's native collective until every rank has acknowledged
the same mmap. Afterwards failures abort the shared group; falling back mid-epoch
would make ranks enter different collectives. No GPU work is issued here.
"""
import ctypes
import hashlib
from importlib.metadata import distribution, PackageNotFoundError
import mmap
import os
from pathlib import Path
import platform
import threading
import uuid


def _load_library():
    try:
        package = distribution('sgl-kernel')
        paths = [package.locate_file(p) for p in package.files or ()
                 if p.name.startswith('cpu_metadata.') and p.suffix == '.so']
        if len(paths) != 1:
            return None
        lib = ctypes.CDLL(str(paths[0]))
        lib.hcu_meta_size_v1.restype = ctypes.c_uint64
        lib.hcu_meta_init_v1.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                       ctypes.c_int, ctypes.c_int]
        lib.hcu_meta_sum_v1.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int,
                                      ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int,
                                      ctypes.c_uint64]
        lib.hcu_meta_abort_v1.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        return lib
    except (PackageNotFoundError, OSError, AttributeError):
        return None


def _shared_identity(directory):
    """Conservative admission: one host, IPC namespace and shared mount."""
    identity = (Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                os.stat('/proc/self/ns/ipc').st_ino, os.stat(directory).st_dev,
                platform.machine(), platform.system())
    return hashlib.sha256(repr(identity).encode()).digest()


def _timeout_ns(timeout):
    seconds = 0 if timeout is None else timeout.total_seconds()
    if seconds < 0 or seconds >= (2**64 - 1) / 1e9:
        raise ValueError('CPU metadata timeout is outside the supported range')
    return int(seconds * 1e9)


class CPUInt32Transport:
    def __init__(self, lib, mem, rank, numel, timeout_ns):
        self._lib, self._mem = lib, mem
        self._ptr = ctypes.addressof(ctypes.c_char.from_buffer(mem))
        self._bytes = lib.hcu_meta_size_v1()
        self._rank, self._numel, self._timeout = rank, numel, timeout_ns
        self._epoch, self._pid = 0, os.getpid()
        self._lock = threading.Lock()

    def _check(self):
        if os.getpid() != self._pid:
            raise RuntimeError('CPU metadata transport inherited across fork; use spawn')
        if self._mem is None:
            raise RuntimeError('CPU metadata transport is closed')

    def all_reduce_(self, tensor):
        import torch
        self._check()  # Before acquiring a possibly inherited lock.
        with self._lock:
            self._check()
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                    or tensor.dtype != torch.int32 or not tensor.is_contiguous()
                    or tensor.numel() != self._numel):
                self._lib.hcu_meta_abort_v1(self._ptr, self._bytes)
                raise ValueError('CPU metadata SUM requires its contiguous CPU int32 packet')
            self._epoch += 1
            code = self._lib.hcu_meta_sum_v1(self._ptr, self._bytes, self._rank,
                self._epoch, tensor.data_ptr(), self._numel, self._timeout)
            if code:
                raise RuntimeError(f'CPU metadata shared SUM failed ({code}); restart workers')
            return tensor

    def close(self):
        if os.getpid() != self._pid:
            return  # A forked child must not abort its parent's active group.
        with self._lock:
            if self._mem is not None:
                self._lib.hcu_meta_abort_v1(self._ptr, self._bytes)
                self._mem.close()
                self._mem = None


def create_transport(group, numel, *, shm_dir=Path('/dev/shm')):
    """Collectively create a private transport, or uniformly retain native SUM.

    All ranks must call once in the same order, before using the returned object.
    Positive timeouts bound the whole shared call; None/zero means unbounded,
    matching native CPU wait semantics. Bootstrap native failures propagate.
    """
    import torch
    world, rank = group.world_size, group.rank_in_group
    if not 2 <= world <= 32:
        return None
    lib, identity, timeout = None, bytes(32), 0
    try:
        if platform.system() == 'Linux' and platform.machine() == 'x86_64':
            lib = _load_library()
            identity = _shared_identity(shm_dir)
            timeout = _timeout_ns(group.cpu_timeout)
        capable = (lib is not None and 1 <= numel <= 256
                   and callable(getattr(os, 'posix_fallocate', None)))
    except (OSError, ValueError):
        capable = False
    votes = torch.zeros(world, 35, dtype=torch.int32)
    votes[rank] = torch.tensor([int(capable), world, numel, *identity], dtype=torch.int32)
    group.cpu_all_reduce_(votes)
    if (not bool(torch.all(votes[:, 0] == 1))
            or not bool(torch.all(votes[:, 1] == world))
            or not bool(torch.all(votes[:, 2] == numel))
            or not bool(torch.all(votes[:, 3:] == votes[0, 3:]))):
        return None

    name, mem, path = '', None, None
    announcement = torch.zeros(33, dtype=torch.int32)
    try:
        if rank == 0:
            name = uuid.uuid4().hex
            path = Path(shm_dir) / ('hcu-meta-' + name)
            fd = None
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                os.ftruncate(fd, lib.hcu_meta_size_v1())
                # mmap/ftruncate alone can succeed on an exhausted tmpfs, then
                # SIGBUS in native initialization. Reserve all backing first.
                os.posix_fallocate(fd, 0, lib.hcu_meta_size_v1())
                mem = mmap.mmap(fd, lib.hcu_meta_size_v1())
                ptr = ctypes.addressof(ctypes.c_char.from_buffer(mem))
                if lib.hcu_meta_init_v1(ptr, lib.hcu_meta_size_v1(), world, numel):
                    raise OSError('CPU metadata shared header initialization failed')
                announcement[:] = torch.tensor([1, *name.encode()], dtype=torch.int32)
            except OSError:
                pass  # Root advertises failure through the native collective.
            finally:
                if fd is not None:
                    os.close(fd)
        group.cpu_all_reduce_(announcement)
        announced = int(announcement[0]) == 1
        if announced:
            name = bytes(announcement[1:].tolist()).decode('ascii')
            if len(name) != 32 or any(c not in '0123456789abcdef' for c in name):
                raise RuntimeError('CPU metadata bootstrap name is invalid')
        ok = int(announced)
        if announced and rank != 0:
            fd = None
            try:
                fd = os.open(Path(shm_dir) / ('hcu-meta-' + name), os.O_RDWR)
                mem = mmap.mmap(fd, lib.hcu_meta_size_v1())
            except (OSError, ValueError):
                ok = 0
            finally:
                if fd is not None:
                    os.close(fd)
        acknowledgment = torch.tensor([ok], dtype=torch.int32)
        group.cpu_all_reduce_(acknowledgment)
        # All peer mappings are complete. Agree cleanup too before any rank
        # owns a fast transport: a root-only unlink exception would split it.
        cleanup = torch.ones(1, dtype=torch.int32)
        if rank == 0 and path is not None:
            try:
                path.unlink(missing_ok=True)
                path = None
            except OSError:
                cleanup.zero_()
        group.cpu_all_reduce_(cleanup)
        if int(cleanup[0]) != world:
            raise RuntimeError('CPU metadata bootstrap could not unlink its shared file')
        if int(acknowledgment[0]) != world:
            return None
        result = CPUInt32Transport(lib, mem, rank, numel, timeout)
        mem = None  # Ownership transferred after every rank mapped successfully.
        return result
    finally:
        if mem is not None:
            mem.close()
        if rank == 0 and path is not None:
            path.unlink(missing_ok=True)
