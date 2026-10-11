"""Custom all-reduce for HCU (single-node TP, CUDA-graph safe).

sglang expects a ``sgl_kernel.allreduce`` module (the vllm-style custom AR,
normally a compiled extension on CUDA/ROCm — aiter's custom AR occupies the
same slot on ROCm). This adapter provides it on top of HIP IPC + a small C++
device extension (``libhcu_ar.so``, built from ``ar_ext.cu``).

The protocol is dictated by measured HCU/gfx938 visibility rules (probed
2026-10-11, GPU6/7): cross-GPU WRITES to VRAM from a running kernel are
never reliably readable by a concurrent remote kernel (stores reach the
remote-fetchable point only after kernel end; fences and system-scope
atomic flushes do NOT close this window), and cross-GPU spins on
device-memory flags are impossible (reader-side line caching until kernel
boundary). The one fabric with real mid-kernel cross-GPU coherence is
HOST-MAPPED shared memory (/dev/shm + hipHostRegister): GPU accesses are
uncached PCIe transactions — every poll fetches fresh, writes land in host
DRAM, and posted writes from the same source are hardware-ordered.

So BOTH the flags AND the payload live in host shm. Per round r (1-based,
derived on-device from a u64 sequence word): stage my input into my host
row -> __threadfence_system (host stores complete) -> publish r -> spin
peers' publish -> pull-read peers' host rows -> reduce -> fence -> publish
done -> commit seq. Slot reuse (r-S) waits on the peers' done flags. No
host code runs inside the kernel, so CUDA-graph capture/replay of any
length is correct with zero host involvement; all buffers are static, so
``register_graph_buffers`` is a no-op.

Host shm block (named by a hash of the member IPC handles — identical on
every rank, no extra collective needed): ``[PUB ws x S i64][DONE ws x S
i64][pad to 0x1000][payload rows ws x S x max_size]``. Device buffer (own
allocation, local sync only): ``0x0 u64 seq | 0x8 u32 a1 | 0xc u32 a2 |
0x10 u64 diag``.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import mmap
import os

import torch

logger = logging.getLogger(__name__)

_HIP_LIB_CANDIDATES = (
    "/opt/dtk/hip/lib/libamdhip64.so",
    "libamdhip64.so",
)
_SO_CANDIDATES = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libhcu_ar.so"),
)

HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1

# device buffer layout (bytes) — must match ar_ext.cu
SYNC_OFF = 0x000
DATA_OFF = 0x1000

_MAX_WS = 8
_DEFAULT_BUDGET = 100_000_000  # ~2.5 min worst-case PCIe polls, then degrade


class IpcMemHandle(ctypes.Structure):
    _fields_ = [("x", ctypes.c_char * 64)]


def _load_hip():
    for path in _HIP_LIB_CANDIDATES:
        try:
            lib = ctypes.CDLL(path)
            lib.hipMalloc.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t
            ]
            lib.hipFree.argtypes = [ctypes.c_void_p]
            lib.hipIpcGetMemHandle.argtypes = [
                ctypes.POINTER(IpcMemHandle), ctypes.c_void_p
            ]
            lib.hipIpcOpenMemHandle.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), IpcMemHandle, ctypes.c_uint
            ]
            lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
            lib.hipMemset.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
            ]
            lib.hipMemcpy.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
            ]
            return lib
        except OSError:
            continue
    raise OSError("libamdhip64.so not found for custom AR")


_HIP = _load_hip()


def _load_ext():
    paths = [os.environ.get("HCU_TRITON_AR_SO", "")] + list(_SO_CANDIDATES)
    for path in paths:
        if not path:
            continue
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    raise OSError(
        "libhcu_ar.so not found (build ar_ext.cu with: "
        "hipcc -O2 -fPIC -shared ar_ext.cu -o libhcu_ar.so)"
    )


def _slots_for_ws(ws: int) -> int:
    env = os.environ.get("HCU_TRITON_AR_SLOTS", "")
    if env:
        return int(env)
    return 8 if ws <= 2 else (4 if ws <= 4 else 2)


def _budget() -> int:
    return int(os.environ.get("HCU_TRITON_AR_BUDGET", _DEFAULT_BUDGET))


class TritonARState:
    """Per-communicator state: own allocation, peer IPC mappings, host flags."""

    def __init__(self, handles, offsets, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.disposed = False
        self._mm = None
        self._fd = None

        if world_size > _MAX_WS:
            raise RuntimeError(f"custom AR: world {world_size} > {_MAX_WS}")

        key = bytes(handles[rank])
        alloc = _ACTIVE_ALLOCATIONS.get(key)
        if alloc is None:
            raise RuntimeError("custom AR: own IPC handle not found (allocate first)")
        base, max_size, ws_alloc, s_alloc = alloc
        if world_size > ws_alloc:
            raise RuntimeError(
                f"custom AR: world {world_size} exceeds allocated {ws_alloc}"
            )
        self._own_base = base
        self._max_size = max_size
        self.slots = s_alloc

        # host shm block: [flags 0x1000][payload rows ws x S x max_size],
        # named by the handle set (identical on all ranks — no extra
        # collective needed). Flags AND payload both live here: VRAM stores
        # are NOT ordered against the flag publish for concurrent remote
        # readers on HCU, PCIe posted writes are (see ar_ext.cu header).
        ext = _EXT()
        tag = hashlib.md5(b"".join(sorted(bytes(h) for h in handles))).hexdigest()
        self._shm_path = f"/dev/shm/hcu_ar_{tag}"
        row = self.slots * max_size
        shm_bytes = 0x1000 + world_size * row
        fd = os.open(self._shm_path, os.O_RDWR | os.O_CREAT)
        os.ftruncate(fd, shm_bytes)
        mm = mmap.mmap(fd, shm_bytes)  # MAP_SHARED
        mm[:] = b"\x00" * shm_bytes    # idempotent; handles stale tmpfs files
        dev_ptr = ctypes.c_void_p()
        rc = ext.hcu_register_host(
            ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm))),
            shm_bytes,
            ctypes.byref(dev_ptr),
        )
        if rc != 0:
            raise RuntimeError(f"custom AR: hipHostRegister rc={rc}")
        self._mm = mm
        self._fd = fd
        self._flags = dev_ptr.value
        self._my_data = dev_ptr.value + 0x1000 + rank * row
        self._peer_data = [
            dev_ptr.value + 0x1000 + j * row
            for j in range(world_size)
            if j != rank
        ]

        # nobody may publish before every rank finished zeroing the block
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        self._sync = base + SYNC_OFF
        self._budget = _budget()

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor | None = None):
        if self.disposed:
            raise RuntimeError("custom AR: state disposed")
        code = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}.get(
            inp.dtype
        )
        if code is None:
            raise NotImplementedError(
                f"custom AR: unsupported dtype {inp.dtype} (nccl fallback)"
            )
        n = inp.numel()
        if n * inp.element_size() > self._max_size:
            raise ValueError(
                f"custom AR: {n} elems exceed slot capacity {self._max_size}"
            )
        if not inp.is_contiguous():
            inp = inp.contiguous()
        if out is None or out is inp:
            out = torch.empty_like(inp)
        # the kernel's 16B-vector path requires aligned io + slot bases
        out_to_copy = None
        if inp.data_ptr() & 15:
            aligned = torch.empty_like(inp)
            aligned.copy_(inp)
            inp = aligned
        if out.data_ptr() & 15:
            out_to_copy = out
            out = torch.empty_like(inp)
        peers = (self._peer_data + [0] * (_MAX_WS - 1))[: _MAX_WS - 1]
        rc = _EXT().hcu_ar_launch(
            ctypes.c_void_p(inp.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            self._my_data,
            *peers,
            self._flags,
            self._sync,
            n,
            self._max_size,
            code,
            self.world_size,
            self.rank,
            self.slots,
            self._budget,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        if rc != 0:
            raise RuntimeError(f"custom AR: hcu_ar_launch rc={rc}")
        if out_to_copy is not None:
            out_to_copy.copy_(out)
            return out_to_copy
        return out

    def read_sync_words(self):
        """[seq, a1, a2, diag] host copy (tests/triage)."""
        host = (ctypes.c_uint64 * 3)()
        _HIP.hipMemcpy(
            host, ctypes.c_void_p(self._sync), 24, 2  # hipMemcpyDeviceToHost
        )
        return list(host)

    def dispose(self):
        if self.disposed:
            return
        ext = _EXT()
        if self._mm is not None:
            ext.hcu_unregister_host(
                ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self._mm)))
            )
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.disposed = True
        # own allocation + shm file stay until process exit: peers may still
        # hold mappings (shutdown races); tmpfs is wiped on node reboot and
        # the block is re-zeroed at every init


_ACTIVE_ALLOCATIONS = {}  # ipc handle bytes -> (base, max_size, ws, slots)

_EXT_SO = None


def _EXT():
    global _EXT_SO
    if _EXT_SO is None:
        _EXT_SO = _load_ext()
        # full launch ABI (kept out of _load_ext for one clear listing)
        e = _EXT_SO
        e.hcu_ar_launch.restype = ctypes.c_int
        e.hcu_ar_launch.argtypes = (
            [ctypes.c_void_p, ctypes.c_void_p]      # inp, out
            + [ctypes.c_longlong] * 10              # my_data, p0..p6, flags, sync
            + [ctypes.c_longlong, ctypes.c_longlong]  # n, slot_bytes
            + [ctypes.c_int] * 4                    # elem_mode, ws, rank, slots
            + [ctypes.c_longlong]                   # budget
            + [ctypes.c_void_p]                     # stream
        )
        e.hcu_register_host.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        e.hcu_register_host.restype = ctypes.c_int
        e.hcu_unregister_host.argtypes = [ctypes.c_void_p]
        e.hcu_unregister_host.restype = ctypes.c_int
    return _EXT_SO


def _expected_world_size() -> int:
    ws = 2
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size()
    except Exception:
        pass
    return min(ws, _MAX_WS)


def allocate_meta_buffer(size_bytes: int, device=None):
    """Allocate sync words + slot area; returns IPC handle bytes.

    sglang passes meta_size() + max_size; we round the slot capacity up to
    at least 1 MiB. Each rank stages only its OWN input, so one S x max_size
    region suffices (peers pull-read it over device IPC).
    """
    ws = _expected_world_size()
    slots = _slots_for_ws(ws)
    max_size = max(1 << 20, size_bytes - DATA_OFF)
    max_size = (max_size + 15) & ~15  # 16B-vector kernel path
    total = DATA_OFF + slots * max_size
    ptr = ctypes.c_void_p()
    rc = _HIP.hipMalloc(ctypes.byref(ptr), total)
    if rc != 0:
        raise RuntimeError(f"custom AR: hipMalloc rc={rc} ({total} bytes)")
    _HIP.hipMemset(ptr, 0, total)
    handle = IpcMemHandle()
    rc = _HIP.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
    if rc != 0:
        raise RuntimeError(f"custom AR: hipIpcGetMemHandle rc={rc}")
    _ACTIVE_ALLOCATIONS[bytes(handle)] = (ptr.value, max_size, ws, slots)
    return bytes(handle)


def get_meta_buffer_ipc_handle(meta) -> bytes:
    if isinstance(meta, torch.Tensor):
        # sglang's CustomAllreduce.register_buffer(t) passes a plain torch
        # tensor (its staging buffer). We never map it — return a dummy
        # handle that only our no-op register_buffer ever sees.
        return b"\x00" * 64
    return bytes(meta)  # meta already is the 64-byte handle


def meta_size() -> int:
    return DATA_OFF


def init_custom_ar(meta, rank_data, handles, offsets, rank, full_nvlink):
    world = len(handles)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return TritonARState(handles, offsets, rank, world, device)


def all_reduce_reg(fa: TritonARState, inp, out):
    return fa.all_reduce(inp, out)


def all_reduce_unreg(fa: TritonARState, inp, reg_buffer, out):
    # reg and unreg paths are identical: staging happens in-kernel through
    # the pre-registered IPC buffer, so arbitrary (graph-pool) input/output
    # addresses are fine
    return fa.all_reduce(inp, out)


def deterministic_all_reduce_reg(fa, inp, out):
    return fa.all_reduce(inp, out)


def deterministic_all_reduce_unreg(fa, inp, reg_buffer, out):
    return fa.all_reduce(inp, out)


def dispose(fa: TritonARState):
    fa.dispose()


def register_buffer(fa: TritonARState, t, handles, offsets):
    pass  # staging always goes through the pre-registered IPC buffer


def get_graph_buffer_ipc_meta(fa: TritonARState):
    return b"\x00" * 64, []


def register_graph_buffers(fa: TritonARState, handles, offsets):
    pass  # graph replays reuse the pre-registered buffer addresses


# --- quick AR (aiter quickreduce slot): intentionally unavailable ----------
def _qr_unavailable(*_a, **_k):
    raise NotImplementedError("custom AR: quick all-reduce not provided")


init_custom_qr = _qr_unavailable
qr_get_handle = _qr_unavailable
qr_open_handles = _qr_unavailable
qr_all_reduce = _qr_unavailable
qr_destroy = _qr_unavailable
qr_max_size = _qr_unavailable
