"""ctypes driver test for the C++ HIP AR extension (ar_ext.cu).

1) builds libhcu_ar.so with hipcc (if missing),
2) runs one bounded-diag AR per rank and compares against NCCL,
3) prints diag words (peer pub observed / expected / k / slot / done).

Run: torchrun --nproc-per-node=2 _test_ar_ext.py
"""

import os

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "..", "..", "sgl_kernel", "_adapters", "libhcu_ar.so")

SEQ_OFF, DONE_OFF, ARRIVED_OFF, PUB_OFF, DATA_OFF = 0x0, 0x100, 0x200, 0x400, 0x1000
S = 8
MAX_SIZE = 8 * 1024 * 1024

import ctypes


def build():
    src = os.path.join(HERE, "..", "..", "sgl_kernel", "_adapters", "ar_ext.cu")
    src = os.path.abspath(src)
    if os.path.exists(SO):
        os.remove(SO)
    rc = os.system(
        f"hipcc -O2 -fPIC -shared {src} -o {SO} 2>&1 | tail -5"
    )
    assert rc == 0, f"hipcc failed rc={rc}"


def main():
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")
    nccl = dist.new_group(backend="nccl")

    if rank == 0:
        build()
        print("[build] ok", flush=True)
    dist.barrier(group=gloo)

    so = ctypes.CDLL(SO)
    so.hcu_ar_launch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_longlong, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p,
    ]
    so.hcu_ar_launch.restype = ctypes.c_int

    import sgl_kernel.allreduce as ar

    meta = ar.allocate_meta_buffer(ar.meta_size() + MAX_SIZE)
    handle = bytes(ar.get_meta_buffer_ipc_handle(meta))
    gathered = [None] * ws
    dist.all_gather_object(gathered, (handle, 0), group=gloo)

    from sgl_kernel._adapters.triton_ar import _ACTIVE_ALLOCATIONS, _HIP, IpcMemHandle

    # explicit peer-access enable: hipIpcOpenMemHandle's lazy flag may be a
    # no-op on DTK; without enable, remote WRITES silently vanish
    _HIP.hipDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
    _HIP.hipDeviceEnablePeerAccess.restype = ctypes.c_int
    rc_peer = _HIP.hipDeviceEnablePeerAccess(1 - rank, 0)
    print(f"[r{rank}] hipDeviceEnablePeerAccess rc={rc_peer}", flush=True)

    base = _ACTIVE_ALLOCATIONS[meta][0]
    peer_base = 0
    for i, g in enumerate(gathered):
        if i == rank:
            continue
        hd = IpcMemHandle.from_buffer_copy(g[0])
        addr = ctypes.c_void_p()
        rc = _HIP.hipIpcOpenMemHandle(ctypes.byref(addr), hd, 1)
        assert rc == 0, rc
        peer_base = addr.value

    bases = torch.tensor([base, peer_base], dtype=torch.int64, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    diag = torch.zeros(8, dtype=torch.int32, device="cuda")

    def run(x, out, budget=0):
        n = x.numel()
        rc = so.hcu_ar_launch(
            x.data_ptr(), out.data_ptr(), rank, bases.data_ptr(),
            n, MAX_SIZE // 4, S, ws, 0, budget,
            diag.data_ptr(), 64, stream,
        )
        assert rc == 0, f"launch rc={rc}"

    # round 0 with a big bounded budget (diagnosis-safe)
    torch.manual_seed(7 + rank)
    x = torch.randn(4096, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    run(x, out, budget=20_000_000)
    torch.cuda.synchronize()
    d = diag.cpu().tolist()
    print(
        f"[r{rank}] diag: pub_seen={d[0]} expected={d[1]} k={d[2]} "
        f"slot={d[3]} done={d[4]} nb={d[5]}",
        flush=True,
    )
    ref = x.clone()
    dist.all_reduce(ref, group=nccl)
    md = (out - ref).abs().max().item()
    print(f"[r{rank}] maxdiff vs NCCL = {md:.6f}", flush=True)

    # a few more rounds, unbounded
    for it in range(5):
        x = torch.randn(1 << 16, dtype=torch.float32, device="cuda")
        out = torch.empty_like(x)
        run(x, out)
        torch.cuda.synchronize()
        ref = x.clone()
        dist.all_reduce(ref, group=nccl)
        md = (out - ref).abs().max().item()
        assert md < 1e-3, f"iter {it} maxdiff {md}"
        if rank == 0:
            print(f"[r0] round {it} maxdiff={md:.6f}", flush=True)

    dist.barrier(group=gloo)
    if rank == 0:
        print("EXT TEST DONE", flush=True)
    dist.destroy_process_group()


main()
