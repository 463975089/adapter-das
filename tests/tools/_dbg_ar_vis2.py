"""Where does cross-GPU staleness live, and does the flip-direction
design work? All steps 2-proc, GPU6/7.

  P5  r0 plain-stores 64 floats (unique addr); r1 remote plain load    (data first-touch)
  P6  r0 stores NEW values same addr; r1 remote plain load again       (stale test)
  P7  r0 stores 3rd values; r1 remote load with cache_modifier .cg     (L1 bypass fix?)
  P8  control: r0 stores 4th values; r1 reads via atomic add-0 words    (always fresh?)
  P9  r0 plain-stores INTO r1's buffer (remote write); r1 LOCAL .cg read   (flip design)
  P10 r0 writes data THEN flag into r1's buffer; r1 spins local atomic
      read on flag, then local .cg data read                            (full flip design)

Writer values: 100*(round+1). Reader prints what it saw.
"""

import os
import time

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import sgl_kernel.allreduce as ar
from sgl_kernel._adapters.triton_ar import _ACTIVE_ALLOCATIONS, _HIP, IpcMemHandle


@triton.jit
def _k_store_f(data_addr, val, n, BLOCK: tl.constexpr):
    data_ptr = tl.load(data_addr).to(tl.pointer_type(tl.float32))
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(data_ptr + offs, val + offs.to(tl.float32), mask=mask)


@triton.jit
def _k_load_f(out_ptr, data_addr, n, BLOCK: tl.constexpr, CG: tl.constexpr):
    data_ptr = tl.load(data_addr).to(tl.pointer_type(tl.float32))
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    if CG:
        v = tl.load(data_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")
    else:
        v = tl.load(data_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, v, mask=mask)


@triton.jit
def _k_atomic_read_f(out_ptr, data_addr, n, BLOCK: tl.constexpr):
    p = tl.load(data_addr).to(tl.pointer_type(tl.float32))
    for i in tl.static_range(BLOCK):
        if i < n:
            v = tl.atomic_add(p + i, 0.0, sem="acquire", scope="sys")
            tl.store(out_ptr + i, v)


@triton.jit
def _k_write_then_flag(data_addr, flag_addr, val, n, BLOCK: tl.constexpr):
    data_ptr = tl.load(data_addr).to(tl.pointer_type(tl.float32))
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(data_ptr + offs, val + offs.to(tl.float32), mask=mask)
    fp = tl.load(flag_addr).to(tl.pointer_type(tl.int32))
    tl.store(fp, val)


@triton.jit
def _k_spin_local_flag(out_ptr, flag_addr, want, BLOCK: tl.constexpr):
    fp = tl.load(flag_addr).to(tl.pointer_type(tl.int32))
    v = tl.atomic_add(fp, 0, sem="acquire", scope="sys")
    while v < want:
        v = tl.atomic_add(fp, 0, sem="acquire", scope="sys")
    tl.store(out_ptr + tl.arange(0, BLOCK), v, mask=tl.arange(0, BLOCK) < 1)


def main():
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")

    meta = ar.allocate_meta_buffer(ar.meta_size() + 4096)
    base = _ACTIVE_ALLOCATIONS[meta][0]
    handle = bytes(ar.get_meta_buffer_ipc_handle(meta))
    gathered = [None] * ws
    dist.all_gather_object(gathered, (handle, 0), group=gloo)
    import ctypes

    peer_base = 0
    for i, g in enumerate(gathered):
        if i == rank:
            continue
        hd = IpcMemHandle.from_buffer_copy(g[0])
        addr = ctypes.c_void_p()
        rc = _HIP.hipIpcOpenMemHandle(ctypes.byref(addr), hd, 1)
        assert rc == 0, rc
        peer_base = addr.value

    def i64(v):
        return torch.tensor([v], dtype=torch.int64, device="cuda")

    N = 64
    my_data = i64(base + 0x1000)
    peer_data = i64(peer_base + 0x1000)
    out = torch.zeros(N, dtype=torch.float32, device="cuda")

    # P5..P8: r0 writes successive rounds; r1 remote-reads with varying method
    for it in range(4):
        val = 100.0 * (it + 1)
        if rank == 0:
            _k_store_f[(1,)](my_data, val, N, BLOCK=N)
            torch.cuda.synchronize()
            dist.barrier(group=gloo)
        else:
            dist.barrier(group=gloo)
            if it == 0 or it == 1:
                _k_load_f[(1,)](out, peer_data, N, BLOCK=N, CG=False)
                torch.cuda.synchronize()
                tag = "P5 first-touch" if it == 0 else "P6 re-read"
            elif it == 2:
                _k_load_f[(1,)](out, peer_data, N, BLOCK=N, CG=True)
                torch.cuda.synchronize()
                tag = "P7 cg re-read"
            else:
                out.zero_()
                _k_load_f[(1,)](out, peer_data, 4, BLOCK=4, CG=False)
                torch.cuda.synchronize()
                tag = "P8 plain re-read"
            exp = val + torch.arange(3, device="cuda", dtype=torch.float32)
            ok = torch.allclose(out[:3], exp)
            print(f"[{tag} r1] saw={out[:3].tolist()} expect={exp.tolist()} ok={ok}", flush=True)

    # P9: r0 plain-stores INTO r1's buffer; r1 reads LOCALLY with .cg
    r1_data = i64(base + 0x2000)
    r1_flag = i64(base + 0x840)
    r0_peer_data = i64(peer_base + 0x2000)
    r0_peer_flag = i64(peer_base + 0x840)
    if rank == 0:
        dist.barrier(group=gloo)
        _k_store_f[(1,)](r0_peer_data, 500.0, N, BLOCK=N)
        torch.cuda.synchronize()
        print("[P9 r0] remote plain store sent", flush=True)
        dist.barrier(group=gloo)
        _k_write_then_flag[(1,)](r0_peer_data, r0_peer_flag, 600, N, BLOCK=N)
        torch.cuda.synchronize()
        print("[P10 r0] remote data+flag write sent", flush=True)
        dist.barrier(group=gloo)
    else:
        dist.barrier(group=gloo)
        _k_load_f[(1,)](out, r1_data, N, BLOCK=N, CG=True)
        torch.cuda.synchronize()
        exp = torch.tensor([500.0, 501.0, 502.0], device="cuda")
        ok = torch.allclose(out[:3], exp)
        print(f"[P9 r1] LOCAL cg read of remote-written data: saw={out[:3].tolist()} ok={ok}", flush=True)
        dist.barrier(group=gloo)
        t0 = time.time()
        res = torch.zeros(4, dtype=torch.int32, device="cuda")
        _k_spin_local_flag[(1,)](res, r1_flag, 600, BLOCK=4)
        _k_load_f[(1,)](out, r1_data, N, BLOCK=N, CG=True)
        torch.cuda.synchronize()
        exp = torch.tensor([600.0, 601.0, 602.0], device="cuda")
        ok = torch.allclose(out[:3], exp)
        print(f"[P10 r1] spin_flag={res[0].item()} dt={time.time()-t0:.1f}s data ok={ok} saw={out[:3].tolist()}", flush=True)
        dist.barrier(group=gloo)

    if rank == 0:
        print("VIS2 DONE", flush=True)
    dist.destroy_process_group()


main()
