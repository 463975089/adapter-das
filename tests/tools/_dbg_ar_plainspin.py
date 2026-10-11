"""Does a remote PLAIN-load spin see a peer's V1-fenced flag mid-kernel?

writer(r0): local plain store flag=1 + post-fence (V1 recipe)
reader(r1): bounded spin with plain loads (no atomics, no volatile)
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
def _k_writer(flag_addr, fence_addr, val):
    fp = tl.load(flag_addr).to(tl.pointer_type(tl.int32))
    fep = tl.load(fence_addr).to(tl.pointer_type(tl.int32))
    tl.store(fp, val)
    tl.atomic_add(fep, 0, sem="release", scope="sys")  # V1 post-fence


@triton.jit
def _k_reader_plain(out_ptr, flag_addr, want, SPIN_MAX: tl.constexpr, CG: tl.constexpr):
    fp = tl.load(flag_addr).to(tl.pointer_type(tl.int32))
    if CG:
        v = tl.load(fp, cache_modifier=".cg")
    else:
        v = tl.load(fp)
    it = 0
    while (v < want) & (it < SPIN_MAX):
        if CG:
            v = tl.load(fp, cache_modifier=".cg")
        else:
            v = tl.load(fp)
        it += 1
    tl.store(out_ptr + 0, v)
    tl.store(out_ptr + 1, it)


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

    my_flag = i64(base + 0xB00)
    peer_flag = i64(peer_base + 0xB00)
    fence_w = i64(base + 0xB40)
    out = torch.zeros(4, dtype=torch.int32, device="cuda")

    import threading
    for rnd in range(2):
        want = rnd + 1
        if rank == 0:
            dist.barrier(group=gloo)
            time.sleep(2.0)  # let the reader's first fetch go stale FIRST
            _k_writer[(1,)](my_flag, fence_w, want)
            torch.cuda.synchronize()
            dist.barrier(group=gloo)
        else:
            dist.barrier(group=gloo)
            out.zero_()
            cg = rnd == 1
            _k_reader_plain[(1,)](out, peer_flag, want, SPIN_MAX=20_000_000, CG=cg)
            torch.cuda.synchronize()
            print(
                f"[rnd{rnd} cg={cg} r1] stale-start spin: saw={out[0].item()} want={want} iters={out[1].item()}",
                flush=True,
            )
            dist.barrier(group=gloo)

    if rank == 0:
        print("PLAINSPIN DONE", flush=True)
    dist.destroy_process_group()


main()
