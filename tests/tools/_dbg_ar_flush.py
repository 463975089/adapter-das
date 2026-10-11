"""Which store primitive makes a flag visible to the PEER MID-KERNEL?

Writer publishes flag=1 with variant V; reader (concurrent kernel, other
GPU) bounded-spins via remote atomic read and reports how long it took.

  V1 fence(atomic release) -> plain store -> fence(atomic release)
  V2 fence -> plain store cache_modifier=.wt
  V3 fence -> plain store cache_modifier=.cg
  V4 fence -> atomic_xchg(flag, 1, release, sys)

Reader reports seen value and rough spin iterations (via a counter in out).
"""

import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import sgl_kernel.allreduce as ar
from sgl_kernel._adapters.triton_ar import _ACTIVE_ALLOCATIONS, _HIP, IpcMemHandle


@triton.jit
def _k_writer(flag_addr, variant, fence_addr):
    fp = tl.load(flag_addr).to(tl.pointer_type(tl.int32))
    fep = tl.load(fence_addr).to(tl.pointer_type(tl.int32))
    tl.atomic_add(fep, 0, sem="release", scope="sys")  # fence before
    if variant == 0:
        tl.store(fp, 1)
        tl.atomic_add(fep, 0, sem="release", scope="sys")  # fence after
    elif variant == 1:
        tl.store(fp, 1, cache_modifier=".wt")
    elif variant == 2:
        tl.store(fp, 1, cache_modifier=".cg")
    else:
        tl.atomic_xchg(fp, 1, sem="release", scope="sys")


@triton.jit
def _k_reader(out_ptr, flag_addr, SPIN_MAX: tl.constexpr, BLOCK: tl.constexpr):
    fp = tl.load(flag_addr).to(tl.pointer_type(tl.int32))
    v = tl.atomic_add(fp, 0, sem="acquire", scope="sys")
    it = 0
    while (v < 1) & (it < SPIN_MAX):
        v = tl.atomic_add(fp, 0, sem="acquire", scope="sys")
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
    base = _ACTIVE_ALLOCATIONS[meta]
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

    out = torch.zeros(4, dtype=torch.int32, device="cuda")

    # flag word per variant at 0xA00 + variant*4; fence word at 0xA40
    for variant, name in enumerate(["V1 postfence", "V2 wt", "V3 cg", "V4 xchg"]):
        my_flag = i64(base + 0xA00 + variant * 4)
        peer_flag = i64(peer_base + 0xA00 + variant * 4)
        fence_w = i64(base + 0xA40)
        # writer = rank0, reader = rank1, concurrently
        if rank == 0:
            dist.barrier(group=gloo)  # both ready
            _k_writer[(1,)](my_flag, variant, fence_w)
            torch.cuda.synchronize()
            dist.barrier(group=gloo)  # reader done
        else:
            dist.barrier(group=gloo)
            out.zero_()
            _k_reader[(1,)](out, peer_flag, SPIN_MAX=2_000_000, BLOCK=4)
            torch.cuda.synchronize()
            print(
                f"[{name} r1] saw={out[0].item()} after_iters={out[1].item()}",
                flush=True,
            )
            dist.barrier(group=gloo)
        dist.barrier(group=gloo)

    if rank == 0:
        print("FLUSH PROBE DONE", flush=True)
    dist.destroy_process_group()


main()
