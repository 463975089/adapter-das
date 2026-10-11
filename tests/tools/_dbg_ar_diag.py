"""Diagnostic: run the AR kernel protocol with BOUNDED spins and report
what the peer flags actually read as. Both ranks print observations.

out[0]=peer_pub_seen  out[1]=expected_pub  out[2]=k  out[3]=slot
out[4]=peer_done_seen out[5]=arrived_count(observed by this block0)
"""

import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import sgl_kernel.allreduce as ar
from sgl_kernel._adapters import triton_ar as ta


@triton.jit
def _ar_diag(
    inp_ptr, out_ptr, rank,
    data_addrs, pub_addrs, done_addrs, seq_addr, arrived_addr,
    n_elements,
    SLOT_CAP: tl.constexpr, S: tl.constexpr, WS: tl.constexpr,
    BLOCK: tl.constexpr, SPIN_MAX: tl.constexpr,
):
    pid = tl.program_id(0)
    nblocks = tl.num_programs(0)
    my_seq = tl.load(seq_addr).to(tl.pointer_type(tl.int32))
    arrived = tl.load(arrived_addr).to(tl.pointer_type(tl.int32))
    my_data = tl.load(data_addrs + rank).to(tl.pointer_type(inp_ptr.dtype.element_ty))
    my_pub = tl.load(pub_addrs + rank).to(tl.pointer_type(tl.int32))

    k = tl.atomic_add(my_seq, 0, sem="acquire", scope="sys")
    slot = k % S
    expected_pub = k // S + 1

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    tl.store(
        my_data + slot * SLOT_CAP + offs,
        tl.load(inp_ptr + offs, mask=mask, other=0.0),
        mask=mask,
    )
    n_arr = tl.atomic_add(arrived + slot, 1, sem="acq_rel", scope="sys")
    if n_arr % nblocks == nblocks - 1:
        tl.atomic_add(my_seq, 0, sem="release", scope="sys")
        tl.store(my_pub + slot, expected_pub)
        tl.atomic_add(my_seq, 0, sem="release", scope="sys")  # V1 flush fence

    # bounded spin, record what we saw
    peer_val = -1
    iters = 0
    for i in tl.static_range(WS):
        if i != rank:
            pp = tl.load(pub_addrs + i).to(tl.pointer_type(tl.int32))
            v = tl.atomic_add(pp + slot, 0, sem="acquire", scope="sys")
            it = 0
            while (v < expected_pub) & (it < SPIN_MAX):
                v = tl.atomic_add(pp + slot, 0, sem="acquire", scope="sys")
                it += 1
            peer_val = v

    if pid == 0:
        tl.store(out_ptr + 0, peer_val.to(inp_ptr.dtype.element_ty))
        tl.store(out_ptr + 1, expected_pub.to(inp_ptr.dtype.element_ty))
        tl.store(out_ptr + 2, k.to(inp_ptr.dtype.element_ty))
        tl.store(out_ptr + 3, slot.to(inp_ptr.dtype.element_ty))


def main():
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")

    meta = ar.allocate_meta_buffer(ar.meta_size() + 8 * 1024 * 1024)
    handle = bytes(ar.get_meta_buffer_ipc_handle(meta))
    gathered = [None] * ws
    dist.all_gather_object(gathered, (handle, 0), group=gloo)
    fa = ar.init_custom_ar(
        meta, None, [g[0] for g in gathered], [0] * ws, rank, True
    )
    dist.barrier(group=gloo)
    if rank == 0:
        print("[setup] ok", flush=True)

    for rnd in range(3):
        n = 4096
        x = torch.randn(n, dtype=torch.float32, device="cuda")
        out = torch.zeros(n, dtype=torch.float32, device="cuda")
        _ar_diag[(triton.cdiv(n, 2048),)](
            x, out, fa.rank,
            fa._data_addrs, fa._pub_addrs, fa._done_addrs,
            fa._seq, fa._arrived,
            n,
            SLOT_CAP=fa._payload_bytes // 4, S=ta.NUM_SLOTS, WS=fa.world_size,
            BLOCK=2048, SPIN_MAX=200000,
        )
        torch.cuda.synchronize()
        print(
            f"[diag r{rank} round{rnd}] peer_pub_seen={out[0].item():.0f} "
            f"expected={out[1].item():.0f} k={out[2].item():.0f} slot={out[3].item():.0f}",
            flush=True,
        )
        dist.barrier(group=gloo)

    if rank == 0:
        print("DIAG DONE", flush=True)
    dist.destroy_process_group()


main()
