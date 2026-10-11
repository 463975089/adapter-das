"""Bisect the _ar_kernel hang: K_A = full body without spin loops,
K_B = K_A + peer-pub spin. Uses the real TritonARState machinery."""

import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import sgl_kernel.allreduce as ar
from sgl_kernel._adapters import triton_ar as ta


@triton.jit
def _k_a(  # no spins: stage + rendezvous + reduce + done, garbage results ok
    inp_ptr, out_ptr, rank,
    data_addrs, pub_addrs, done_addrs, seq_addr, arrived_addr,
    n_elements,
    SLOT_CAP: tl.constexpr, S: tl.constexpr, WS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nblocks = tl.num_programs(0)
    dtype = inp_ptr.dtype.element_ty
    my_seq = tl.load(seq_addr).to(tl.pointer_type(tl.int32))
    arrived = tl.load(arrived_addr).to(tl.pointer_type(tl.int32))
    my_data = tl.load(data_addrs + rank).to(tl.pointer_type(dtype))
    my_pub = tl.load(pub_addrs + rank).to(tl.pointer_type(tl.int32))
    my_done = tl.load(done_addrs + rank).to(tl.pointer_type(tl.int32))
    k = tl.atomic_add(my_seq, 0, sem="acquire", scope="sys")
    slot = k % S
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    tl.store(
        my_data + slot * SLOT_CAP + offs,
        tl.load(inp_ptr + offs, mask=mask, other=0.0),
        mask=mask,
    )
    n_arr = tl.atomic_add(arrived + slot, 1, sem="acq_rel", scope="sys")
    if n_arr % nblocks == nblocks - 1:
        tl.atomic_add(my_pub + slot, 1, sem="release", scope="sys")
    acc = tl.load(my_data + slot * SLOT_CAP + offs, mask=mask, other=0.0).to(tl.float32)
    for i in tl.static_range(WS):
        if i != rank:
            pdata = tl.load(data_addrs + i).to(tl.pointer_type(dtype))
            acc += tl.load(pdata + slot * SLOT_CAP + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, acc.to(dtype), mask=mask)
    n_arr2 = tl.atomic_add(arrived + S + slot, 1, sem="acq_rel", scope="sys")
    if n_arr2 % nblocks == nblocks - 1:
        tl.atomic_add(my_done, 1, sem="release", scope="sys")
        tl.atomic_add(my_seq, 1, sem="release", scope="sys")


@triton.jit
def _k_b(  # K_A + spin on peer pub
    inp_ptr, out_ptr, rank,
    data_addrs, pub_addrs, done_addrs, seq_addr, arrived_addr,
    n_elements,
    SLOT_CAP: tl.constexpr, S: tl.constexpr, WS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nblocks = tl.num_programs(0)
    dtype = inp_ptr.dtype.element_ty
    my_seq = tl.load(seq_addr).to(tl.pointer_type(tl.int32))
    arrived = tl.load(arrived_addr).to(tl.pointer_type(tl.int32))
    my_data = tl.load(data_addrs + rank).to(tl.pointer_type(dtype))
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
        tl.atomic_add(my_pub + slot, 1, sem="release", scope="sys")
    for i in tl.static_range(WS):
        if i != rank:
            pp = tl.load(pub_addrs + i).to(tl.pointer_type(tl.int32))
            v = tl.atomic_add(pp + slot, 0, sem="acquire", scope="sys")
            while v < expected_pub:
                v = tl.atomic_add(pp + slot, 0, sem="acquire", scope="sys")
    acc = tl.load(my_data + slot * SLOT_CAP + offs, mask=mask, other=0.0).to(tl.float32)
    for i in tl.static_range(WS):
        if i != rank:
            pdata = tl.load(data_addrs + i).to(tl.pointer_type(dtype))
            acc += tl.load(pdata + slot * SLOT_CAP + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, acc.to(dtype), mask=mask)


def run_variant(name, kern, fa, n=4096):
    x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, 2048),)
    kern[grid](
        x, out, fa.rank,
        fa._data_addrs, fa._pub_addrs, fa._done_addrs,
        fa._seq, fa._arrived,
        n,
        SLOT_CAP=fa._payload_bytes // 2, S=ta.NUM_SLOTS, WS=fa.world_size,
        BLOCK=2048,
    )
    torch.cuda.synchronize()
    print(f"[{name} r{fa.rank}] returned ok (values may be garbage)", flush=True)


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
    handles = [g[0] for g in gathered]
    fa = ar.init_custom_ar(meta, None, handles, [0] * ws, rank, True)
    dist.barrier(group=gloo)
    if rank == 0:
        print("[setup] state ok", flush=True)

    run_variant("K_A-nospin", _k_a, fa)
    dist.barrier(group=gloo)
    run_variant("K_B-pubspin", _k_b, fa)
    dist.barrier(group=gloo)
    if rank == 0:
        print("BISECT DONE", flush=True)
    dist.destroy_process_group()


main()
