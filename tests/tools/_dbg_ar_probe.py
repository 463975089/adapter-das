"""Stepwise probe for the triton custom-AR hang: which primitive breaks?

Steps (each prints before/after on both ranks):
  S1 kernel reads own seq counter via atomic-add-0      -> launch+int2ptr
  S2 kernel does local atomic publish, plain read back  -> local atomics
  S3 rank0 remote-atomic +1 on peer's word, rank1 spins
     reading own word via atomic-add-0                  -> remote RMW write
  S4 rank0 plain-store to own word; rank1 spins via
     remote atomic-add-0 read                           -> remote RMW read
  S5 both: local spin via plain volatile load on a word the
     peer plain-stored (S5a peer store, S5b local spin)  -> plain visibility

Run: torchrun --nproc-per-node=2 _dbg_ar_probe.py
"""

import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def _k_read_seq(out_ptr, seq_addr, BLOCK: tl.constexpr):
    p = tl.load(seq_addr).to(tl.pointer_type(tl.int32))
    k = tl.atomic_add(p, 0, sem="acquire", scope="sys")
    tl.store(out_ptr + tl.arange(0, BLOCK), k, mask=tl.arange(0, BLOCK) < 1)


@triton.jit
def _k_local_pub(out_ptr, pub_addr, BLOCK: tl.constexpr):
    p = tl.load(pub_addr).to(tl.pointer_type(tl.int32))
    tl.atomic_add(p, 1, sem="release", scope="sys")
    v = tl.atomic_add(p, 0, sem="acquire", scope="sys")
    tl.store(out_ptr + tl.arange(0, BLOCK), v, mask=tl.arange(0, BLOCK) < 1)


@triton.jit
def _k_remote_bump(peer_addr):
    p = tl.load(peer_addr).to(tl.pointer_type(tl.int32))
    tl.atomic_add(p, 1, sem="release", scope="sys")


@triton.jit
def _k_remote_read(out_ptr, peer_addr, BLOCK: tl.constexpr):
    p = tl.load(peer_addr).to(tl.pointer_type(tl.int32))
    v = tl.atomic_add(p, 0, sem="acquire", scope="sys")
    tl.store(out_ptr + tl.arange(0, BLOCK), v, mask=tl.arange(0, BLOCK) < 1)


@triton.jit
def _k_plain_store(val, word_addr):
    p = tl.load(word_addr).to(tl.pointer_type(tl.int32))
    tl.store(p, val)


@triton.jit
def _k_volatile_load(out_ptr, word_addr, BLOCK: tl.constexpr):
    p = tl.load(word_addr).to(tl.pointer_type(tl.int32))
    v = tl.load(p, volatile=True)
    tl.store(out_ptr + tl.arange(0, BLOCK), v, mask=tl.arange(0, BLOCK) < 1)


def spin_read_local(out, word_addr_t, tag, want=1, tries=200):
    for _ in range(tries):
        _k_volatile_load[(1,)](out, word_addr_t, BLOCK=16)
        torch.cuda.synchronize()
        if out[0].item() >= want:
            return True
    return False


def main():
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")

    import ctypes

    from sgl_kernel._adapters.triton_ar import (
        _ACTIVE_ALLOCATIONS, _HIP, IpcMemHandle, allocate_meta_buffer,
        get_meta_buffer_ipc_handle, meta_size,
    )

    meta = allocate_meta_buffer(meta_size() + 4096)
    my_base = _ACTIVE_ALLOCATIONS[meta]  # int base address of own allocation
    handle = bytes(get_meta_buffer_ipc_handle(meta))
    gathered = [None] * ws
    dist.all_gather_object(gathered, (handle, 0), group=gloo)

    peer_base = [0]
    for i, g in enumerate(gathered):
        if i == rank:
            continue
        hd = IpcMemHandle.from_buffer_copy(g[0])
        addr = ctypes.c_void_p()
        rc = _HIP.hipIpcOpenMemHandle(ctypes.byref(addr), hd, 1)
        assert rc == 0, rc
        peer_base[0] = addr.value
    dist.barrier(group=gloo)
    if rank == 0:
        print("[setup] ipc open ok", flush=True)

    def i64(v):
        return torch.tensor([v], dtype=torch.int64, device="cuda")

    seq_t = i64(my_base + 0x0)
    pub_t = i64(my_base + 0x400)
    peer_pub_t = i64(peer_base[0] + 0x400)
    out = torch.zeros(16, dtype=torch.int32, device="cuda")

    # S1
    _k_read_seq[(1,)](out, seq_t, BLOCK=16)
    torch.cuda.synchronize()
    print(f"[S1 r{rank}] seq={out[0].item()} (expect 0)", flush=True)

    # S2
    _k_local_pub[(1,)](out, pub_t, BLOCK=16)
    torch.cuda.synchronize()
    print(f"[S2 r{rank}] pub={out[0].item()} (expect 1)", flush=True)
    dist.barrier(group=gloo)

    # S3: rank0 bumps peer's pub (offset 0x404 to not clash), rank1 spins locally
    peer_word_t = i64(peer_base[0] + 0x404)
    my_word_t = i64(my_base + 0x404)
    if rank == 0:
        dist.barrier(group=gloo)
        _k_remote_bump[(1,)](peer_word_t)
        torch.cuda.synchronize()
        print(f"[S3 r0] remote atomic +1 sent", flush=True)
    else:
        ok = False
        for _ in range(200):
            _k_volatile_load[(1,)](out, my_word_t, BLOCK=16)
            torch.cuda.synchronize()
            if out[0].item() >= 1:
                ok = True
                break
        print(f"[S1 r1] S3 remote-atomic-write seen={ok}", flush=True)
        dist.barrier(group=gloo)

    # S4: rank0 spins reading peer word via remote atomic-add-0;
    # rank1 plain-stores 1 to that word
    if rank == 1:
        dist.barrier(group=gloo)
        _k_plain_store[(1,)](1, my_word_t)
        torch.cuda.synchronize()
        print(f"[S4 r1] plain store sent", flush=True)
    else:
        ok = False
        for _ in range(200):
            _k_remote_read[(1,)](out, peer_word_t, BLOCK=16)
            torch.cuda.synchronize()
            if out[0].item() >= 1:
                ok = True
                break
        print(f"[S4 r0] remote-atomic-read saw plain store={ok}", flush=True)
        dist.barrier(group=gloo)

    dist.barrier(group=gloo)
    if rank == 0:
        print("PROBE DONE", flush=True)
    dist.destroy_process_group()


main()
