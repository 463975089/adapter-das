"""Minimal: one eager AR, print. Fast iteration on the hang."""

import os

import torch
import torch.distributed as dist

import sgl_kernel.allreduce as ar


def main():
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")
    nccl = dist.new_group(backend="nccl")

    meta = ar.allocate_meta_buffer(ar.meta_size() + 8 * 1024 * 1024)
    handle = bytes(ar.get_meta_buffer_ipc_handle(meta))
    gathered = [None] * ws
    dist.all_gather_object(gathered, (handle, 0), group=gloo)
    fa = ar.init_custom_ar(meta, None, [g[0] for g in gathered], [0] * ws, rank, True)
    print(f"[r{rank}] init ok", flush=True)

    n = 4096
    x = torch.randn(n, dtype=torch.float32, device="cuda")
    out = ar.all_reduce_reg(fa, x, torch.empty_like(x))
    print(f"[r{rank}] launch returned", flush=True)
    torch.cuda.synchronize()
    print(f"[r{rank}] GPU SYNCED", flush=True)
    ref = x.clone()
    dist.all_reduce(ref, group=nccl)
    torch.cuda.synchronize()
    print(f"[r{rank}] maxdiff={(out - ref).abs().max().item():.5f}", flush=True)
    dist.barrier(group=gloo)
    if rank == 0:
        print("MINI DONE", flush=True)
    dist.destroy_process_group()


main()
