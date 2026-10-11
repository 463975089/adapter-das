"""Two-GPU unit test for the HCU triton custom all-reduce.

Run inside the container (repo root on PYTHONPATH):

    CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH=/root/hcu_t2 \
    torchrun --nproc-per-node=2 --master-port=29777 \
    tests/tools/_test_triton_ar.py

Checks, against NCCL as the numerical reference:
  1. eager all-reduce across dtypes/sizes (incl. non-power-of-2 tails),
  2. CUDA-graph capture of TWO chained AR launches, replayed 200x with a
     fresh input each time — exercises the on-device round counter, slot
     rotation and the recycle handshake across many replays.

Exit line: ``RESULT: PASS`` only if every check passed on rank 0.
"""

import os

import torch
import torch.distributed as dist

import sgl_kernel.allreduce as ar

MAX_PAYLOAD = 8 * 1024 * 1024  # sglang's default custom-AR max_size
N_REPLAY = 200


def main():
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    assert ws == 2, "this test is written for world_size=2"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")
    nccl = dist.new_group(backend="nccl")

    meta = ar.allocate_meta_buffer(ar.meta_size() + MAX_PAYLOAD)
    handle = bytes(ar.get_meta_buffer_ipc_handle(meta))
    gathered = [None] * ws
    dist.all_gather_object(gathered, (handle, 0), group=gloo)
    handles = [g[0] for g in gathered]
    offsets = [g[1] for g in gathered]
    fa = ar.init_custom_ar(meta, None, handles, offsets, rank, True)
    if rank == 0:
        print(f"[init] world={ws} slots payload={MAX_PAYLOAD} ok", flush=True)

    fails = 0

    def check_eager(n, dtype):
        nonlocal fails
        torch.manual_seed(1234 + rank)
        x = torch.randn(n, dtype=dtype, device="cuda")
        ref = x.clone()
        dist.all_reduce(ref, group=nccl)
        out = ar.all_reduce_reg(fa, x, torch.empty_like(x))
        torch.cuda.synchronize()
        diff = (out.float() - ref.float()).abs().max().item()
        scale = max(1.0, ref.float().abs().max().item())
        ok = diff <= 0.05 * scale
        fails += not ok
        if rank == 0:
            print(
                f"[eager] n={n:<8} {str(dtype).split('.')[-1]:<8} "
                f"maxdiff={diff:.5f} {'PASS' if ok else 'FAIL'}",
                flush=True,
            )

    for n, dt in [
        (4096, torch.bfloat16),
        (1 << 20, torch.bfloat16),
        (12345, torch.float32),
        (3, torch.float32),
        (777777, torch.float16),
    ]:
        check_eager(n, dt)

    # --- graph capture: two chained ARs, rotating inputs across replays ---
    n = 8192
    sin_ = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    s1 = torch.empty_like(sin_)
    s2 = torch.empty_like(sin_)
    ar.all_reduce_reg(fa, sin_, s1)  # eager warm-up: triton compile
    torch.cuda.synchronize()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        ar.all_reduce_reg(fa, sin_, s1)
        ar.all_reduce_reg(fa, s1, s2)
    torch.cuda.current_stream().wait_stream(side)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ar.all_reduce_reg(fa, sin_, s1)
        ar.all_reduce_reg(fa, s1, s2)

    bad = 0
    torch.manual_seed(99 + rank)
    for it in range(N_REPLAY):
        sin_.copy_(torch.randn(n, dtype=torch.bfloat16, device="cuda"))
        g.replay()
        ref1 = sin_.clone()
        dist.all_reduce(ref1, group=nccl)
        ref2 = ref1.clone()
        dist.all_reduce(ref2, group=nccl)
        torch.cuda.synchronize()
        d1 = (s1.float() - ref1.float()).abs().max().item()
        d2 = (s2.float() - ref2.float()).abs().max().item()
        if d1 > 0.1 or d2 > 0.1:
            bad += 1
        if rank == 0 and (it < 3 or it == N_REPLAY - 1):
            print(f"[graph] it={it:<4} d1={d1:.5f} d2={d2:.5f}", flush=True)
    if rank == 0:
        print(f"[graph] replay done bad={bad}/{N_REPLAY}", flush=True)
    fails += bad > 0

    ar.dispose(fa)
    flag = torch.tensor([fails], dtype=torch.int64)
    dist.all_reduce(flag, group=gloo)
    if rank == 0:
        print(f"RESULT: {'PASS' if int(flag[0]) == 0 else 'FAIL'}", flush=True)
    dist.barrier(group=gloo)
    dist.destroy_process_group()


main()
