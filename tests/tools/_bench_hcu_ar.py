"""Quick latency bench: custom AR vs NCCL (bf16, TP2).

Run: HIP_VISIBLE_DEVICES=6,7 torchrun --nproc-per-node=2 --master-port=29815 \
        tests/tools/_bench_hcu_ar.py
"""

import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, "/root/hcu_t2")
from sgl_kernel._adapters import triton_ar  # noqa: E402

rank = int(os.environ["RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
gloo = dist.new_group(backend="gloo")
hb = triton_ar.allocate_meta_buffer(triton_ar.DATA_OFF + 8 * 1024 * 1024)
hs = [None] * 2
dist.all_gather_object(hs, hb, group=gloo)
st = triton_ar.init_custom_ar(None, None, hs, None, rank, True)


def bench(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) / iters * 1e6


if rank == 0:
    print(f"{'size':>8s} {'nccl_us':>8s} {'custom_us':>9s}", flush=True)
for kb in (4, 64, 256, 1024, 4096):
    n = kb * 1024 // 2
    x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    r = x.clone()
    t_nccl = bench(lambda: dist.all_reduce(r))
    t_cust = bench(lambda: st.all_reduce(x))
    if rank == 0:
        print(f"{kb:7d}K {t_nccl:8.1f} {t_cust:9.1f}", flush=True)
triton_ar.dispose(st)
dist.destroy_process_group()
