"""bench3: does a graph containing a cross-rank NCCL allreduce hit the
~20ms launch tick? 4 procs (cards 0-3), compare:
  A) graph with allreduce captured inside (50 kernels + ar + 50 kernels)
  B) same-size graph without allreduce (control)
Measure replay() host-call duration per variant.
"""
import os, time, collections, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--gk", type=int, default=100)
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29811")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world)
    torch.cuda.set_device(args.rank)
    dev = torch.device("cuda")
    work = torch.ones(2048, device=dev)
    ar = torch.ones(8192, dtype=torch.bfloat16, device=dev)

    def hist(name, xs):
        ms = sorted(x / 1000 for x in xs)
        b = collections.Counter()
        for m in ms:
            if m < 1: b["<1ms"] += 1
            elif m < 5: b["1-5ms"] += 1
            elif m < 15: b["5-15ms"] += 1
            elif m < 25: b["15-25ms"] += 1
            else: b[">25ms"] += 1
        slow = [f"{m:.1f}" for m in ms if m > 5]
        print(f"[r{args.rank}] {name}: n={len(ms)} p50={ms[len(ms)//2]:.3f}ms "
              f"p99={ms[int(len(ms)*0.99)]:.3f}ms max={ms[-1]:.3f}ms {dict(b)}", flush=True)
        if slow:
            print(f"[r{args.rank}]   slow(ms): {slow[:20]}", flush=True)

    # warm streams
    s0 = torch.cuda.Stream()
    with torch.cuda.stream(s0):
        work.add_(1.0)
    dist.all_reduce(ar)
    torch.cuda.synchronize()
    dist.barrier()

    # A) graph WITH allreduce inside
    gA = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gA):
        for _ in range(args.gk):
            work.add_(1.0)
        dist.all_reduce(ar)
        for _ in range(args.gk):
            work.add_(1.0)
    dist.barrier()

    # B) control graph without allreduce
    gB = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gB):
        for _ in range(args.gk * 2):
            work.add_(1.0)
    dist.barrier()

    lat = []
    for _ in range(args.iters):
        time.sleep(0.002)
        t0 = time.perf_counter_ns()
        gB.replay()
        lat.append((time.perf_counter_ns() - t0) / 1000)
    torch.cuda.synchronize(); dist.barrier()
    hist("B control graph (no AR) replay-call", lat)

    lat = []
    for _ in range(args.iters):
        time.sleep(0.002)
        t0 = time.perf_counter_ns()
        gA.replay()
        lat.append((time.perf_counter_ns() - t0) / 1000)
    torch.cuda.synchronize(); dist.barrier()
    hist("A graph-with-AR replay-call", lat)

    # C) eager allreduce (no graph) for reference
    lat = []
    for _ in range(args.iters):
        time.sleep(0.002)
        t0 = time.perf_counter_ns()
        dist.all_reduce(ar)
        lat.append((time.perf_counter_ns() - t0) / 1000)
    torch.cuda.synchronize(); dist.barrier()
    hist("C eager allreduce call", lat)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[r{args.rank}] done", flush=True)

main()
