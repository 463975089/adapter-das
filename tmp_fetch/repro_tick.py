"""Synthetic reproducer for the ~19.4ms completion tick (W11 family A/B).

Mimics the exact pattern dissected from the nmz18 graph-mode trace:
  per step: [graph replay (2 internal streams, cross-stream edge)]
            + eager kernels
            + copy-stream: wait(compute event) -> pinned DtoH -> record
            + host: Event.synchronize()   <- process_batch_result pattern
            + NCCL allreduce across ranks (TP-like)
            + brief host idle
Knobs to bisect the trigger:  --nccl 0/1  --graph 0/1  --pinned 0/1
                              --xdst 0/1 (copy stream waits compute event)
                              --idle_ms 2  --iters 1500  --gk 200 (kernels in graph)
Report: sync latency histogram per rank + step-time outliers.
"""
import argparse, os, time, collections

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nccl", type=int, default=1)
    ap.add_argument("--graph", type=int, default=1)
    ap.add_argument("--pinned", type=int, default=1)
    ap.add_argument("--xdst", type=int, default=1)
    ap.add_argument("--idle_ms", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--gk", type=int, default=200)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--master_port", type=int, default=29711)
    args = ap.parse_args()

    import torch

    if args.world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args.master_port))
        import torch.distributed as dist
        dist.init_process_group("nccl", rank=args.rank, world_size=args.world)
        torch.cuda.set_device(args.rank)
        ar_buf = torch.ones(8192, dtype=torch.bfloat16, device="cuda")

    dev = torch.device("cuda")
    compute = torch.cuda.Stream()
    copy_s = torch.cuda.Stream()
    ev_tail = torch.cuda.Event(enable_timing=False)
    ev_copy = torch.cuda.Event(enable_timing=False)

    work = torch.ones(4096, device=dev)
    if args.pinned:
        host_buf = torch.empty(4096, pin_memory=True)
    else:
        host_buf = torch.empty(4096)

    g = None
    if args.graph:
        # warm side stream before capture
        warm = torch.cuda.Stream()
        with torch.cuda.stream(warm):
            work.add_(1.0)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s_b = torch.cuda.Stream()
        with torch.cuda.graph(g):
            # stream A portion (main capture stream)
            for _ in range(args.gk // 2):
                work.add_(1.0)
            ev_in = torch.cuda.Event()
            ev_in.record()
            # stream B portion on side stream, then join back
            with torch.cuda.stream(s_b):
                s_b.wait_event(ev_in)
                for _ in range(args.gk // 2):
                    work.add_(1.0)
                ev_join = torch.cuda.Event()
                ev_join.record(s_b)
            torch.cuda.current_stream().wait_event(ev_join)

    def hist(name, xs):
        ms = sorted(x / 1000 for x in xs)
        b = collections.Counter()
        for m in ms:
            if m < 0.1: b["<0.1ms"] += 1
            elif m < 1: b["0.1-1ms"] += 1
            elif m < 5: b["1-5ms"] += 1
            elif m < 25: b["5-25ms"] += 1
            elif m < 60: b["25-60ms"] += 1
            elif m < 130: b["60-130ms"] += 1
            else: b[">130ms"] += 1
        slow = [f"{m:.1f}" for m in ms if m > 5]
        print(f"[r{args.rank}] {name}: n={len(ms)} p50={ms[len(ms)//2]:.3f}ms "
              f"p99={ms[int(len(ms)*0.99)]:.3f}ms max={ms[-1]:.3f}ms {dict(b)}", flush=True)
        if slow:
            print(f"[r{args.rank}]   slow(ms): {slow[:30]}", flush=True)

    sync_lat, step_lat = [], []
    for it in range(args.iters):
        t_step = time.perf_counter_ns()
        with torch.cuda.stream(compute):
            if args.graph:
                g.replay()
            else:
                for _ in range(50):
                    work.add_(1.0)
            # eager tail on compute stream
            work.mul_(1.0000001)
            ev_tail.record(compute)
        with torch.cuda.stream(copy_s):
            if args.xdst:
                copy_s.wait_event(ev_tail)
            host_buf.copy_(work, non_blocking=True)
            ev_copy.record(copy_s)
        if args.nccl and args.world > 1:
            dist.all_reduce(ar_buf)
        time.sleep(args.idle_ms / 1000.0)
        t0 = time.perf_counter_ns()
        ev_copy.synchronize()
        sync_lat.append((time.perf_counter_ns() - t0) / 1000)
        step_lat.append((time.perf_counter_ns() - t_step) / 1000)

    hist("Event.synchronize(copy) lat", sync_lat)
    hist("step wall", step_lat)
    if args.world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    print(f"[r{args.rank}] done", flush=True)

if __name__ == "__main__":
    main()
