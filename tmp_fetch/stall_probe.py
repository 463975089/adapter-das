"""Pure-torch stall probe: does the box freeze GPU work with ~117ms quanta?

8 workers, one per GPU. Phase A: CUDA-graph replay loop (100 tiny kernels
per graph). Phase B: eager kernel loop (same kernels, no graph). Record
per-iteration wall time; any iteration >20ms is a stall event with its
monotonic timestamp. Outputs /root/stall_g{i}.json for cross-worker
synchrony analysis.
"""
import json
import os
import sys
import time


def worker(gpu, seconds, phase):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch

    x = torch.randn(8192, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            x.mul_(1.0000001).add_(1e-7)
    torch.cuda.current_stream().wait_stream(s)

    if phase == "graph":
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(100):
                x.mul_(1.0000001).add_(1e-7)
        op = g.replay
    else:
        def op():
            for _ in range(100):
                x.mul_(1.0000001).add_(1e-7)

    for _ in range(20):
        op()
    torch.cuda.synchronize()

    stalls = []
    n = 0
    total_us = 0.0
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        t0 = time.perf_counter()
        op()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e6
        n += 1
        total_us += dt
        if dt > 20000:
            stalls.append([round(time.time() * 1000), round(dt / 1000, 1)])
    out = {
        "gpu": gpu, "phase": phase, "iters": n,
        "avg_us": round(total_us / max(n, 1), 1),
        "stalls": stalls,
    }
    with open(f"/root/stall_g{gpu}_{phase}.json", "w") as f:
        json.dump(out, f)
    print(f"gpu{gpu} {phase}: {n} iters avg {out['avg_us']:.0f}us "
          f"stalls>20ms: {len(stalls)}", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 45
    phase = sys.argv[2] if len(sys.argv) > 2 else "graph"
    for ph in ([phase] if phase != "both" else ["graph", "eager"]):
        ps = []
        for gpu in range(8):
            p = mp.Process(target=worker, args=(gpu, seconds, ph))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
    print("PROBE_DONE", flush=True)
