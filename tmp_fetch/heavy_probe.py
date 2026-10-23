"""Heavy-load stall probe: reproduce the ~117ms quanta with big VRAM resident.

Phase graph : fill VRAM to ~12G free, CUDA-graph replay loop (100 kernels)
Phase churn : same resident, per-iter alloc/free 256MB (serving-like churn)
Stalls >20ms recorded with wall timestamps for cross-GPU synchrony.
"""
import json
import os
import sys
import time


def worker(gpu, seconds, phase):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch

    # fill VRAM: allocate 8G chunks until <12G free (keep 8G slack)
    chunks = []
    while True:
        free, total = torch.cuda.mem_get_info()
        if free < 12 * 2**30:
            break
        try:
            chunks.append(torch.empty(2 * 2**30, dtype=torch.float32, device="cuda"))
        except RuntimeError:
            break
    free, _ = torch.cuda.mem_get_info()
    resident_g = (sum(c.numel() * 4 for c in chunks)) / 2**30

    x = torch.randn(8192, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            x.mul_(1.0000001).add_(1e-7)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(100):
            x.mul_(1.0000001).add_(1e-7)

    def op():
        if phase == "graph":
            g.replay()
        else:
            t = torch.empty(64 * 2**22, dtype=torch.uint8, device="cuda")  # 256MB
            del t

    for _ in range(15):
        op()
    torch.cuda.synchronize()

    stalls = []
    n = 0
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        t0 = time.perf_counter()
        op()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e6
        n += 1
        if dt > 20000:
            stalls.append([round(time.time() * 1000), round(dt / 1000, 1)])
    out = {"gpu": gpu, "phase": phase, "resident_g": round(resident_g, 1),
           "free_g": round(free / 2**30, 1), "iters": n, "stalls": stalls}
    with open(f"/root/hstall_g{gpu}_{phase}.json", "w") as f:
        json.dump(out, f)
    print(f"gpu{gpu} {phase}: resident {resident_g:.0f}G free {free/2**30:.0f}G "
          f"{n} iters stalls>20ms: {len(stalls)}", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    for ph in ("graph", "churn"):
        ps = []
        for gpu in range(8):
            p = mp.Process(target=worker, args=(gpu, seconds, ph))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
    print("HEAVY_PROBE_DONE", flush=True)
