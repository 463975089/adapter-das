"""IPC-mapping stall probe: 2 processes per GPU sharing a CUDA tensor
(cross-process IPC mapping, like custom-allreduce buffers), both looping
CUDA-graph replays. If the 117ms quanta appear -> exporter x IPC-mappings
interaction; if clean -> remaining delta is nccl/sglang internals."""
import json
import os
import sys
import time

import torch.multiprocessing as mp


def rank_loop(gpu, rank, seconds, q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch

    if rank == 0:
        shared = torch.randn(256 * 2**20 // 4, device="cuda")  # 256MB shared
        q.put(shared)
    else:
        shared = q.get()
        shared.add_(0.0)  # touch once to establish remote mapping

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

    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()

    stalls = []
    n = 0
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e6
        n += 1
        if dt > 20000:
            stalls.append([round(time.time() * 1000), round(dt / 1000, 1)])
    with open(f"/root/ipc_g{gpu}_r{rank}.json", "w") as f:
        json.dump({"gpu": gpu, "rank": rank, "iters": n, "stalls": stalls}, f)
    print(f"gpu{gpu} r{rank}: {n} iters stalls: {len(stalls)}", flush=True)


def gpu_pair(gpu, seconds):
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    ps = [ctx.Process(target=rank_loop, args=(gpu, r, seconds, q)) for r in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()


if __name__ == "__main__":
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 45
    procs = []
    ctx = mp.get_context("spawn")
    for gpu in range(8):
        p = ctx.Process(target=gpu_pair, args=(gpu, seconds))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("IPC_PROBE_DONE", flush=True)
