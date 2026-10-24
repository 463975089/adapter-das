"""Memory-pressure hypothesis test: does high VRAM fill trigger the ~20ms
graph-launch / event-sync tick on a healthy card?

  --fill_gb N : allocate N GB and hold during the benches
Measures: g.replay() host-call duration, Event.synchronize latency,
kernel-launch latency — histograms each at the given fill level.
"""
import torch, time, argparse, collections

ap = argparse.ArgumentParser()
ap.add_argument("--fill_gb", type=float, default=0)
ap.add_argument("--n", type=int, default=1500)
args = ap.parse_args()

torch.cuda.init()
free, total = torch.cuda.mem_get_info(0)
print(f"before fill: {free/2**30:.1f}G free / {total/2**30:.1f}G")

hold = []
gb = int(args.fill_gb)
for i in range(gb):
    try:
        hold.append(torch.empty(1 << 30, dtype=torch.uint8, device="cuda"))
    except Exception as e:
        print(f"fill stopped at {i}G: {e}")
        break
free2, _ = torch.cuda.mem_get_info(0)
print(f"after fill({len(hold)}G): {free2/2**30:.1f}G free")

def hist(name, xs):
    ms = sorted(x / 1000 for x in xs)
    b = collections.Counter()
    for m in ms:
        if m < 0.1: b["<0.1ms"] += 1
        elif m < 1: b["0.1-1ms"] += 1
        elif m < 5: b["1-5ms"] += 1
        elif m < 25: b["5-25ms"] += 1
        elif m < 60: b["25-60ms"] += 1
        else: b[">60ms"] += 1
    slow = [f"{m:.1f}" for m in ms if m > 5]
    print(f"{name}: n={len(ms)} p50={ms[len(ms)//2]:.3f}ms p99={ms[int(len(ms)*0.99)]:.3f}ms "
          f"max={ms[-1]:.3f}ms {dict(b)}", flush=True)
    if slow:
        print(f"   slow(ms): {slow[:40]}", flush=True)

a = torch.ones(1024, device="cuda")

# graph with 100 kernels
s0 = torch.cuda.Stream()
with torch.cuda.stream(s0):
    a.add_(1.0)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(100):
        a.add_(1.0)

ev = torch.cuda.Event(enable_timing=False)
s = torch.cuda.Stream()

# 1) graph replay host-call duration after 2ms idle
lat = []
for _ in range(args.n):
    time.sleep(0.002)
    t0 = time.perf_counter_ns()
    g.replay()
    lat.append((time.perf_counter_ns() - t0) / 1000)
torch.cuda.synchronize()
hist("graph-replay-call @fill", lat)

# 2) event sync after idle
lat = []
with torch.cuda.stream(s):
    for _ in range(args.n):
        a.add_(1.0)
        ev.record(s)
        time.sleep(0.002)
        t0 = time.perf_counter_ns()
        ev.synchronize()
        lat.append((time.perf_counter_ns() - t0) / 1000)
hist("event-sync @fill", lat)

# 3) plain launch after idle
lat = []
for _ in range(args.n):
    time.sleep(0.002)
    t0 = time.perf_counter_ns()
    a.add_(1.0)
    lat.append((time.perf_counter_ns() - t0) / 1000)
torch.cuda.synchronize()
hist("launch @fill", lat)

# 4) graph replay back-to-back (no idle) — production steps are ~1s apart,
# but the ~20ms launches in the trace followed a sync/idle
lat = []
for _ in range(args.n):
    t0 = time.perf_counter_ns()
    g.replay()
    lat.append((time.perf_counter_ns() - t0) / 1000)
torch.cuda.synchronize()
hist("graph-replay-call back2back @fill", lat)
print("done @fill", gb, "G")
