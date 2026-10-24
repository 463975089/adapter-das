"""Unit-level repro of the ~20ms completion-detection tick.

Hypothesis from trace W11: when a queue goes briefly idle between work items,
event completion detection / next submission takes a quantized ~19.5ms tick
(clusters 20/58/116ms = 1x/3x/6x). Test on an idle card:

  A) event.synchronize() right after a tiny kernel        (warm path)
  B) event.synchronize() after a tiny gap (host sleep)    (idle path)
  C) spin on event.query() then synchronize               (query dodge)
  D) raw kernel-launch latency after brief idle           (launch tick)
  E) tiny graph replay launch latency after brief idle    (graph tick)
Histogram each (us), count >5ms hits.
"""
import torch, time, statistics, collections

torch.cuda.init()
s = torch.cuda.Stream()
ev = torch.cuda.Event(enable_timing=False)

def hist(name, xs):
    ms = sorted(x / 1000 for x in xs)
    big = [m for m in ms if m > 5]
    buckets = collections.Counter()
    for m in ms:
        if m < 1: buckets["<1ms"] += 1
        elif m < 5: buckets["1-5ms"] += 1
        elif m < 25: buckets["5-25ms"] += 1
        elif m < 60: buckets["25-60ms"] += 1
        elif m < 80: buckets["60-80ms"] += 1
        elif m < 130: buckets["80-130ms"] += 1
        else: buckets[">130ms"] += 1
    print(f"{name}: n={len(ms)} p50={ms[len(ms)//2]:.3f}ms p99={ms[int(len(ms)*0.99)]:.3f}ms "
          f"max={ms[-1]:.3f}ms >5ms:{len(big)} {dict(buckets)}")
    if big:
        print(f"   slow samples (ms): {[f'{m:.1f}' for m in big[:30]]}")

N = 2000
a = torch.ones(1024, device="cuda")

# A) warm path: kernel -> record -> sync immediately
lat = []
with torch.cuda.stream(s):
    for _ in range(N):
        t0 = time.perf_counter_ns()
        a.add_(1.0)
        ev.record(s)
        ev.synchronize()
        lat.append((time.perf_counter_ns() - t0) / 1000)
hist("A sync-warm", lat)

# B) idle path: kernel -> 2ms host sleep (queue drains+idles) -> sync
lat = []
with torch.cuda.stream(s):
    for _ in range(N):
        a.add_(1.0)
        ev.record(s)
        time.sleep(0.002)
        t0 = time.perf_counter_ns()
        ev.synchronize()
        lat.append((time.perf_counter_ns() - t0) / 1000)
hist("B sync-after-2ms-idle", lat)

# C) query spin dodge on same idle path
lat = []
with torch.cuda.stream(s):
    for _ in range(N):
        a.add_(1.0)
        ev.record(s)
        time.sleep(0.002)
        t0 = time.perf_counter_ns()
        while not ev.query():
            pass
        lat.append((time.perf_counter_ns() - t0) / 1000)
hist("C query-spin-after-2ms-idle", lat)

# D) launch latency after brief idle
lat = []
for _ in range(N):
    time.sleep(0.002)
    t0 = time.perf_counter_ns()
    a.add_(1.0)
    lat.append((time.perf_counter_ns() - t0) / 1000)
torch.cuda.synchronize()
hist("D launch-after-2ms-idle", lat)

# E) tiny graph replay after brief idle
g = torch.cuda.CUDAGraph()
s0 = torch.cuda.Stream()
with torch.cuda.stream(s0):
    a.add_(1.0)
torch.cuda.synchronize()
with torch.cuda.graph(g):
    for _ in range(4):
        a.add_(1.0)
lat = []
for _ in range(N):
    time.sleep(0.002)
    t0 = time.perf_counter_ns()
    g.replay()
    lat.append((time.perf_counter_ns() - t0) / 1000)
torch.cuda.synchronize()
hist("E graph-replay-after-2ms-idle", lat)

# F) sync-after-longer-idle (50ms) — does tick count grow?
lat = []
with torch.cuda.stream(s):
    for _ in range(500):
        a.add_(1.0)
        ev.record(s)
        time.sleep(0.05)
        t0 = time.perf_counter_ns()
        ev.synchronize()
        lat.append((time.perf_counter_ns() - t0) / 1000)
hist("F sync-after-50ms-idle", lat)
print("done")
