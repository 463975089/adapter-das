"""Round 2: reproduce the 20ms execution tick under queue pressure.
Variants:
  G) NS streams all busy-looping tiny kernels; measure event sync latency on a probe stream
  G2) NS streams + one graph being replayed repeatedly
  H) (run separately) two processes sharing the card
Usage: bench_sync_tick2.py [nstreams] [mode]
"""
import torch, time, sys, threading, collections

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
MODE = sys.argv[2] if len(sys.argv) > 2 else "g"

torch.cuda.init()
dev = torch.device("cuda")

bufs = [torch.ones(256, device=dev) for _ in range(NS)]
streams = [torch.cuda.Stream() for _ in range(NS)]
stop = threading.Event()
ev = torch.cuda.Event(enable_timing=False)

def worker(i):
    s = streams[i]
    b = bufs[i]
    while not stop.is_set():
        with torch.cuda.stream(s):
            b.add_(1.0)
        s.synchronize()

threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(NS)]
for t in threads:
    t.start()
time.sleep(1.0)

probe = torch.ones(256, device=dev)
ps = torch.cuda.Stream()

if MODE == "g2":
    g = torch.cuda.CUDAGraph()
    s0 = torch.cuda.Stream()
    with torch.cuda.stream(s0):
        probe.add_(1.0)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(8):
            probe.add_(1.0)

def hist(name, xs):
    ms = sorted(x / 1000 for x in xs)
    buckets = collections.Counter()
    for m in ms:
        if m < 1: buckets["<1ms"] += 1
        elif m < 5: buckets["1-5ms"] += 1
        elif m < 25: buckets["5-25ms"] += 1
        elif m < 60: buckets["25-60ms"] += 1
        elif m < 130: buckets["60-130ms"] += 1
        else: buckets[">130ms"] += 1
    print(f"{name}: n={len(ms)} p50={ms[len(ms)//2]:.3f}ms p99={ms[int(len(ms)*0.99)] if len(ms)>1 else 0:.3f}ms "
          f"max={ms[-1]:.3f}ms {dict(buckets)}", flush=True)
    slow = [f"{m:.1f}" for m in ms if m > 5]
    if slow:
        print(f"   slow(ms): {slow[:40]}", flush=True)

N = 1000
lat = []
for _ in range(N):
    time.sleep(0.002)
    with torch.cuda.stream(ps):
        probe.add_(1.0)
        ev.record(ps)
        t0 = time.perf_counter_ns()
        ev.synchronize()
        lat.append((time.perf_counter_ns() - t0) / 1000)
hist(f"sync under {NS} busy streams (mode {MODE})", lat)

if MODE == "g2":
    lat = []
    for _ in range(N):
        time.sleep(0.002)
        t0 = time.perf_counter_ns()
        g.replay()
        lat.append((time.perf_counter_ns() - t0) / 1000)
    torch.cuda.synchronize()
    hist(f"graph replay under {NS} busy streams", lat)

# launch latency under pressure
lat = []
for _ in range(N):
    time.sleep(0.002)
    t0 = time.perf_counter_ns()
    probe.add_(1.0)
    lat.append((time.perf_counter_ns() - t0) / 1000)
torch.cuda.synchronize()
hist(f"launch under {NS} busy streams", lat)

stop.set()
for t in threads:
    t.join(timeout=5)
print("done")
