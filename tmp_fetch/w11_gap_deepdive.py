"""W11b: deep-dive ONE stall gap — dump ALL events (all cats, all threads)
in [g0-8ms, g1+8ms] to see: GPU truly idle? host threads doing what?
which launch owns the next kernel (by correlation across ALL runtime calls)?
"""
import gzip, json, sys, bisect, collections

path = sys.argv[1] if len(sys.argv) > 1 else r"E:\GITS\hcu-sglang-adapter\prof_1024\TP-0.trace.json.gz"
GAP_MS = 15.0
PICK = int(sys.argv[2]) if len(sys.argv) > 2 else 70  # which gap to dump

with gzip.open(path, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"]

# cat census
cats = collections.Counter(e.get("cat", "?") for e in ev if e.get("ph") == "X")
print("cat census:", dict(cats))

# GPU-side busy = kernel + gpu_memcpy + gpu_memset (+ anything with 'gpu' in cat)
gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
kernels = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"]
gpu_all = [e for e in ev if e.get("ph") == "X" and e.get("cat") in gpu_cats]
print(f"kernels={len(kernels)} gpu_all={len(gpu_all)}")

busy = sorted([(g["ts"], g["ts"] + g["dur"]) for g in gpu_all])
merged = []
for s, t in busy:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], t)
    else:
        merged.append([s, t])
gaps = []
for i in range(1, len(merged)):
    g0, g1 = merged[i - 1][1], merged[i][0]
    if g1 - g0 > GAP_MS * 1000:
        gaps.append((g0, g1, (g1 - g0) / 1000.0))
print(f"gaps(with memcpy union) > {GAP_MS}ms: {len(gaps)} (was 149 kernel-only)")
print("durations sample:", [f"{g[2]:.0f}" for g in gaps[:20]])

g0, g1, ms = gaps[PICK]
print(f"\n=== GAP #{PICK}: {ms:.1f}ms [{g0}..{g1}] ===")
lo, hi = g0 - 8000, g1 + 8000
win = [e for e in ev if e.get("ph") == "X" and lo <= e["ts"] <= hi]
win.sort(key=lambda x: (x["ts"], -x.get("dur", 0)))
for e in win:
    ts = (e["ts"] - g0) / 1000.0
    dur = e.get("dur", 0) / 1000.0
    cat = e.get("cat", "?")[:12]
    name = e.get("name", "?")[:70]
    tid = e.get("tid")
    corr = e.get("args", {}).get("correlation", "")
    print(f"  t={ts:+9.2f}ms dur={dur:8.2f}ms {cat:12} tid={tid} corr={corr} {name}")

# who owns the next kernel? search ALL events with same correlation
nextk = min((k for k in gpu_all if k["ts"] >= g1), key=lambda x: x["ts"])
corr = nextk.get("args", {}).get("correlation")
print(f"\nnext kernel: {nextk['name'][:60]} corr={corr} ts={(nextk['ts']-g0)/1000:+.2f}ms")
peers = [e for e in ev if e.get("args", {}).get("correlation") == corr and e.get("ph") == "X"]
for e in peers:
    print(f"  peer: t={(e['ts']-g0)/1000:+9.2f}ms dur={e.get('dur',0)/1000:8.2f}ms {e.get('cat','?')[:12]} {e.get('name','?')[:70]}")
