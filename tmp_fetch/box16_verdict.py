import gzip, json, glob, collections, sys
p = sorted(glob.glob("/root/prof_arm1/*TP-0*"))[0]
print("census on", p, flush=True)
with gzip.open(p, "rt") as f:
    ev = json.load(f)["traceEvents"]
gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
gpu.sort(key=lambda x: x["ts"])
print("gpu events:", len(gpu), flush=True)
busy = []
for e in gpu:
    s, t = e["ts"], e["ts"] + e["dur"]
    if busy and s <= busy[-1][1]:
        busy[-1][1] = max(busy[-1][1], t)
    else:
        busy.append([s, t])
gaps = [(busy[i][0] - busy[i - 1][1]) / 1000 for i in range(1, len(busy))
        if busy[i][0] - busy[i - 1][1] > 15000]
h = collections.Counter()
for m in gaps:
    if m < 25: h["15-25ms"] += 1
    elif m < 40: h["25-40ms"] += 1
    elif m < 70: h["40-70ms(A)"] += 1
    elif m < 130: h["70-130ms"] += 1
    else: h[">130ms"] += 1
print(f"GAPS>15ms: {len(gaps)} total={sum(gaps):.0f}ms {dict(h)}", flush=True)
if gaps:
    print("  all gaps(ms):", [f"{m:.0f}" for m in sorted(gaps, reverse=True)[:25]], flush=True)
gl = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"
      and "GraphLaunch" in e.get("name", "")]
tick = [e for e in gl if e.get("dur", 0) > 15000]
fast = sorted(e.get("dur", 0) / 1000 for e in gl if e.get("dur", 0) <= 15000)
print(f"graphLaunches n={len(gl)} tick(>15ms)={len(tick)} fast_p50={fast[len(fast)//2] if fast else -1:.3f}ms fast_max={fast[-1] if fast else -1:.2f}ms", flush=True)
if tick:
    print("  tick durs(ms):", sorted(f"{e.get('dur',0)/1000:.1f}" for e in tick)[:25], flush=True)
# span & step estimate
span = (busy[-1][1] - busy[0][0]) / 1e6
print(f"trace busy span {span:.2f}s", flush=True)
