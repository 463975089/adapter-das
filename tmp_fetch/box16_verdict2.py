import gzip, json, glob, collections, sys
d = sys.argv[1]
p = sorted(glob.glob(d + "/*TP-0*"))[0]
print("census on", p, flush=True)
ev = json.load(gzip.open(p, "rt"))["traceEvents"]
gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
gpu.sort(key=lambda x: x["ts"])
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
    elif m < 70: h["40-70ms(A)"] += 1
    elif m < 130: h["70-130ms"] += 1
    else: h[">130ms"] += 1
gl = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"
      and "GraphLaunch" in e.get("name", "")]
durs = sorted(e.get("dur", 0) / 1000 for e in gl)
tick = [x for x in durs if x > 15]
print(f"GAPS>15ms: {len(gaps)} {dict(h)}", flush=True)
print(f"graphLaunches n={len(gl)} tick(>15ms)={len(tick)} "
      f"p50={durs[len(durs)//2] if durs else -1:.2f}ms max={durs[-1] if durs else -1:.2f}ms", flush=True)
