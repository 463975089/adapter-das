"""W11d: reconstruct stream/event dependency chain around gap #70.
Dump ALL cuda_runtime calls in [g0-1020ms, g1+60ms] with args (stream, event),
plus which stream each DtoH/kernel near the gap ran on (args.stream).
"""
import gzip, json, sys, bisect

path = sys.argv[1] if len(sys.argv) > 1 else r"E:\GITS\hcu-sglang-adapter\prof_1024\TP-0.trace.json.gz"
GAP_MS = 15.0
PICK = int(sys.argv[2]) if len(sys.argv) > 2 else 70

with gzip.open(path, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"]

gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
gpu_all = [e for e in ev if e.get("ph") == "X" and e.get("cat") in gpu_cats]
busy = sorted([(g["ts"], g["ts"] + g["dur"]) for g in gpu_all])
merged = []
for s, t in busy:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], t)
    else:
        merged.append([s, t])
gaps = [(merged[i - 1][1], merged[i][0]) for i in range(1, len(merged))
        if merged[i][0] - merged[i - 1][1] > GAP_MS * 1000]
g0, g1 = gaps[PICK]
print(f"gap #{PICK}: {(g1-g0)/1000:.1f}ms  [{g0:.0f}..{g1:.0f}]")

rt = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"],
            key=lambda x: x["ts"])
print("\n=== runtime calls [g0-1020ms, g1+60ms] (event/stream ops + graph launches) ===")
for e in rt:
    ts = e["ts"]
    if ts < g0 - 1020000 or ts > g1 + 60000:
        continue
    name = e["name"]
    if not any(k in name for k in ("Event", "Stream", "Graph", "Memcpy", "DeviceSynchronize")):
        continue
    args = e.get("args", {})
    trel = (ts - g0) / 1000.0
    print(f"  t={trel:+10.2f}ms dur={e.get('dur',0)/1000:8.2f}ms {name:32} args={ {k:v for k,v in args.items() if k in ('stream','event','correlation','context')} }")

# stream attribution of gpu events around the gap
print("\n=== gpu events [g0-3ms, g1+3ms] with stream ===")
ge = sorted([e for e in gpu_all], key=lambda x: x["ts"])
i = bisect.bisect_left([e["ts"] for e in ge], g0 - 3000)
j = bisect.bisect_left([e["ts"] for e in ge], g1 + 3000)
for e in ge[i:j]:
    args = e.get("args", {})
    trel = (e["ts"] - g0) / 1000.0
    print(f"  t={trel:+8.2f}ms dur={e.get('dur',0)/1000:6.2f}ms {e.get('cat','?')[:10]:10} stream={args.get('stream')} tid={e.get('tid')} {e.get('name','?')[:50]}")
