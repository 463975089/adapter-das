"""W11c: what do HOST threads do during a stall gap? Dump all non-GPU-track
events overlapping [g0, g1] for gap N, plus the cuda_runtime call that brackets
the resume (first host activity after g1).
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
print(f"gap #{PICK}: {(g1-g0)/1000:.1f}ms")

# host events = everything not gpu-side; show those overlapping [g0-1ms, g1+3ms]
host = [e for e in ev if e.get("ph") == "X" and e.get("cat") not in gpu_cats
        and e["ts"] < g1 + 3000 and e["ts"] + e.get("dur", 0) > g0 - 1000]
host.sort(key=lambda x: x["ts"])
print(f"host events overlapping window: {len(host)}")
# collapse tiny python frames: show runtime + cpu_op + annotations + long python only
shown = 0
for e in host:
    cat = e.get("cat", "?")
    dur = e.get("dur", 0) / 1000.0
    if cat == "python_function" and dur < 5.0:
        continue
    ts = (e["ts"] - g0) / 1000.0
    tid = e.get("tid")
    name = e.get("name", "?")[:90]
    print(f"  t={ts:+9.2f} dur={dur:9.2f}ms {cat[:16]:16} tid={tid} {name}")
    shown += 1
    if shown > 60:
        print("  ...")
        break

# resume point: first cuda_runtime event after g1
rt = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"],
            key=lambda x: x["ts"])
i = bisect.bisect_left([e["ts"] for e in rt], g1)
print("\nfirst cuda_runtime events after gap end:")
for e in rt[i:i+12]:
    print(f"  t={(e['ts']-g0)/1000:+9.2f}ms dur={e.get('dur',0)/1000:7.2f}ms {e['name'][:60]}")

# graph launch nearest before g1
gls = [e for e in rt if e["name"] == "hipGraphLaunch"]
j = bisect.bisect_left([e["ts"] for e in gls], g1)
print("\nhipGraphLaunch around gap:")
for e in gls[max(0, j-2):j+2]:
    print(f"  t={(e['ts']-g0)/1000:+9.2f}ms dur={e.get('dur',0)/1000:7.2f}ms corr={e.get('args',{}).get('correlation')}")
