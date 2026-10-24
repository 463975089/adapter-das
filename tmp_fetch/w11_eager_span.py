"""W11f: for each eager >15ms kernel gap, find events that SPAN most of the gap
(python_function / cpu_op / cuda_runtime / user_annotation). If a python frame
spans [g0,g1] -> host-side 20ms python stall. If nothing spans it and last
launch before g1 was long before -> GPU sat on queued work.
"""
import gzip, json, sys, glob, bisect, collections

folder = sys.argv[1] if len(sys.argv) > 1 else r"E:\GITS\hcu-sglang-adapter\prof_1024\eager"
GAP_MS = 15.0
MAXG = int(sys.argv[2]) if len(sys.argv) > 2 else 8

path = sorted(glob.glob(folder + "/*.json.gz"))[0]
print("file:", path)
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
print(f"gaps: {len(gaps)}")

others = [e for e in ev if e.get("ph") == "X" and e.get("cat") not in gpu_cats
          and e.get("cat") != "Trace"]
others.sort(key=lambda x: x["ts"])
o_ts = [e["ts"] for e in others]

span_count = collections.Counter()
for gi, (g0, g1) in enumerate(gaps[:MAXG]):
    ms = (g1 - g0) / 1000.0
    # events covering >=70% of gap
    spanners = []
    i = bisect.bisect_left(o_ts, g0) - 2
    i = max(0, i)
    while i < len(others) and others[i]["ts"] <= g1:
        e = others[i]
        if e["ts"] <= g0 + 0.3 * (g1 - g0) and e["ts"] + e.get("dur", 0) >= g1 - 0.3 * (g1 - g0):
            spanners.append(e)
        i += 1
    print(f"\n--- gap {gi}: {ms:.1f}ms")
    if not spanners:
        print("   NOTHING spans 70% of gap")
    spanners.sort(key=lambda e: -e.get("dur", 0))
    for e in spanners[:8]:
        cov = (min(e["ts"] + e.get("dur", 0), g1) - max(e["ts"], g0)) / (g1 - g0) * 100
        print(f"   span {cov:5.1f}% dur={e.get('dur',0)/1000:7.1f}ms {e.get('cat','?')[:14]:14} tid={e.get('tid')} {e.get('name','?')[:100]}")
    # last runtime call before gap end
    rt = [e for e in others if e.get("cat") == "cuda_runtime"]
    j = bisect.bisect_left([e["ts"] for e in rt], g1)
    for e in rt[max(0, j - 3):j]:
        print(f"   pre-end rt: t={(e['ts']-g0)/1000:+7.2f}ms dur={e.get('dur',0)/1000:6.2f}ms {e['name'][:60]}")
