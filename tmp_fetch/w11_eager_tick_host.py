"""W11e: eager trace (no DSPARK) — for each >15ms kernel gap, find the host
cuda_runtime call overlapping the gap (its own duration tells whether the
runtime call itself blocked ~20ms: submission tick, or returned fast while
the gap came from elsewhere).
"""
import gzip, json, sys, glob, bisect, collections

folder = sys.argv[1] if len(sys.argv) > 1 else r"E:\GITS\hcu-sglang-adapter\prof_1024\eager"
GAP_MS = 15.0

paths = sorted(glob.glob(folder + "/*.json.gz")) or sorted(glob.glob(folder + "/*.gz"))
print("files:", paths)
for path in paths:
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    ev = data["traceEvents"]
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    gpu_all = [e for e in ev if e.get("ph") == "X" and e.get("cat") in gpu_cats]
    if not gpu_all:
        print(f"{path}: no gpu events?")
        continue
    busy = sorted([(g["ts"], g["ts"] + g["dur"]) for g in gpu_all])
    merged = []
    for s, t in busy:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], t)
        else:
            merged.append([s, t])
    gaps = [(merged[i - 1][1], merged[i][0]) for i in range(1, len(merged))
            if merged[i][0] - merged[i - 1][1] > GAP_MS * 1000]
    print(f"\n=== {path}: gaps>{GAP_MS}ms = {len(gaps)}")
    rt = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"],
                key=lambda x: x["ts"])
    rt_ts = [e["ts"] for e in rt]
    for g0, g1 in gaps:
        ms = (g1 - g0) / 1000.0
        # runtime calls whose [ts, ts+dur] overlaps the gap
        ov = []
        i = bisect.bisect_left(rt_ts, g0) - 3
        i = max(0, i)
        while i < len(rt) and rt[i]["ts"] <= g1:
            e = rt[i]
            if e["ts"] + e.get("dur", 0) >= g0:
                ov.append(e)
            i += 1
        big = [e for e in ov if e.get("dur", 0) > 3000]
        desc = ", ".join(f"{e['name']}={e['dur']/1000:.1f}ms@{(e['ts']-g0)/1000:+.1f}" for e in big[:5]) or \
               "NONE>3ms (" + ", ".join(f"{e['name']}@{(e['ts']-g0)/1000:+.1f}" for e in ov[:4]) + ")"
        print(f"  gap {ms:6.1f}ms: {desc}")
