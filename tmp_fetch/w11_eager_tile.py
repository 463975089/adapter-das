"""W11g: for each eager gap, compute python_function coverage on the forward
thread inside the gap + top frames. Distinguishes:
  - python ~100%: host-python stall (slow op in python / lock in python land)
  - python ~0%:   host blocked in untracked native (GIL-released futex etc.)
"""
import gzip, json, sys, glob, bisect, collections

folder = sys.argv[1] if len(sys.argv) > 1 else r"E:\GITS\hcu-sglang-adapter\prof_1024\eager"
GAP_MS = 15.0
MAXG = int(sys.argv[2]) if len(sys.argv) > 2 else 75

path = sorted(glob.glob(folder + "/*.json.gz"))[0]
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

py = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "python_function"]
py.sort(key=lambda x: x["ts"])
py_ts = [e["ts"] for e in py]

for gi, (g0, g1) in enumerate(gaps[:MAXG]):
    ms = (g1 - g0) / 1000.0
    # union of python intervals overlapping gap (any thread)
    i = bisect.bisect_left(py_ts, g0) - 2
    i = max(0, i)
    segs = []
    tids = collections.Counter()
    top = collections.Counter()
    while i < len(py) and py[i]["ts"] <= g1:
        e = py[i]
        s2, t2 = max(e["ts"], g0), min(e["ts"] + e.get("dur", 0), g1)
        if t2 > s2:
            segs.append((s2, t2, e))
            tids[e.get("tid")] += t2 - s2
            top[e.get("name", "?")[:80]] += t2 - s2
        i += 1
    segs.sort()
    cov = 0
    cur = g0
    for s2, t2, e in segs:
        if t2 <= cur:
            continue
        cov += t2 - max(s2, cur)
        cur = max(cur, t2)
    pct = cov / (g1 - g0) * 100
    tt = tids.most_common(2)
    tt_s = ",".join(f"tid{t}:{d/1000:.1f}ms" for t, d in tt)
    tops = top.most_common(3)
    tops_s = " | ".join(f"{n[:60]}={d/1000:.1f}ms" for n, d in tops)
    print(f"gap {gi:2d} {ms:5.1f}ms: python-cov={pct:5.1f}%  {tt_s}")
    if pct > 20:
        print(f"          top: {tops_s}")
