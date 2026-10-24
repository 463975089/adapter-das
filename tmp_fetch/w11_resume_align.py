"""W11j: for the all-rank simultaneous freeze windows (TP-0 60.5ms family),
print each rank's first GPU event timestamp after the window end.
Simultaneous resume (us..1ms spread) -> one shared release event.
Staggered by ~19.4ms multiples -> independent per-rank ticks.
"""
import gzip, json, glob, bisect

folder = r"E:\GITS\hcu-sglang-adapter\prof_1024"
GAP_MS = 15.0

def load_gpu_sorted(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    ev = [e for e in data["traceEvents"] if e.get("ph") == "X" and e.get("cat") in gpu_cats]
    ev.sort(key=lambda x: x["ts"])
    return ev

ranks = {}
for path in sorted(glob.glob(folder + "/TP-*.trace.json.gz")):
    r = path.split("TP-")[-1].split(".")[0]
    ranks[r] = load_gpu_sorted(path)

def busy_merge(ev):
    busy = []
    for e in ev:
        s, t = e["ts"], e["ts"] + e["dur"]
        if busy and s <= busy[-1][1]:
            busy[-1][1] = max(busy[-1][1], t)
        else:
            busy.append([s, t])
    return busy

busy = {r: busy_merge(ranks[r]) for r in ranks}
gaps = []
b0 = busy["0"]
for i in range(1, len(b0)):
    g0, g1 = b0[i - 1][1], b0[i][0]
    if g1 - g0 > GAP_MS * 1000:
        # only the simultaneous-freeze family: all ranks mostly idle
        def idlefrac(r):
            bs = busy[r]
            cov = 0
            for s, t in bs:
                if t <= g0: continue
                if s >= g1: break
                cov += min(t, g1) - max(s, g0)
            return 1 - cov / (g1 - g0)
        fr = [idlefrac(r) for r in ranks]
        if min(fr) > 0.85:
            gaps.append((g1 - g0, g0, g1))
print(f"simultaneous-freeze windows: {len(gaps)}")
for ms, g0, g1 in gaps[:12]:
    resumes = {}
    for r, ev in ranks.items():
        i = bisect.bisect_left([e["ts"] for e in ev], g1 - 500)
        while i < len(ev) and ev[i]["ts"] < g1:
            i += 1
        if i < len(ev):
            resumes[r] = (ev[i]["ts"], ev[i].get("name", "?")[:44])
    if not resumes:
        continue
    base = min(t for t, _ in resumes.values())
    spread = max(t for t, _ in resumes.values()) - base
    detail = "  ".join(f"TP-{r}:{(t-base)/1000:+.2f}ms {n[:28]}" for r, (t, n) in sorted(resumes.items()))
    print(f"\nfreeze {ms/1000:.1f}ms -> resume spread {spread/1000:.2f}ms")
    print(f"   {detail}")
