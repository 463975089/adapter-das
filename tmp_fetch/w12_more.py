"""W12: (1) hipGraphLaunch duration histogram — quantify the submission tick;
(2) per-rank local stall census -> Family A (all-rank idle) vs B (straggler,
others nccl-spin); straggler rotation table; (3) Family A resume cascade.
"""
import gzip, json, glob, collections, bisect

folder = r"E:\GITS\hcu-sglang-adapter\prof_1024"
GAP_MS = 15.0

def load(path, want_gpu=True, want_rt=False):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    out = {"gpu": [], "rt": []}
    for e in data["traceEvents"]:
        if e.get("ph") != "X":
            continue
        c = e.get("cat", "")
        if want_gpu and c in gpu_cats:
            out["gpu"].append(e)
        elif want_rt and c == "cuda_runtime":
            out["rt"].append(e)
    out["gpu"].sort(key=lambda x: x["ts"])
    return out

# --- (1) graph launch durations on TP-0
d0 = load(folder + "/TP-0.trace.json.gz", want_gpu=False, want_rt=True)
gls = [e for e in d0["rt"] if e["name"] == "hipGraphLaunch"]
h = collections.Counter()
for e in gls:
    ms = e.get("dur", 0) / 1000
    if ms < 1: h["<1ms"] += 1
    elif ms < 5: h["1-5ms"] += 1
    elif ms < 15: h["5-15ms"] += 1
    elif ms < 25: h["15-25ms"] += 1
    elif ms < 60: h["25-60ms"] += 1
    else: h[">60ms"] += 1
print(f"(1) TP-0 hipGraphLaunch n={len(gls)}: {dict(h)}")
vals = sorted(e.get('dur', 0) / 1000 for e in gls)
print(f"    top10 durs(ms): {[f'{v:.1f}' for v in vals[-10:]]}")

# --- (2)+(3) cross-rank gap census
ranks_gpu = {}
for path in sorted(glob.glob(folder + "/TP-*.trace.json.gz")):
    r = path.split("TP-")[-1].split(".")[0]
    ranks_gpu[r] = load(path)["gpu"]
    print(f"loaded TP-{r}: {len(ranks_gpu[r])} gpu events")

def busy_merge(ev):
    busy = []
    for e in ev:
        s, t = e["ts"], e["ts"] + e["dur"]
        if busy and s <= busy[-1][1]:
            busy[-1][1] = max(busy[-1][1], t)
        else:
            busy.append([s, t])
    return busy

busy = {r: busy_merge(ranks_gpu[r]) for r in ranks_gpu}

def idle_frac(r, a, b):
    cov = 0
    for s, t in busy[r]:
        if t <= a: continue
        if s >= b: break
        cov += min(t, b) - max(s, a)
    return 1 - cov / (b - a)

famA, famB = [], []
seen = set()
for r in sorted(ranks_gpu):
    bs = busy[r]
    for i in range(1, len(bs)):
        g0, g1 = bs[i - 1][1], bs[i][0]
        if g1 - g0 <= GAP_MS * 1000:
            continue
        key = (round(g0 / 50000), round(g1 / 50000))  # dedupe overlapping windows across ranks
        fr = {rr: idle_frac(rr, g0, g1) for rr in ranks_gpu}
        nidle = sum(1 for v in fr.values() if v > 0.8)
        if nidle >= 6:
            if key not in seen:
                seen.add(key)
                famA.append((g1 - g0, g0, g1))
        else:
            stragglers = [rr for rr, v in fr.items() if v > 0.8]
            famB.append(((g1 - g0) / 1000, r, stragglers, g0, g1))

famA.sort(key=lambda x: -x[0])
famB.sort(key=lambda x: -x[0])
print(f"\n(2) Family A (>=6 ranks idle) unique windows: {len(famA)}")
print(f"    Family B (local stall, others busy): {len(famB)} events")
rot = collections.Counter()
for ms, r, strs, g0, g1 in famB:
    rot[(r, tuple(sorted(strs)))] += 1
print("    straggler rotation (victim-rank -> fully-idle ranks):")
for k, n in rot.most_common(12):
    print(f"      stall-rank {k[0]} idle-ranks {k[1]}: {n}x")

# (3) Family A resume cascade
print(f"\n(3) Family A resume alignment:")
for ms, g0, g1 in famA[:8]:
    resumes = {}
    for r, ev in ranks_gpu.items():
        i = bisect.bisect_left([e["ts"] for e in ev], g1)
        if i < len(ev):
            resumes[r] = ev[i]["ts"]
    if not resumes:
        continue
    base = min(resumes.values())
    spread = max(resumes.values()) - base
    s = " ".join(f"{r}:{(t-base)/1000:+.1f}" for r, t in sorted(resumes.items()))
    print(f"  freeze {ms/1000:6.1f}ms -> resume spread {spread/1000:.1f}ms | {s}")
