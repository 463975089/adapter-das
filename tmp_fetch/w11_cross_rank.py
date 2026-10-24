"""W11h: cross-rank absolute-time alignment of graph-mode stall windows.
For TP-0's biggest gaps, check how many of TP-1..7 are ALSO gpu-idle in the
same absolute window, and print each rank's idle-overlap fraction.
Simultaneous on all 8 -> mutual wait (collective/semaphore chain).
Staggered -> local driver tick per rank.
"""
import gzip, json, glob, bisect, sys

folder = sys.argv[1] if len(sys.argv) > 1 else r"E:\GITS\hcu-sglang-adapter\prof_1024"
GAP_MS = 15.0
TOPN = 20

def load_busy(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    ev = [e for e in data["traceEvents"] if e.get("ph") == "X" and e.get("cat") in gpu_cats]
    ev.sort(key=lambda x: x["ts"])
    busy = []
    for e in ev:
        s, t = e["ts"], e["ts"] + e["dur"]
        if busy and s <= busy[-1][1]:
            busy[-1][1] = max(busy[-1][1], t)
        else:
            busy.append([s, t])
    return busy

def idle_frac(busy, a, b):
    # fraction of [a,b] not covered by busy
    cov = 0
    for s, t in busy:
        if t <= a:
            continue
        if s >= b:
            break
        cov += min(t, b) - max(s, a)
    return 1 - cov / (b - a)

ranks = {}
for path in sorted(glob.glob(folder + "/TP-*.trace.json.gz")):
    r = path.split("TP-")[-1].split(".")[0]
    ranks[r] = load_busy(path)
    print(f"TP-{r}: {len(ranks[r])} busy intervals")

b0 = ranks["0"]
gaps = []
for i in range(1, len(b0)):
    g0, g1 = b0[i - 1][1], b0[i][0]
    if g1 - g0 > GAP_MS * 1000:
        gaps.append((g1 - g0, g0, g1))
gaps.sort(reverse=True)
print(f"\nTP-0 gaps>{GAP_MS}ms: {len(gaps)}; top {TOPN} cross-rank idle fractions:")
for ms, g0, g1 in gaps[:TOPN]:
    fr = {r: idle_frac(ranks[r], g0, g1) for r in sorted(ranks)}
    allidle = sum(1 for v in fr.values() if v > 0.9)
    s = " ".join(f"{r}:{v:.2f}" for r, v in fr.items())
    print(f"  {ms/1000:6.1f}ms  idle≥90% ranks={allidle}/8   {s}")
