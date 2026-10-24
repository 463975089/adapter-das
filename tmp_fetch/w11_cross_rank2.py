"""W11i: during TP-0's top stall gaps, WHAT are TP-1..7 running?
If ncclDevKernel dominates their busy time -> they are spin-waiting for TP-0
in a collective: TP-0 is the straggler; the stall is TP-0-local but gates all.
Also verify trace clock alignment via ts ranges.
"""
import gzip, json, glob, collections

folder = r"E:\GITS\hcu-sglang-adapter\prof_1024"
GAP_MS = 15.0
TOPN = 8

def load_gpu(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    ev = [e for e in data["traceEvents"] if e.get("ph") == "X" and e.get("cat") in gpu_cats]
    ev.sort(key=lambda x: x["ts"])
    return ev

def busy_merge(ev):
    busy = []
    for e in ev:
        s, t = e["ts"], e["ts"] + e["dur"]
        if busy and s <= busy[-1][1]:
            busy[-1][1] = max(busy[-1][1], t)
        else:
            busy.append([s, t])
    return busy

ranks_ev, ranges = {}, {}
for path in sorted(glob.glob(folder + "/TP-*.trace.json.gz")):
    r = path.split("TP-")[-1].split(".")[0]
    ev = load_gpu(path)
    ranks_ev[r] = ev
    ranges[r] = (ev[0]["ts"], ev[-1]["ts"] + ev[-1]["dur"])
print("trace ts ranges (ms offsets vs TP-0 start):")
t00 = ranges["0"][0]
for r in sorted(ranges):
    a, b = ranges[r]
    print(f"  TP-{r}: {(a-t00)/1000:+9.1f}ms .. {(b-t00)/1000:+9.1f}ms  span={(b-a)/1e6:.2f}s")

busy = {r: busy_merge(ranks_ev[r]) for r in ranks_ev}
b0 = busy["0"]
gaps = []
for i in range(1, len(b0)):
    g0, g1 = b0[i - 1][1], b0[i][0]
    if g1 - g0 > GAP_MS * 1000:
        gaps.append((g1 - g0, g0, g1))
gaps.sort(reverse=True)

import bisect
for ms, g0, g1 in gaps[:TOPN]:
    print(f"\nTP-0 stall {ms/1000:.1f}ms:")
    for r in ["1", "2", "3"]:
        ev = ranks_ev[r]
        i = bisect.bisect_left([e["ts"] for e in ev], g0 - 2000)
        tot = collections.Counter()
        n = 0
        while i < len(ev) and ev[i]["ts"] <= g1:
            e = ev[i]
            ov = min(e["ts"] + e["dur"], g1) - max(e["ts"], g0)
            if ov > 0:
                nm = e.get("name", "?")
                key = "ncclDevKernel" if "nccl" in nm else \
                      ("allgather/crossdev" if any(k in nm for k in ("allgather", "cross_device", "reduce")) else
                       ("memcpy" if e.get("cat") == "gpu_memcpy" else "other:" + nm[:36]))
                tot[key] += ov / 1000.0
                n += 1
            i += 1
        w = (g1 - g0) / 1000.0
        top = ", ".join(f"{k}={v:.1f}ms" for k, v in tot.most_common(4))
        print(f"  TP-{r}: {n} ev, busy-by-kind: {top}  (window {w:.1f}ms)")
