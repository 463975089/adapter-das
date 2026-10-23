# -*- coding: utf-8 -*-
"""Cross-rank stall correlation: extract >50ms GPU gap windows from several
rank traces (same host, CLOCK_MONOTONIC-comparable ts) and test pairwise
coincidence.  Synchronized stalls -> global cause (host/driver/power event);
independent stalls -> per-GPU cause."""
import gzip
import json
import sys

TOL = 15000  # 15ms tolerance for "coincident"


def gap_windows(path, thresh=50000):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    kern = []
    for e in data["traceEvents"]:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            kern.append((e.get("ts", 0), min(e.get("dur", 0), 1)))
    kern.sort()
    gaps = []
    cur_end = None
    for ts, d in kern:
        if cur_end is not None and ts > cur_end and ts - cur_end > thresh:
            gaps.append((cur_end, ts))
        cur_end = max(cur_end or ts, ts + d)
    return gaps


paths = sys.argv[1:]
gaps = {}
for p in paths:
    g = gap_windows(p)
    tag = p.split("/")[-1][:34]
    gaps[tag] = g
    tot = sum(e - s for s, e in g) / 1000
    q117 = sum(1 for s, e in g if 100 < (e - s) / 1000 < 135)
    print(f"{tag}: gaps>50ms n={len(g)} total={tot:.0f}ms ~117ms-quanta={q117}")

print(f"\npairwise coincidence (gap-start within +-{TOL//1000}ms):")
tags = list(gaps)
for i in range(len(tags)):
    for j in range(i + 1, len(tags)):
        a, b = gaps[tags[i]], gaps[tags[j]]
        hit = 0
        for s, e in a:
            for s2, e2 in b:
                if abs(s - s2) < TOL or abs(s - e2) < TOL or abs(e - s2) < TOL:
                    hit += 1
                    break
        pct = hit / len(a) * 100 if a else 0
        print(f"  {tags[i][:20]:20s} vs {tags[j][:20]:20s}: {hit}/{len(a)} = {pct:.0f}% coincident")

# absolute ts of first 5 big gaps per rank, to eyeball alignment
print("\nfirst 4 gap-start ts (us) per rank:")
for tag in tags:
    print(f"  {tag}: {[int(s) for s, _ in gaps[tag][:4]]}")
