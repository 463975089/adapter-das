# -*- coding: utf-8 -*-
"""Are the 8 scheduler processes' hipGraphLaunch host calls serialized in
wall-clock (global driver lock) or concurrent?  Same-host monotonic ts."""
import gzip
import json
import sys

def launches(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    out = []
    for e in data["traceEvents"]:
        if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "runtime", "hip_runtime") \
                and "raphLaunch" in e.get("name", ""):
            out.append((e.get("ts", 0), e.get("dur", 0)))
    out.sort()
    return out

import re as _re
paths = sys.argv[1:]
L = {}
for p in paths:
    tag = _re.search(r"tp(\d+)", p).group(0)
    L[tag] = launches(p)

merged = []
for tag, lst in L.items():
    for ts, d in lst:
        merged.append((ts, ts + d, tag))
merged.sort()

# overlap: any interval of tag A overlapping tag B
overlap_pairs = 0
total_pairs = 0
for i in range(len(merged)):
    for j in range(i + 1, len(merged)):
        if merged[j][0] >= merged[i][1]:
            break
        if merged[i][2] != merged[j][2]:
            overlap_pairs += 1
# coverage: fraction of wall time covered by >=2 launches
t0, t1 = merged[0][0], max(e for _, e, _ in merged)
events = []
for s, e, _ in merged:
    events.append((s, 1))
    events.append((e, -1))
events.sort()
depth = 0
cov2 = 0
last = events[0][0]
for t, dv in events:
    if depth >= 2:
        cov2 += t - last
    depth += dv
    last = t
covered = sum(e - s for s, e, _ in merged)
print(f"launches per rank: { {t: len(v) for t, v in L.items()} }")
durs = sorted(d for v in L.values() for _, d in v)
print(f"launch dur ms: p10={durs[len(durs)//10]/1000:.1f} p50={durs[len(durs)//2]/1000:.1f} "
      f"p90={durs[len(durs)*9//10]/1000:.1f} max={durs[-1]/1000:.1f}")
print(f"cross-rank overlapping launch pairs: {overlap_pairs}")
print(f"wall span {(t1-t0)/1e6:.1f}s; launch-covered {covered/1e6:.2f}s "
      f"({covered/(t1-t0)*100:.0f}%); depth>=2 time {cov2/1e6:.3f}s")

# show a sample window: first 12 launches timeline
print("\nfirst 12 launches (start_ms_rel, dur_ms, rank):")
base = merged[0][0]
for s, e, tag in merged[:12]:
    print(f"  +{(s-base)/1000:9.1f}ms  {(e-s)/1000:6.1f}ms  {tag}")
