# -*- coding: utf-8 -*-
"""Wave6 recount with profiler-boundary artifact filtering.

Monster events (~116ms) appear across ALL families at the profile stop
boundary -- drop dur>20ms, audit where/when they cluster, recompute honest
family totals, GPU util and gaps.
"""
import gzip
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)

ev = data["traceEvents"]
kern = []
for e in ev:
    if e.get("ph") != "X":
        continue
    if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
        kern.append((e.get("ts", 0), e.get("dur", 0), e.get("name", "")))
kern.sort()

# --- monster audit: dur>20ms events, their ts positions relative to span
t0 = kern[0][0]
t1 = max(e[0] + e[1] for e in kern)
monsters = [(ts, d, nm) for ts, d, nm in kern if d > 20000]
mtot = sum(d for _, d, _ in monsters)
mts = sorted(ts for ts, _, _ in monsters)
print(f"span {(t1-t0)/1e6:.2f}s; monsters(>20ms) n={len(monsters)} "
      f"total {mtot/1000:.0f}ms = {mtot/(t1-t0)*100:.1f}% of span")
if monsters:
    half = (t1 - t0) / 2
    late = sum(1 for ts, _, _ in monsters if ts - t0 > half)
    print(f"  monster ts quantiles (s into span): "
          f"{[round((ts-t0)/1e6,2) for ts in mts[:8]]} ... late-half {late}/{len(monsters)}")

# --- filtered census
def fam(name):
    n = re.sub(r"\(\d+\)", "", name)
    n = re.sub(r"<.*", "", n)
    n = re.sub(r"\d+", "N", n)
    return n[:70]

fams = defaultdict(lambda: [0, 0.0, 0.0])  # count, total_kept, total_dropped
for ts, d, nm in kern:
    f = fams[fam(nm)]
    f[0] += 1
    if d > 20000:
        f[2] += d
    else:
        f[1] += d

STEPS = 39
tot_kept = sum(v[1] for v in fams.values())
tot_drop = sum(v[2] for v in fams.values())
print(f"\nfiltered: kept {tot_kept/1000:.0f}ms, dropped {tot_drop/1000:.0}fms" if False else
      f"\nfiltered: kept {tot_kept/1000:.0f}ms ({tot_kept/1000/STEPS:.1f}ms/step), "
      f"dropped {tot_drop/1000:.0f}ms as artifacts")
print(f"== top 24 families by KEPT total (count, ms/step, avg_us, p50-ish):")
for name, (c, t, drep) in sorted(fams.items(), key=lambda kv: -kv[1][1])[:24]:
    print(f"  {t/1000/STEPS:7.2f}ms/step {c:6d}x {t/c:7.1f}us  dropped {drep/1000:6.1f}ms  {name}")

# --- filtered gaps: clip monster durs at 1us for gap computation
span2 = [(ts, min(d, 1)) for ts, d, nm in kern]
cur_end = None
gaps = []
for ts, d in span2:
    if cur_end is not None and ts > cur_end:
        gaps.append((ts - cur_end, cur_end))
    cur_end = max(cur_end or ts, ts + d)
big = [(g, s) for g, s in gaps if g > 2000]
bigtot = sum(g for g, _ in big)
print(f"\ngaps>2ms (filtered): {len(big)} totaling {bigtot/1000:.0f}ms "
      f"({bigtot/1000/STEPS:.1f}ms/step)")
h = defaultdict(int)
for g, s in big:
    b = int(g // 10000) * 10
    h[b] += 1
print("  gap size histogram (10ms buckets -> count):",
      dict(sorted(h.items())))
print("  top-8 gaps: " + "; ".join(f"{g/1000:.0f}ms@+{(s-t0)/1000:.0f}" for g, s in sorted(big, reverse=True)[:8]))
