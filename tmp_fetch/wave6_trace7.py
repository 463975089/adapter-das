# -*- coding: utf-8 -*-
"""During the 117ms GPU stall windows, what was the host CPU doing?
If CPU events also gap -> whole-process freeze (host/global cause).
If CPU busy -> GPU-only freeze (device/driver cause)."""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)

ev = data["traceEvents"]
kern = []
cpu = []  # (ts, dur, name, tid)
for e in ev:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "")
    if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        kern.append((e.get("ts", 0), e.get("dur", 0)))
    elif cat in ("cpu_op", "python_function", "user_annotation", "cuda_runtime", "runtime", "hip_runtime"):
        cpu.append((e.get("ts", 0), e.get("dur", 0), e.get("name", ""), e.get("tid", 0)))

kern.sort()
span2 = [(ts, min(d, 1)) for ts, d in kern]
cur_end = None
gaps = []
for ts, d in span2:
    if cur_end is not None and ts > cur_end and ts - cur_end > 100000:  # >100ms
        gaps.append((cur_end, ts))
    cur_end = max(cur_end or ts, ts + d)
gaps.sort(key=lambda g: -(g[1] - g[0]))
top = [g for g in gaps if g[1] - g[0] > 100000 and 100 < (g[1]-g[0])/1000 < 130][:12]
print(f"gaps 100-130ms: {len(top)} sampled")

# merged CPU coverage per window
iv = sorted((ts, ts + d) for ts, d, nm, tid in cpu if d > 0)
merged = []
for s, t in iv:
    if merged and s <= merged[-1][1]:
        if t > merged[-1][1]:
            merged[-1][1] = t
    else:
        merged.append([s, t])

import bisect
def cpu_busy(s, e):
    i = bisect.bisect_left(merged, [s + 1e18])
    # scan back
    j = bisect.bisect_right([m[1] for m in merged], s)
    tot = 0
    for ms, mt in merged[max(0, j - 2):i]:
        if mt <= s:
            continue
        if ms >= e:
            break
        tot += min(mt, e) - max(ms, s)
    return tot

print(f"{'gap_ms':>8} {'cpu_busy_frac':>13}  top CPU ops in window")
for gs, ge in top:
    w = ge - gs
    b = cpu_busy(gs, ge)
    ops = defaultdict(float)
    for ts, d, nm, tid in cpu:
        if ts >= ge:
            break
        if ts + d > gs and d > 0:
            ov = min(ts + d, ge) - max(ts, gs)
            if ov > 0:
                ops[nm[:52]] += ov
    tops = sorted(ops.items(), key=lambda kv: -kv[1])[:3]
    tops = "; ".join(f"{t/1000:.0f}ms {n}" for n, t in tops)
    print(f"{w/1000:8.1f} {b/w:13.2f}  {tops[:110]}")
