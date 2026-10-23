# -*- coding: utf-8 -*-
"""Do the 116ms monster events overlap EACH OTHER on the same GPU?
Overlapping 'kernels' = impossible execution -> timestamp distortion
during a device freeze (both in flight when the freeze hit)."""
import gzip
import json
import sys

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)

mon = []
for e in data["traceEvents"]:
    if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
        d = e.get("dur", 0)
        if d > 20000:
            mon.append((e.get("ts", 0), e.get("ts", 0) + d, d, e.get("name", "")[:44]))

mon.sort()
n = len(mon)
print(f"monsters: {n}")
pairs = 0
self_ov = 0
for i in range(n):
    for j in range(i + 1, n):
        if mon[j][0] >= mon[i][1]:
            break
        pairs += 1
print(f"overlapping monster pairs: {pairs}")
if pairs:
    shown = 0
    for i in range(n):
        for j in range(i + 1, n):
            if mon[j][0] >= mon[i][1]:
                break
            print(f"  ov: [{mon[i][0]:.0f}+{mon[i][2]/1000:.0f}ms {mon[i][3]}] x "
                  f"[{mon[j][0]:.0f}+{mon[j][2]/1000:.0f}ms {mon[j][3]}]")
            shown += 1
            if shown >= 10:
                sys.exit(0)
