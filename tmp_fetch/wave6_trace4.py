# -*- coding: utf-8 -*-
"""Wave6 gap/monster anatomy: do >50ms GPU gaps overlap monster-duration
kernels (kernel resident during a device nap), and where do gaps sit
relative to host graph-launch calls?"""
import gzip
import json
import sys
import bisect

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)

ev = data["traceEvents"]
kern = []
launches = []  # host hipGraphLaunch ts
for e in ev:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "")
    if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        kern.append((e.get("ts", 0), e.get("dur", 0), e.get("name", "")))
    elif cat in ("cuda_runtime", "runtime", "hip_runtime") and "raphLaunch" in e.get("name", ""):
        launches.append((e.get("ts", 0), e.get("dur", 0)))

kern.sort()
launches.sort()
lt = [l[0] for l in launches]
t0 = kern[0][0]

# monsters: dur>20ms
monsters = [(ts, ts + d, nm, d) for ts, d, nm in kern if d > 20000]

# gaps from clipped timeline
span2 = [(ts, min(d, 1)) for ts, d, nm in kern]
cur_end = None
gaps = []
for ts, d in span2:
    if cur_end is not None and ts > cur_end:
        gaps.append((cur_end, ts))  # (start, end)
    cur_end = max(cur_end or ts, ts + d)
big = [(s, e) for s, e in gaps if e - s > 50000]

print(f"gaps>50ms: {len(big)}; monsters: {len(monsters)}")

# overlap stats
ov = 0
for gs, ge in big:
    if any(ms < ge and me > gs for ms, me, _, _ in monsters):
        ov += 1
print(f"gaps overlapping a monster kernel window: {ov}/{len(big)}")

# per-gap classification: monster overlap + distance to nearest launch call
print(f"\n{'gap_ms':>8} {'at_+s':>7} {'mon_ovl':>7} {'to_prev_launch_ms':>17} {'to_next_launch_ms':>17} monster_names")
rows = []
for gs, ge in big:
    i = bisect.bisect_left(lt, gs)
    prev = lt[i - 1] if i > 0 else None
    nxt = lt[i] if i < len(lt) else None
    dp = (gs - prev) / 1000 if prev is not None else float("nan")
    dn = (nxt - ge) / 1000 if nxt is not None else float("nan")
    names = [nm[:38] for ms, me, nm, d in monsters if ms < ge and me > gs][:2]
    rows.append((ge - gs, gs, names, dp, dn))
rows.sort(key=lambda r: -r[0])
for gms, gs, names, dp, dn in rows[:24]:
    print(f"{gms/1000:8.1f} {(gs-t0)/1e6:7.2f} {'Y' if names else '-':>7} "
          f"{dp:17.1f} {dn:17.1f}  {','.join(names)}")

# how many big gaps start within 30ms after a launch call vs elsewhere
import statistics
near_prev = sum(1 for gms, gs, n, dp, dn in rows if dp is not None and dp < 30)
near_next = sum(1 for gms, gs, n, dp, dn in rows if dn is not None and dn < 30)
far = len(rows) - near_prev - near_next
print(f"\ngap-start <30ms after a launch call: {near_prev}; "
      f"gap-end <30ms before next launch: {near_next}; elsewhere: {far}")

# monster names histogram
from collections import Counter
mc = Counter(nm[:50] for _, _, nm, _ in monsters)
print("\nmonster kernel names:", dict(mc.most_common(8)))
# monster durs p50
md = sorted(d for _, _, _, d in monsters)
print(f"monster dur ms: p10={md[len(md)//10]/1000:.0f} p50={md[len(md)//2]/1000:.0f} "
      f"p90={md[len(md)*9//10]/1000:.0f}")
