# -*- coding: utf-8 -*-
"""Sample grid/block of silu + moe_sum + mix_stats events at trace-middle
(steady decode) vs first event (capture/warmup)."""
import gzip
import json
import sys

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)

targets = {
    "_silu_clamp_ptgq": [],
    "_moe_sum_scale": [],
    "_hc_mix_stats_partial": [],
    "_router_triton": [],
    "fused_moe_kernel": [],
}
evs = [e for e in data["traceEvents"]
       if e.get("ph") == "X" and e.get("cat") == "kernel"]
evs.sort(key=lambda e: e.get("ts", 0))
for e in evs:
    nm = e.get("name", "")
    for t in targets:
        if t in nm:
            a = e.get("args") or {}
            targets[t].append((e.get("dur", 0), a.get("grid"), a.get("block")))
            break

for t, rows in targets.items():
    print(f"== {t}  n={len(rows)}")
    m = len(rows)
    for idx in sorted({0, m // 4, m // 2, (3 * m) // 4, m - 1}):
        d, grid, block = rows[idx]
        print(f"   [{idx:5d}] dur={d/1000:8.1f}us grid={grid} block={block}")
