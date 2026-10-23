# -*- coding: utf-8 -*-
"""Longer-name dump + lightop/paged_mqa census + uniformity check of the eager chain."""
import gzip
import json
import sys
from collections import Counter

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)
kern = sorted(
    (e for e in data["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"),
    key=lambda e: e["ts"],
)

names = Counter(e.get("name", "")[:80] for e in kern)
print("== mqa/lightop/paged-related kernels:")
for n, c in names.most_common():
    if any(k in n.lower() for k in ("mqa", "paged", "lightop", "indexer", "logits")):
        print(f"  {c:6d}x ({c/39:6.1f}/st)  {n}")

idx = [i for i, e in enumerate(kern) if "apply_rotary_emb" in e.get("name", "")]
print(f"\n== chain after rope #{idx[200]} (long names):")
c = idx[200]
for i in range(c, min(c + 44, len(kern))):
    e = kern[i]
    nm = e.get("name", "")
    print(f"  {i-c:+3d} {e.get('dur',0):6.1f}us {nm[:150]}")

# uniformity: count occurrences of softmax_warp_forward & Cijk gemms and their
# distance to nearest preceding rope
sw = [i for i, e in enumerate(kern) if "softmax_warp_forward" in e.get("name", "")]
gk = [i for i, e in enumerate(kern) if e.get("name", "").startswith("Cijk_")]
print(f"\nsoftmax_warp_forward {len(sw)} ({len(sw)/39:.1f}/st); Cijk gemms {len(gk)} ({len(gk)/39:.1f}/st)")
