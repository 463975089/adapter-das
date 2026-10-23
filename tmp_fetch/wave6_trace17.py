# -*- coding: utf-8 -*-
"""Attribute the per-layer eager elementwise soup via python_function stacks.
Find python frames whose (file:line) repeats ~44/step, and the leaf stacks of
cpu ops like exp/log/softmax/masked_fill/arange/clamp."""
import gzip
import json
import sys
from collections import Counter, defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"]
py = [e for e in ev if e.get("cat") == "python_function" and e.get("ph") == "X"]
py.sort(key=lambda e: e["ts"])
print(f"python_function events: {len(py)}")

# 1) hot frames by (name, line): count, total dur
hot = defaultdict(lambda: [0, 0.0])
for e in py:
    nm = e.get("name", "")
    if not nm or "sglang" not in str(e.get("args", {}).get("..." , "")):
        pass
    key = nm.split(" (")[-1].rstrip(")") if " (" in nm else nm
    hot[key][0] += 1
    hot[key][1] += e.get("dur", 0)
rows = sorted(hot.items(), key=lambda kv: -kv[1][0])[:40]
S = 39
print("== hottest python frames containing sglang paths (count/step):")
shown = 0
for k, (c, t) in rows:
    if "sglang" not in k:
        continue
    print(f"  {c:6d}x {c/S:6.1f}/st {t/1000:8.1f}ms  {k[:110]}")
    shown += 1
    if shown >= 30:
        break
