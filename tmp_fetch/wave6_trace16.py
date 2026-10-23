# -*- coding: utf-8 -*-
"""Full family census: normalized kernel names, count/step, total dur."""
import gzip
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)
kern = [e for e in data["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
fam = defaultdict(lambda: [0, 0.0])
for e in kern:
    nm = e.get("name", "")
    # normalize: drop template args, keep distinguishing functor
    m = re.match(r"([A-Za-z_][\w:]*)", nm)
    base = m.group(1) if m else nm[:40]
    fn = re.search(r"::(\w+_?functor\w*|(\w+)Functor)", nm, re.I)
    key = base if not fn else base + "|" + fn.group(1)
    fam[key][0] += 1
    fam[key][1] += e.get("dur", 0)
S = 39
rows = sorted(fam.items(), key=lambda kv: -kv[1][0])
print(f"families {len(rows)}, kernels {len(kern)}")
for k, (c, t) in rows[:70]:
    print(f"  {c:6d}x {c/S:7.1f}/st {t/1000:8.1f}ms  {k[:110]}")
