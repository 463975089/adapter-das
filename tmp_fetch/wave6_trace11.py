# -*- coding: utf-8 -*-
"""Attribute the elementwise GPU kernel family to their launching aten ops
via profiler correlation ids. Counts are contamination-proof; this yields
the P1.6 fusion hit-list."""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)

ev = data["traceEvents"]
kern_by_corr = {}
cpu_by_corr = {}
kern = []
for e in ev:
    if e.get("ph") != "X":
        continue
    a = e.get("args") or {}
    cid = a.get("correlation") or a.get("correlation_id") or a.get("External id")
    cat = e.get("cat", "")
    if cat == "kernel":
        kern.append(e)
        if cid is not None:
            kern_by_corr[cid] = e.get("name", "")
    elif cat == "cpu_op" and cid is not None:
        cpu_by_corr[cid] = e.get("name", "")

print(f"kernels {len(kern)}; with corr {len(kern_by_corr)}; cpu ops with corr {len(cpu_by_corr)}")

FAMS = ("vectorized_elementwise", "elementwise_kernel_manual_unroll",
        "elementwise_kernel_with_index", "index_elementwise",
        "CatArrayBatchedCopy", "reduce_kernel", "indexSelectSmallIndex")
agg = defaultdict(lambda: [0, 0.0])
orphan = defaultdict(int)
for e in kern:
    nm = e.get("name", "")
    if not any(f in nm for f in FAMS):
        continue
    a = e.get("args") or {}
    cid = a.get("correlation") or a.get("correlation_id") or a.get("External id")
    src = cpu_by_corr.get(cid) if cid is not None else None
    if src is None:
        orphan[nm[:40]] += 1
    key = (nm.split("<")[0].split("(")[0][:34], src or "?")
    agg[key][0] += 1
    agg[key][1] += e.get("dur", 0)

STEPS = 39
print(f"== elementwise/copy families by (kernel, launching aten op):")
for (kn, src), (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:30]:
    print(f"  {c:6d}x {c/STEPS:6.1f}/step {t/1000:7.1f}ms  {kn}  <-  {src[:60]}")
print("== no-corr orphans:", dict(list(orphan.items())[:6]))
