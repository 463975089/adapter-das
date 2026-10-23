# -*- coding: utf-8 -*-
"""P1.6 hit-list: elementwise kernels are graph-replayed (no corr ids), but
at::native elementwise kernel names carry the op functor in template args.
Parse functor + neighborhood anchors + grid (first occurrence). Counts are
contamination-proof."""
import gzip
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)

ev = [e for e in data["traceEvents"] if e.get("ph") == "X"]
kern = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
print(f"kernels {len(kern)}")

ELW = re.compile(r"vectorized_elementwise|elementwise_kernel|index_elementwise|CatArrayBatchedCopy|triton__|reduce_kernel|copy_kernel")

def functor(name: str) -> str:
    # e.g. void at::native::vectorized_elementwise_kernel<8, at::native::
    # (anonymous namespace)::BinaryFunctor...>, ... -> keep functor-ish tail
    m = re.search(r"::(\w+_functor|(\w+)Functor)", name)
    if m:
        return m.group(1)
    m = re.search(r"(add|mul|div|sub|copy|fill|sigmoid|silu|relu|exp|log|abs|neg|min|max|clamp|cast|compare|eq|lt|gt|ne|where|masked|gather|scatter|index|reduce|sum|amax|norm|rms|softmax|gelu|tanh|sqrt|rsqrt|reciprocal|bitwise|and|or|not)_kernel", name)
    if m:
        return m.group(1) + "_kernel"
    m = re.search(r"Functor<[^>]*,\s*void,\s*\w+::(\w+)>", name)
    if m:
        return m.group(1)
    return name[:48]

agg = defaultdict(lambda: [0, 0.0])
grid_first = {}
for i, e in enumerate(kern):
    nm = e.get("name", "")
    if not ELW.search(nm):
        continue
    if "vectorized_elementwise" not in nm and "elementwise_kernel" not in nm:
        continue
    key = (nm.split("<")[0].split("(")[0].split("::")[-1], functor(nm))
    agg[key][0] += 1
    agg[key][1] += e.get("dur", 0)
    if key not in grid_first:
        g = (e.get("args") or {}).get("grid") or "?"
        grid_first[key] = g

STEPS = 39
rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
print(f"== elementwise by parsed functor (per step, {STEPS} steps):")
tot = 0
for (kn, fn), (c, t) in rows[:36]:
    print(f"  {c:6d}x {c/STEPS:7.1f}/st {t/1000:7.1f}ms grid={grid_first[(kn,fn)]}  {fn}")
    tot += c
print(f"  TOTAL listed {tot} ({tot/STEPS:.0f}/step); families {len(rows)}")

# neighborhood anchor for the top functor only
top_key = rows[0][0]
ctx = defaultdict(int)
for i, e in enumerate(kern):
    nm = e.get("name", "")
    if ("vectorized_elementwise" not in nm and "elementwise_kernel" not in nm):
        continue
    if functor(nm) != top_key[1]:
        continue
    p = n = "?"
    for j in range(i - 1, max(0, i - 6), -1):
        pj = kern[j].get("name", "")
        if not ELW.search(pj) or "triton__" in pj:
            p = pj.split("(")[0].split("<")[0][-52:]
            break
    for j in range(i + 1, min(len(kern), i + 6)):
        nj = kern[j].get("name", "")
        if not ELW.search(nj) or "triton__" in nj:
            n = nj.split("(")[0].split("<")[0][-52:]
            break
    ctx[(p, n)] += 1
print(f"== top functor '{top_key[1]}' neighborhood anchors:")
for (p, n), c in sorted(ctx.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {c:6d}x  {p}  ->[it]->  {n}")
