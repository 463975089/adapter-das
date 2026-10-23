# -*- coding: utf-8 -*-
"""Neighborhood anchors for the per-layer (44/step) elementwise families."""
import gzip
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)

kern = sorted(
    (e for e in data["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"),
    key=lambda e: e["ts"],
)

def functor(name: str) -> str:
    m = re.search(r"::(\w+_functor|(\w+)Functor)", name)
    if m:
        return m.group(1)
    m = re.search(r"(add|mul|div|sub|copy|fill|sigmoid|silu|relu|exp|log|abs|neg|min|max|clamp|cast|compare|eq|lt|gt|ne|where|masked|gather|scatter|index|reduce|sum|amax|norm|rms|softmax|gelu|tanh|sqrt|rsqrt|reciprocal|bitwise|and|or|not)_kernel", name)
    return m.group(1) + "_kernel" if m else None

def is_elw(n):
    return "elementwise" in n or "copy_kernel" in n or "reduce_kernel" in n or "scatter_gather" in n

def short(n):
    return n.split("(")[0].split("<")[0].split("::")[-1][-48:]

def anchor(i, step, stop):
    j = i
    while 0 <= j < len(kern) and abs(j - i) <= 8:
        nj = kern[j].get("name", "")
        if not is_elw(nj):
            return short(nj)
        j += step if stop else -step
    return "?"

for want in ["exp_kernel", "log_kernel", "AbsFunctor", "FillFunctor", "fill_kernel",
             "not_kernel", "CompareFunctor", "index_kernel", "BinaryFunctor",
             "AUnaryFunctor", "BUnaryFunctor", "CUDAFunctor"]:
    ctx = defaultdict(int)
    for i, e in enumerate(kern):
        nm = e.get("name", "")
        if not is_elw(nm):
            continue
        if functor(nm) != want:
            continue
        ctx[(anchor(i, -1, False), anchor(i, 1, True))] += 1
    top = sorted(ctx.items(), key=lambda kv: -kv[1])[:6]
    tot = sum(ctx.values())
    print(f"== {want}  ({tot}x, {tot/39:.1f}/step)")
    for (p, n), c in top:
        print(f"   {c:5d}x  {p}  ->[it]->  {n}")
