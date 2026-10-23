# -*- coding: utf-8 -*-
"""Dump the raw kernel sequence between two rope kernels (one site, full names)."""
import gzip
import json
import sys

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)
kern = sorted(
    (e for e in data["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"),
    key=lambda e: e["ts"],
)
idx = [i for i, e in enumerate(kern) if "apply_rotary_emb" in e.get("name", "")]
print(f"rope kernels: {len(idx)} ({len(idx)/39:.1f}/step)")
# take a window well inside the trace: rope #200, print +-40 kernels
c = idx[200]
for i in range(c - 6, min(c + 46, len(kern))):
    e = kern[i]
    nm = e.get("name", "")
    short = nm if len(nm) < 96 else nm[:93] + "..."
    d = e.get("dur", 0)
    mark = " <<<" if i == c else ""
    print(f"{i-c:+4d} {d:7.1f}us  {short}{mark}")
