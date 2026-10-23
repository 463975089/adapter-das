# -*- coding: utf-8 -*-
"""In-container quick census for the P0'/R-20 window trace.
Writes a verdict block to stdout: steps, kernels/step, family table,
>20ms stalls (P0'), R-20 engagement counters."""
import glob
import gzip
import json
import re
import sys
from collections import defaultdict

paths = sorted(glob.glob("/root/prof_new/*.trace.json.gz"))
if not paths:
    print("NO TRACE FILES")
    sys.exit(1)
path = paths[0]
with gzip.open(path, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"]

steps = sum(1 for e in ev if e.get("cat") == "user_annotation"
            and str(e.get("name", "")).startswith("ProfilerStep"))
kern = [e for e in ev if e.get("cat") == "kernel"]
stalls = [e for e in kern if e.get("dur", 0) > 20000]
S = steps or 39

fam = defaultdict(lambda: [0, 0.0])
for e in kern:
    nm = e.get("name", "")
    m = re.match(r"([A-Za-z_][\w:]*)", nm)
    fn = re.search(r"::(\w+_?functor\w*|(\w+)Functor)", nm, re.I)
    key = (m.group(1) if m else nm[:36]) + (("|" + fn.group(1)) if fn else "")
    fam[key][0] += 1
    fam[key][1] += e.get("dur", 0)

r20 = sum(1 for e in kern if "_ragged_mqa_logits_kernel" in e.get("name", ""))
w10 = sum(1 for e in kern if "_moe_align_fused_small_kernel" in e.get("name", ""))
w10q = sum(1 for e in kern if "_moe_align_hist_kernel" in e.get("name", ""))
fills = sum(1 for e in kern if "FillFunctor" in e.get("name", ""))
bbh = sum(1 for e in kern if e.get("name", "").startswith("Cijk_") and "_BBH_" in e.get("name", ""))
dtou = sum(1 for e in ev if e.get("cat") == "gpu_memcpy"
           and (e.get("args", {}) or {}).get("name", "").upper().find("D2H") >= 0)

print(f"=== CENSUS {path.split('/')[-1]}")
print(f"steps={steps} kernels={len(kern)} per_step={len(kern)/S:.1f}")
print(f"STALLS>20ms: {len(stalls)} (P0' verdict: expect 0)")
for e in stalls[:5]:
    print(f"   stall {e.get('dur')/1000:.1f}ms {e.get('name','')[:60]}")
print(f"R20 engaged: _ragged_mqa_logits_kernel={r20} ({r20/S:.1f}/step, expect ~44)")
print(f"W10 engaged: fused_moe_align={w10} ({w10/S:.1f}/step, expect ~44) "
      f"quartet_hist_remnant={w10q} ({w10q/S:.1f}/step, expect 0)")
print(f"FillFunctor: {fills} ({fills/S:.1f}/step; wave6 had ~134 incl. 88 moe_align setup)")
print(f"emu-remnant Cijk_BBH gemms: {bbh} ({bbh/S:.1f}/step; wave6 was ~47)")
print(f"DtoH copies: {dtou} ({dtou/S:.1f}/step)")
print("top families:")
for k, (c, t) in sorted(fam.items(), key=lambda kv: -kv[1][0])[:25]:
    print(f"   {c:6d}x {c/S:7.1f}/st {t/1000:8.1f}ms  {k[:100]}")
tot_ms = sum(e.get("dur", 0) for e in kern) / 1000 / S
print(f"total kernel time/step: {tot_ms:.1f}ms (wave6 contaminated: ~3750ms; clean target)")
