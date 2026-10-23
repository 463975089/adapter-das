# -*- coding: utf-8 -*-
"""Wave6 fresh-trace census (TP0, DSPARK decode, 40 steps, bs2/rank).

Metrics:
  1. step count via CUDA/ROCm graph-launch runtime calls
  2. kernels/step (all GPU kernel events / steps)
  3. DtoH memcpy per step (P1.5 acceptance) + all memcpy families
  4. top kernel families by total time (1.6 baseline)
  5. GPU idle gaps > 5ms over the busy span
"""
import gzip
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)

ev = data["traceEvents"]
kern = []
memcpy = []
runtime = defaultdict(lambda: [0, 0.0])
for e in ev:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "")
    if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        rec = (e.get("ts", 0), e.get("dur", 0), e.get("name", ""), cat)
        (memcpy if cat == "gpu_memcpy" else kern).append(rec)
    elif cat in ("cuda_runtime", "runtime", "hip_runtime"):
        r = runtime[e.get("name", "")]
        r[0] += 1
        r[1] += e.get("dur", 0.0)

gl_names = [n for n in runtime if "raphLaunch" in n]
gl_count = sum(runtime[n][0] for n in gl_names)
sync_names = [n for n in runtime if "Synchronize" in n or "synchronize" in n]

print(f"total events: {len(ev)}  kernel/memset: {len(kern)}  memcpy: {len(memcpy)}")
print(f"graph launches ({','.join(gl_names)}): {gl_count}")
for n in sorted(sync_names):
    print(f"  sync: {n}  x{runtime[n][0]}  {runtime[n][1]/1000:.1f}ms total")

if gl_count:
    print(f"kernels/step (all/{gl_count}): {len(kern)/gl_count:.1f}")

# memcpy families
mm = defaultdict(lambda: [0, 0.0])
for _, d, name, _ in memcpy:
    mm[name][0] += 1
    mm[name][1] += d
print("== memcpy families (count, total_ms, per-step count):")
for name, (c, t) in sorted(mm.items(), key=lambda kv: -kv[1][1]):
    ps = c / gl_count if gl_count else 0
    print(f"  {t/1000:9.1f}ms {c:6d}x  {ps:6.1f}/step  {name[:90]}")

# kernel family normalization
def fam(name):
    n = re.sub(r"\(\d+\)", "", name)          # kernel(N)
    n = re.sub(r"<.*", "", n)                  # templates
    n = re.sub(r"\d+", "N", n)                 # digits
    n = n[:72]
    return n

fams = defaultdict(lambda: [0, 0.0])
for _, d, name, _ in kern:
    fams[fam(name)][0] += 1
    fams[fam(name)][1] += d

tot_k = sum(v[1] for v in fams.values())
print(f"== top 30 kernel families (of {len(fams)}), total GPU kernel time {tot_k/1000:.1f}ms:")
for name, (c, t) in sorted(fams.items(), key=lambda kv: -kv[1][1])[:30]:
    ps = c / gl_count if gl_count else 0
    print(f"  {t/1000:8.1f}ms {c:6d}x {ps:5.1f}/step {t/c:7.1f}us  {name}")

# softmax port presence
for name, (c, t) in fams.items():
    if "softmax_rows" in name:
        print(f"== softmax port: {c}x {t/1000:.1f}ms total, {t/c:.1f}us avg, {c/gl_count if gl_count else 0:.1f}/step")

# gaps on merged GPU timeline (kernels+memcpy)
span = sorted(kern + memcpy)
if span:
    t0, t1 = span[0][0], max(e[0] + e[1] for e in span)
    busy = sum(e[1] for e in span)
    print(f"== GPU span {(t1-t0)/1e6:.2f}s  busy {busy/1e6:.2f}s  util {busy/(t1-t0)*100:.1f}%")
    gaps = []
    cur_end = None
    for ts, dur, _, _ in span:
        if cur_end is not None and ts > cur_end:
            gaps.append((ts - cur_end, cur_end))
        cur_end = max(cur_end or ts, ts + dur)
    big = [(g, s) for g, s in gaps if g > 5000]
    print(f"   gaps>5ms: {len(big)}, total {sum(g for g,_ in big)/1e3:.1f}ms")
    for g, s in sorted(big, reverse=True)[:8]:
        print(f"   gap {g/1000:8.1f}ms at +{(s-t0)/1e3:9.1f}ms")
