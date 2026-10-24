"""W11: static verification — are 116ms stall windows graph-replay boundaries?

Mechanism hypothesis: DSPARK splits decode into per-block graph replays
(speculative_dspark_block_size=5); between replays the host does boundary
work (rs+rsv reduction, sampling, bookkeeping); all 8 ranks freeze together.

Tests on graph+DSPARK TP-0 trace:
  T1: stall gaps (>15ms, no kernel running on ANY compute stream) aligned
      with the interval BETWEEN two consecutive hipGraphLaunch host calls?
  T2: kernel right after gap = first kernel of a NEW graph replay?
      (correlation match: kernel.correlation == graphlaunch.correlation)
  T3: hipGraphLaunch count per decode step ~= boundaries+draft+verify?
  T4: what host-side runtime calls overlap the gap window?
"""
import gzip, json, sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else r"E:\GITS\hcu-sglang-adapter\prof_1024\TP-0.trace.json.gz"
GAP_MS = 15.0

print(f"loading {path} ...")
with gzip.open(path, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"]
print(f"events: {len(ev)}")

kernels, graph_launches, other_runtime = [], [], []
for e in ev:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "")
    name = e.get("name", "")
    if cat == "kernel":
        kernels.append(e)
    elif cat in ("cuda_runtime", "runtime"):
        if name == "hipGraphLaunch":
            graph_launches.append(e)
        else:
            other_runtime.append(e)

kernels.sort(key=lambda x: x["ts"])
graph_launches.sort(key=lambda x: x["ts"])
print(f"kernels={len(kernels)} graph_launches={len(graph_launches)}")

# union busy intervals across compute streams (GPU kernels only) -> global gaps
busy = [(k["ts"], k["ts"] + k["dur"]) for k in kernels]
busy.sort()
merged = []
for s, t in busy:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], t)
    else:
        merged.append([s, t])
gaps = []
for i in range(1, len(merged)):
    g0, g1 = merged[i - 1][1], merged[i][0]
    if g1 - g0 > GAP_MS * 1000:
        gaps.append((g0, g1, (g1 - g0) / 1000.0))
print(f"\nT1: global GPU-busy gaps > {GAP_MS}ms: {len(gaps)}")
for g0, g1, ms in gaps:
    print(f"  gap {ms:7.1f}ms  [{g0}..{g1}]")

# T2: for each gap, find graph launches & kernels around it
print("\nT2: boundary alignment per gap")
gl_ts = [g["ts"] for g in graph_launches]
import bisect
kern_by_corr = {}
for k in kernels:
    kern_by_corr.setdefault(k.get("correlation"), []).append(k)

n_aligned = 0
for g0, g1, ms in gaps:
    # launches strictly inside the gap (host launched while GPU idle)
    inside = [g for g in graph_launches if g0 <= g["ts"] <= g1]
    # launch immediately before gap start (within 5ms)
    i = bisect.bisect_left(gl_ts, g0) - 1
    prev_gl = graph_launches[i] if i >= 0 else None
    # kernel immediately after gap: first kernel with ts >= g1
    j = bisect.bisect_left([k["ts"] for k in kernels], g1)
    nxt = kernels[j] if j < len(kernels) else None
    # which launch owns the next kernel?
    owner = None
    if nxt is not None:
        for g in graph_launches:
            if g.get("correlation") == nxt.get("correlation"):
                owner = g
                break
    ok = owner is not None and g0 <= owner["ts"] <= g1
    if ok:
        n_aligned += 1
    print(f"  gap {ms:6.1f}ms: launches_inside={len(inside)} "
          f"next_kern={nxt['name'][:40] if nxt else None} "
          f"owner_launch_in_gap={ok} "
          f"owner_ts_delta={(owner['ts']-g0)/1000 if owner else -1:.1f}ms_after_gap_start")
print(f"aligned {n_aligned}/{len(gaps)}")

# T3: graph launches per step. Detect step boundaries via big host gaps or
# simply histogram of inter-launch deltas
print("\nT3: hipGraphLaunch inter-arrival histogram (ms buckets)")
deltas = [(graph_launches[i+1]["ts"] - graph_launches[i]["ts"]) / 1000.0
          for i in range(len(graph_launches) - 1)]
h = collections.Counter()
for d in deltas:
    if d < 1: h["<1ms"] += 1
    elif d < 5: h["1-5ms"] += 1
    elif d < 20: h["5-20ms"] += 1
    elif d < 60: h["20-60ms"] += 1
    elif d < 130: h["60-130ms"] += 1
    else: h[">130ms"] += 1
for k in sorted(h):
    print(f"  {k:>8}: {h[k]}")

# step segmentation: gap > 200ms between launches = step boundary (scheduler)
step_bounds = [0]
for i, d in enumerate(deltas):
    if d > 200:
        step_bounds.append(i + 1)
step_bounds.append(len(graph_launches))
steps = [step_bounds[i+1] - step_bounds[i] for i in range(len(step_bounds) - 1)]
print(f"launches per step: {steps}")

# T4: host runtime calls overlapping each gap
print("\nT4: host runtime overlap inside gaps (top by total dur)")
overlap_tot = collections.Counter()
overlap_cnt = collections.Counter()
for e in other_runtime:
    for g0, g1, ms in gaps:
        if e["ts"] < g1 and e["ts"] + e["dur"] > g0:
            ov = min(e["ts"] + e["dur"], g1) - max(e["ts"], g0)
            if ov > 0:
                overlap_tot[e["name"]] += ov / 1000.0
                overlap_cnt[e["name"]] += 1
                    # only count once per event
            break
for name, tot in overlap_tot.most_common(15):
    print(f"  {name[:60]:60} n={overlap_cnt[name]:4} total={tot:.1f}ms")
