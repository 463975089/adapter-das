# -*- coding: utf-8 -*-
"""Wave6 follow-up: is the slowness kernel-side (pathological branch/shape)
or host-side (launch bubbles)?

Part 1 (single rank): per-call duration distributions of hot kernel families
(bimodal = conditional slow branch; uniformly slow = shape/doing-too-much),
plus grid/block args, plus host-busy fraction inside inter-graph-launch gaps.
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
kern = []  # (ts, dur, name)
cpu = []   # (ts, dur, name, cat, tid)
runtime_launch = []  # hipGraphLaunch (ts, dur)
for e in ev:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "")
    if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        kern.append((e.get("ts", 0), e.get("dur", 0), e.get("name", "")))
        if "args" in e and len(runtime_launch) < 3 and "apply_rotary" in e.get("name", ""):
            pass
    elif cat in ("cuda_runtime", "runtime", "hip_runtime"):
        if "raphLaunch" in e.get("name", ""):
            runtime_launch.append((e.get("ts", 0), e.get("dur", 0)))
    elif cat in ("cpu_op", "python_function", "user_annotation"):
        cpu.append((e.get("ts", 0), e.get("dur", 0), e.get("name", ""), cat, e.get("tid", 0)))

kern.sort()
runtime_launch.sort()

print(f"kernels {len(kern)}  graph-launches {len(runtime_launch)}  cpu-ev {len(cpu)}")
ld = [d for _, d in runtime_launch]
ld.sort()
print(f"hipGraphLaunch dur us: p50={ld[len(ld)//2]} max={ld[-1]} total={sum(ld)/1000:.1f}ms")

# --- per-family duration distribution for hot families
TARGETS = [
    "apply_rotary_emb", "silu_clamp_ptgq", "hc_mix_stats", "moe_sum_scale",
    "allgather_vec", "ncclDevKernel", "rmsnorm_kernel", "router_triton",
    "vectorized_elementwise", "manual_unroll", "per_token_group_quant",
    "dequantize_k_cache", "softmax_rows", "wNaN_block", "mix_reduce_sinkhorn",
    "combine_kernel", "fused_k_norm_rope", "hc_combine",
]
famcalls = defaultdict(list)
famargs = {}
for ts, d, name in kern:
    for t in TARGETS:
        if t in name:
            famcalls[t].append(d)
            if t not in famargs:
                famargs[t] = name
            break

print(f"\n== per-call dur distribution (us) p50/p90/max + bimodality:")
for t, durs in sorted(famcalls.items(), key=lambda kv: -sum(kv[1])):
    durs.sort()
    n = len(durs)
    p10, p50, p90 = durs[n // 10], durs[n // 2], durs[(n * 9) // 10]
    med = p50
    frac_fast = sum(1 for d in durs if d < med / 3) / n
    frac_slow = sum(1 for d in durs if d > med * 3) / n
    tag = ""
    if frac_fast > 0.05 and frac_slow > 0.05:
        tag = "  <== BIMODAL"
    elif p90 > med * 5:
        tag = "  <= heavy tail"
    print(f"  {t:26s} n={n:6d} p10={p10:8.1f} p50={p50:8.1f} p90={p90:8.1f} "
          f"max={durs[-1]:9.1f}{tag}")

# grid/block args of a few families (first matching event carrying args)
print("\n== first-event args (grid/block) of pathological suspects:")
want = ("apply_rotary_emb", "silu_clamp_ptgq", "hc_mix_stats", "moe_sum_scale")
seen = set()
for e in ev:
    if e.get("ph") != "X" or e.get("cat") != "kernel":
        continue
    nm = e.get("name", "")
    for w in want:
        if w in nm and w not in seen:
            seen.add(w)
            a = e.get("args", {})
            grid = a.get("grid", [])
            block = a.get("block", [])
            print(f"  {w:22s} grid={grid} block={block} dur={e.get('dur')}")
    if len(seen) == len(want):
        break

# --- host-busy inside inter-launch windows
if len(runtime_launch) >= 10:
    print("\n== inter-graph-launch windows: host busy vs idle")
    # merge CPU intervals per process (cheap: global merge, single scheduler proc dominates)
    iv = sorted((ts, ts + d) for ts, d, _, _, _ in cpu if d > 0)
    merged = []
    for s, t in iv:
        if merged and s <= merged[-1][1]:
            if t > merged[-1][1]:
                merged[-1][1] = t
        else:
            merged.append([s, t])

    def host_busy(s, t):
        tot = 0
        for ms, mt in merged:
            if mt <= s:
                continue
            if ms >= t:
                break
            tot += min(mt, t) - max(ms, s)
        return tot

    stats = []
    for i in range(2, len(runtime_launch) - 2):  # skip warm edges
        s = runtime_launch[i][0] + runtime_launch[i][1]
        t = runtime_launch[i + 1][0]
        w = t - s
        if w <= 0:
            continue
        stats.append((w, host_busy(s, t) / w))
    stats.sort(key=lambda x: -x[0])
    ws = [w for w, _ in stats]
    bfrac = sorted(f for _, f in stats)
    print(f"  windows n={len(stats)} p50={ws[len(ws)//2]/1000:.1f}ms "
          f"max={ws[0]/1000:.1f}ms; host-busy frac p50={bfrac[len(bfrac)//2]:.2f} "
          f"min={bfrac[0]:.2f} max={bfrac[-1]:.2f}")
    print("  top-6 widest windows (width_ms, host_busy_frac):")
    for w, f in stats[:6]:
        print(f"    {w/1000:8.1f}ms  busy={f:.2f}")

# --- what CPU ops dominate the widest windows
print("\n== top CPU ops inside the 3 widest inter-launch windows:")
if len(runtime_launch) >= 10:
    wide = []
    for i in range(2, len(runtime_launch) - 2):
        s = runtime_launch[i][0] + runtime_launch[i][1]
        t = runtime_launch[i + 1][0]
        if t > s:
            wide.append((t - s, s, t))
    wide.sort(reverse=True)
    for w, s, t in wide[:3]:
        agg = defaultdict(float)
        cnt = defaultdict(int)
        for ts, d, nm, cat, tid in cpu:
            if ts >= s and ts < t and d > 0:
                agg[nm[:60]] += d
                cnt[nm[:60]] += 1
        top = sorted(agg.items(), key=lambda kv: -kv[1])[:5]
        print(f"  window {w/1000:.1f}ms:")
        for nm, tot in top:
            print(f"    {tot/1000:7.1f}ms x{cnt[nm]:3d}  {nm}")
