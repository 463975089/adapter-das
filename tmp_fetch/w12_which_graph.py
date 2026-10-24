"""W12b: which graph launch eats the ~20ms tick — draft or verify?
For each hipGraphLaunch on TP-0 (ts, dur), find the gpu_user_annotation
('step[...]') covering the first kernel launched by it.
"""
import gzip, json, bisect, collections

path = r"E:\GITS\hcu-sglang-adapter\prof_1024\TP-0.trace.json.gz"
with gzip.open(path, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"]

rt = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"
             and e["name"] == "hipGraphLaunch"], key=lambda x: x["ts"])
ann = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation"
              and e.get("name", "").startswith("step[")], key=lambda x: x["ts"])
ann_ts = [e["ts"] for e in ann]

def covering(t):
    i = bisect.bisect_right(ann_ts, t) - 1
    while i >= 0:
        e = ann[i]
        if e["ts"] <= t <= e["ts"] + e.get("dur", 0):
            return e["name"]
        i -= 1
    return "?"

rows = []
for e in rt:
    dur = e.get("dur", 0) / 1000
    # phase of the launch window (use launch end as approx first-kernel time)
    phase_end = covering(e["ts"] + e.get("dur", 0))
    phase_start = covering(e["ts"])
    rows.append((e["ts"], dur, phase_start, phase_end))

tick = [r for r in rows if r[1] > 15]
fast = [r for r in rows if r[1] <= 15]
print(f"launches: {len(rows)}, tick(>15ms): {len(tick)}, fast: {len(fast)}")
print("\ntick launches phase (start-cover -> end-cover):")
c = collections.Counter()
for ts, dur, a, b in tick:
    c[(a, b)] += 1
for k, n in c.most_common():
    print(f"  {k[0][:40]} -> {k[1][:40]} : {n}x")
print("\nfast launches phase:")
c2 = collections.Counter()
for ts, dur, a, b in fast:
    c2[(a, b)] += 1
for k, n in c2.most_common():
    print(f"  {k[0][:40]} -> {k[1][:40]} : {n}x")

# sequence pattern: print first 16 launches with durs in time order
print("\nfirst 16 launches in order (dur ms):")
print("  " + " ".join(f"{d:.1f}" for _, d, _, _ in rows[:16]))
