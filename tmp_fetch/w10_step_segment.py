"""Decode-only re-census: cut the trace at graph-launch boundaries."""
import gzip, json, glob, collections, re
path = sorted(glob.glob("/root/prof_new/*.trace.json.gz"))[0]
data = json.load(gzip.open(path, "rt"))
ev = data["traceEvents"]
kern = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
t0, t1 = kern[0]["ts"], kern[-1]["ts"] + kern[-1].get("dur", 0)
print(f"kernel span {(t1-t0)/1e6:.1f}s, {len(kern)} kernels")
# time-bucket density to find the steady decode tail
W = 50_000  # 50ms buckets
nb = int((t1 - t0) // W) + 1
cnt = [0] * nb
for e in kern:
    cnt[min(int((e["ts"] - t0) // W), nb - 1)] += 1
# print coarse profile (per 0.5s)
line = []
for i in range(0, nb, 10):
    line.append(sum(cnt[i:i+10]))
print("kern/0.5s:", line)
# runtime graph launches = decode steps
gl = [e for e in ev if e.get("cat") in ("cuda_runtime","runtime") and
      "GraphLaunch" in e.get("name", "")]
print("graph launches:", len(gl))
