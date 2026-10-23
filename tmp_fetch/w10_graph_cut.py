"""Cut kernel stream at hipGraphLaunch boundaries -> per-step decode account."""
import gzip, json, glob, collections
path = sorted(glob.glob("/root/prof_new/*.trace.json.gz"))[0]
data = json.load(gzip.open(path, "rt"))
ev = data["traceEvents"]
kern = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
rts = sorted([e for e in ev if "GraphLaunch" in e.get("name", "")], key=lambda e: e["ts"])
print(f"kernels={len(kern)} graph_launches={len(rts)}")
# intervals between consecutive graph launches
cuts = [e["ts"] for e in rts]
seg_sizes = []
seg_kernels = []
ki = 0
for a, b in zip(cuts, cuts[1:]):
    ks = []
    while ki < len(kern) and kern[ki]["ts"] < b:
        if kern[ki]["ts"] >= a:
            ks.append(kern[ki])
        ki += 1
    seg_kernels.append(ks)
    seg_sizes.append(len(ks))
print("kernels/interval:", seg_sizes[:40])
# steady tail: last 20 intervals
tail = seg_kernels[-20:]
tot = sum(len(s) for s in tail)
print(f"tail20: {tot} kernels, {tot/20:.0f}/interval")
fam = collections.Counter()
dur = collections.Counter()
for s in tail:
    for e in s:
        n = e.get("name", "")
        m = n.split("<")[0].split("(")[0][:64]
        fam[m] += 1
        dur[m] += e.get("dur", 0)
print("tail families (count per interval avg):")
for n, c in fam.most_common(28):
    print(f"   {c/20:7.1f}/it {dur[n]/20/1000:8.2f}ms  {n}")
stalls = [e for s in tail for e in s if e.get("dur", 0) > 20000]
print(f"tail stalls>20ms: {len(stalls)}")
for e in stalls[:8]:
    print(f"   {e['dur']/1000:.1f}ms {e.get('name','')[:70]}")
step_ms = (cuts[-1] - cuts[-21]) / 20 / 1000
print(f"interval wall: {step_ms:.1f}ms")
