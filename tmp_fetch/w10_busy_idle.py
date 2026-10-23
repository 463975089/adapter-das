"""Busy vs gap account inside steady decode segments."""
import gzip, json, glob
path = sorted(glob.glob("/root/prof_new/*.trace.json.gz"))[0]
data = json.load(gzip.open(path, "rt"))
ev = data["traceEvents"]
kern = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
# steady tail: last 10% of timeline
t0, t1 = kern[0]["ts"], kern[-1]["ts"]
lo = t0 + (t1 - t0) * 0.85
ks = [e for e in kern if e["ts"] >= lo]
busy = sum(e.get("dur", 0) for e in ks)
span = ks[-1]["ts"] + ks[-1].get("dur", 0) - ks[0]["ts"]
gaps, prev_end = 0, None
gap_hist = {}
for e in ks:
    if prev_end is not None and e["ts"] > prev_end:
        g = e["ts"] - prev_end
        gaps += g
        b = min(int(g // 1_000_000), 10)  # ms bucket
        gap_hist[b] = gap_hist.get(b, 0) + 1
    prev_end = max(prev_end or 0, e["ts"] + e.get("dur", 0))
print(f"tail window {span/1e6:.2f}s kernels={len(ks)}")
print(f"busy {busy/1e6:.2f}s ({busy/span*100:.0f}%)  gap {gaps/1e6:.2f}s ({gaps/span*100:.0f}%)")
print("gap histogram (ms bucket: count):", dict(sorted(gap_hist.items())))
# overlap check: are kernels on one stream or concurrent?
# max concurrent kernels via sweep
events = []
for e in ks:
    events.append((e["ts"], 1)); events.append((e["ts"] + e.get("dur", 0), -1))
events.sort()
cur = mx = 0
for _, d in events:
    cur += d; mx = max(mx, cur)
print(f"max concurrent kernels: {mx}")
