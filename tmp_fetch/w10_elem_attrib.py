"""Attribute the elementwise army: neighborhood anchoring in steady decode."""
import gzip, json, glob, collections, re
path = sorted(glob.glob("/root/prof_new/*.trace.json.gz"))[0]
data = json.load(gzip.open(path, "rt"))
ev = data["traceEvents"]
kern = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
t0, t1 = kern[0]["ts"], kern[-1]["ts"]
lo = t0 + (t1 - t0) * 0.85
ks = [e for e in kern if e["ts"] >= lo]

def shortname(n):
    m = re.match(r"void at::native::(\w+)", n)
    if m: return "aten:" + m.group(1)
    m = re.match(r"void (\w+)::(\w+)", n)
    if m: return m.group(1)[:6] + ":" + m.group(2)[:30]
    return n.split("<")[0].split("(")[0][:36]

# neighborhood: for each elementwise run (consecutive aten kernels), record
# the named (non-aten) kernel BEFORE the run and its length
named = re.compile(r"void at::native::")
runs = []
i = 0
while i < len(ks):
    if named.search(ks[i].get("name", "")):
        j = i
        while j < len(ks) and named.search(ks[j].get("name", "")):
            j += 1
        prev = shortname(ks[i-1].get("name","")) if i > 0 else "^"
        runs.append((prev, j - i))
        i = j
    else:
        i += 1
agg = collections.Counter()
ln = collections.Counter()
for prev, n in runs:
    agg[prev] += 1
    ln[prev] += n
print(f"aten runs: {len(runs)}, total aten kernels in runs: {sum(n for _,n in runs)}")
print("anchor_before  runs  aten_kernels")
for prev, c in agg.most_common(20):
    print(f"  {prev:44s} {c:5d} {ln[prev]:6d}")
# longest run contexts
print("longest runs:")
for prev, n in sorted(runs, key=lambda x: -x[1])[:10]:
    print(f"  {n:4d} after {prev}")
# aten functor census
fn = collections.Counter()
for e in ks:
    n = e.get("name","")
    if named.search(n):
        m = re.search(r"elementwise_kernel(?:_manual_unroll)?(?:_with_index)?<\d+,\s*(?:\d+,\s*)?at::native::(?:\(anonymous namespace\)::)?(\w+)", n)
        fn[m.group(1) if m else n[:50]] += 1
print("aten functor census:")
for f, c in fn.most_common(15):
    print(f"   {c:6d}  {f}")
