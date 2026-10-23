import gzip, json, glob, collections
path = sorted(glob.glob("/root/prof_new/*.trace.json.gz"))[0]
data = json.load(gzip.open(path, "rt"))
ev = data["traceEvents"]
kern = [e for e in ev if e.get("cat") == "kernel"]
names = collections.Counter(e.get("name","") for e in kern)
print("total kernels:", len(kern))
for pat in ("mqa", "paged", "ragged", "topk", "indexer", "gather"):
    hits = [(n[:90], c) for n, c in names.items() if pat in n.lower()]
    hits.sort(key=lambda x: -x[1])
    print(f"--- {pat}: {sum(c for _,c in hits)}")
    for n, c in hits[:6]:
        print(f"   {c:6d}  {n}")
# aten void family breakdown (the 1935/st mystery)
void = collections.Counter()
for e in kern:
    n = e.get("name","")
    if n.startswith("void at::native") or n.startswith("void "):
        # capture functor part
        import re
        m = re.search(r"(elementwise_kernel\S*|reduce_kernel\S*|\w+Functor\w*)", n)
        void[(m.group(1) if m else n[:60])] += 1
print("--- aten breakdown top12:")
for n, c in void.most_common(12):
    print(f"   {c:6d}  {n}")
