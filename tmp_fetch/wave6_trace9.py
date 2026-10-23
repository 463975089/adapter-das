# -*- coding: utf-8 -*-
"""Do the inflated-p50 kernels (silu/router/mix_stats/moe_sum) overlap
concurrent nccl/allgather spinners on the same GPU, while clean kernels
(rmsnorm/softmax) run alone?  SM-contention test from the trace."""
import gzip
import json
import sys

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)

kern = []
for e in data["traceEvents"]:
    if e.get("ph") == "X" and e.get("cat") == "kernel":
        kern.append((e.get("ts", 0), e.get("dur", 0), e.get("name", ""),
                     (e.get("args") or {}).get("stream", -1)))
kern.sort()

SPIN = ("ncclDevKernel", "allgather_vec")

def overlap_stats(target, label):
    evs = [(ts, ts + d, d, nm, st) for ts, d, nm, st in kern if target in nm]
    evs.sort(key=lambda e: -e[2])
    n_show = min(12, len(evs))
    with_spin = 0
    others = 0
    total = 0
    for ts, te, d, nm, st in evs:
        # count overlapping kernels on OTHER streams
        ov_names = {}
        for ts2, d2, nm2, st2 in kern:
            if ts2 >= te:
                break
            if ts2 + d2 > ts and st2 != st and nm2 != nm:
                key = nm2[:34]
                ov_names[key] = ov_names.get(key, 0) + 1
        total += 1
        if any(any(s in k for s in SPIN) for k in ov_names):
            with_spin += 1
        elif ov_names:
            others += 1
    print(f"{label}: {total} events; top-{n_show} longest:")
    for ts, te, d, nm, st in evs[:6]:
        ov = {}
        for ts2, d2, nm2, st2 in kern:
            if ts2 >= te:
                break
            if ts2 + d2 > ts and st2 != st and nm2 != nm:
                k = nm2[:30]
                ov[k] = ov.get(k, 0) + 1
        tops = sorted(ov.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {d/1000:7.1f}us stream={st} concurrent: "
              f"{'; '.join(f'{k}x{v}' for k, v in tops) or 'NONE'}")
    print(f"  summary: {with_spin}/{total} overlap spinners, "
          f"{others}/{total} overlap others", flush=True)


overlap_stats("_silu_clamp_ptgq", "silu_clamp_ptgq")
overlap_stats("_router_triton", "router_triton")
overlap_stats("_rmsnorm_kernel", "rmsnorm")
overlap_stats("_hc_mix_stats_partial", "hc_mix_stats")
