# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Diff graph-arm vs eager-arm side-probe dumps (run inside container).

Pairs the verify session replays (g7_r*) with the eager arm's verify call
groups and reports the first diverging call/tensor; see
hcu_adapter/probe_side.py for the dump format and
docs/models.md 2026-10-18 for the battle record.

Usage: python3 _cmp_side_probe.py [graph_dir] [eager_dir]
"""

import glob
import os
import sys
from collections import defaultdict

import torch

GRAPH = sys.argv[1] if len(sys.argv) > 1 else "/root/probe_out_graph"
EAGER = sys.argv[2] if len(sys.argv) > 2 else "/root/probe_out_eager"


def pid_of(fn):
    return fn.rsplit("_p", 1)[1].split(".")[0]


def session_of(fn):
    base = os.path.basename(fn)
    return base.split("_")[0]


def replay_of(fn):
    base = os.path.basename(fn)
    return base.split("_")[1]


def load(files):
    data = []
    for f in files:
        d = torch.load(f, map_location="cuda")
        d["_file"] = os.path.basename(f)
        data.append(d)
    return data


def _rows(c):
    # verify passes q=None (only_qv absorbed form): use output rows
    return c.get("o_shape", c.get("q_shape", (0,)))[0]


def majority_qrows(calls):
    counts = defaultdict(int)
    for c in calls:
        counts[_rows(c)] += 1
    return max(counts.items(), key=lambda kv: kv[1])


def call_key(c):
    return (c.get("fn"), tuple(c.get("q_shape", ())))


def fmaxdiff(a, b):
    if a.shape != b.shape:
        return f"SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return f"DTDTYPE {a.dtype} vs {b.dtype}"
    d = (a.float() - b.float()).abs().max().item()
    return d


def intdiff(a, b):
    if a.shape != b.shape:
        return f"SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"
    ne = (a.int() != b.int())
    if not ne.any():
        return None
    idx = ne.flatten().nonzero()[0].item()
    flat_a = a.flatten().int()
    flat_b = b.flatten().int()
    return f"first@{idx}: {flat_a[idx].item()} vs {flat_b[idx].item()} (n_ne={int(ne.sum())})"


def cmp_call(ga, ea):
    """Compare one graph-arm call vs eager-arm call. Returns dict of diffs."""
    out = {}
    for k in ("cu_q", "cu_k", "cache_seqlens", "page_table"):
        if k in ga and k in ea:
            r = intdiff(ga[k], ea[k])
            if r:
                out[k] = r
    for k in ("q", "qv", "o", "lse"):
        if k in ga and k in ea:
            d = fmaxdiff(ga[k], ea[k])
            if isinstance(d, str) or d > 0:
                out[k] = d
        elif (k in ga) != (k in ea):
            out[k] = "PRESENCE"
    # meta flags that should never drift between arms at the same step
    for k in ("causal", "window", "scale", "max_seqlen_q", "only_qv"):
        if k in ga and k in ea and ga[k] != ea[k]:
            out[f"meta:{k}"] = f"{ga[k]} vs {ea[k]}"
    return out


def group_eager_calls(calls):
    """Split the flat eager call stream into (kind, [calls]) groups.
    kind: 'q<n>' by q row count; varlen fns keyed 'varlen'."""
    groups = []
    for c in calls:
        kind = "varlen" if c.get("fn") == "varlen" else f"q{_rows(c)}"
        if groups and groups[-1][0] == kind:
            groups[-1][1].append(c)
        else:
            groups.append((kind, [c]))
    return groups


def main():
    gfiles = sorted(glob.glob(os.path.join(GRAPH, "g*_r*.pt")))
    efiles = sorted(glob.glob(os.path.join(EAGER, "eagerchunk_*.pt")))
    if not gfiles:
        print("NO graph dumps in", GRAPH)
        return
    if not efiles:
        print("NO eager dumps in", EAGER)
        return

    # ---- pick one rank per arm (rank 0 preferred) --------------------
    gload = load(gfiles)
    eload = load(efiles)
    g_by_rank = defaultdict(list)
    for d in gload:
        g_by_rank[d.get("rank", -1)].append(d)
    e_by_rank = defaultdict(list)
    for d in eload:
        e_by_rank[d.get("rank", -1)].append(d)
    g_rank = 0 if 0 in g_by_rank else sorted(g_by_rank)[0]
    e_rank = 0 if 0 in e_by_rank else sorted(e_by_rank)[0]
    gdat = g_by_rank[g_rank]
    edat = e_by_rank[e_rank]
    print(f"graph arm: {len(gdat)} session-dumps (rank {g_rank}); "
          f"eager arm: {len(edat)} chunks (rank {e_rank})")

    # ---- identify graph sessions ------------------------------------
    sessions = defaultdict(dict)  # sid -> rtag -> dump
    for d in gdat:
        sessions[session_of(d["_file"])][replay_of(d["_file"])] = d
    info = []
    for sid, reps in sorted(sessions.items(), key=lambda kv: int(kv[0][1:])):
        r1 = reps.get(min(reps))
        qrows, n = majority_qrows(r1["calls"])
        info.append((sid, len(r1["calls"]), qrows, n, len(reps)))
        print(f"  session {sid}: {len(r1['calls'])} calls, majority q_rows={qrows}"
              f" x{n}, replays_dumped={len(reps)}")

    verify_sid = max(
        (s for s in info if s[2] not in (0, 1)),
        key=lambda s: s[1],
        default=None,
    )
    if verify_sid is None:
        print("!! no verify-looking session (q_rows>1) found")
        return
    sid = verify_sid[0]
    print(f"-> verify session: {sid} ({verify_sid[1]} calls, q_rows={verify_sid[2]})")

    # ---- eager reference stream -------------------------------------
    ecalls = []
    for d in sorted(edat, key=lambda d: d["base_idx"]):
        ecalls.extend(d["calls"])
    groups = group_eager_calls(ecalls)
    from collections import Counter
    print("eager group kinds:", Counter(k for k, _ in groups))
    qn = f"q{verify_sid[2]}"
    eager_verify_groups = [g for k, g in groups if k == qn]
    if not eager_verify_groups:
        print(f"!! no eager groups with kind {qn}")
        return
    print(f"eager {qn} groups: {len(eager_verify_groups)}, sizes "
          f"{[len(g) for g in eager_verify_groups[:10]]}")

    # ---- diff each dumped replay against the matching eager group ----
    # replay pairing assumption: engine step k uses verify replay k and the
    # k-th eager verify group (before divergence the two arms step in
    # lockstep; after divergence we only read magnitudes, not pairing)
    reps = sessions[sid]
    first_div = None
    for rtag in sorted(reps):
        gcalls = reps[rtag]["calls"]
        step = int(rtag[1:])
        if step <= len(eager_verify_groups):
            ecalls_g = eager_verify_groups[step - 1]
        else:
            ecalls_g = eager_verify_groups[-1]
        n = min(len(gcalls), len(ecalls_g))
        call_diffs = []
        for i in range(n):
            d = cmp_call(gcalls[i], ecalls_g[i])
            if d:
                call_diffs.append((i, d))
        if call_diffs:
            i, d = call_diffs[0]
            qd = d.get("q", 0) if not isinstance(d.get("q", 0), str) else d.get("q")
            print(f"[{rtag}] DIVERGED at call {i}/{n}: {d}")
            if first_div is None:
                first_div = (rtag, i, d)
        else:
            print(f"[{rtag}] identical across {n} calls (inputs+outputs)")
    if first_div:
        print("\nVERDICT: first divergence:", first_div)
    else:
        print("\nVERDICT: all dumped verify replays match eager reference bit-exact")


if __name__ == "__main__":
    main()
