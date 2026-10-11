"""Dual-GPU gate for the host-shm custom AR (ar_ext.cu + triton_ar.py).

1) eager: fp32/bf16/fp16 x {1K, 64K, 1M} elements vs torch-distributed ref
2) graph: capture TWO chained ARs, replay 200 rounds mutating the inputs
   the graph reads, verify every replay against the eager reference
   (catches slot-reuse and stale-line bugs)
3) diag: report the budget-exhaustion counter

Run: HIP_VISIBLE_DEVICES=6,7 torchrun --nproc-per-node=2 --master-port=29812 \
        tests/tools/_test_hcu_ar2.py
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sgl_kernel._adapters import triton_ar  # noqa: E402

CAP_MB = 8
FAILS = []


def check(tag, got, ref, tol):
    ok = torch.allclose(got.float(), ref.float(), atol=tol, rtol=tol)
    if not ok:
        md = (got.float() - ref.float()).abs().max().item()
        FAILS.append(f"{tag}: maxdiff {md}")
    return ok


def main():
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")

    hb = triton_ar.allocate_meta_buffer(
        triton_ar.DATA_OFF + CAP_MB * 1024 * 1024
    )
    handles = [None] * ws
    dist.all_gather_object(handles, hb, group=gloo)
    st = triton_ar.init_custom_ar(None, None, handles, None, rank, True)
    print(f"[r{rank}] init ok (slots={st.slots})", flush=True)

    # ---- eager matrix ----
    for dt, tol in ((torch.float32, 1e-4), (torch.bfloat16, 2e-2),
                    (torch.float16, 2e-2)):
        for n in (1024, 65536, 1024 * 1024):
            x = torch.full((n,), float(rank + 1), dtype=dt, device="cuda")
            ref = x.clone()
            dist.all_reduce(ref)
            out = st.all_reduce(x)
            torch.cuda.synchronize()
            ok = check(f"eager {dt} {n}", out, ref, tol)
            if rank == 0:
                print(f"eager {str(dt):16s} n={n:8d} {'OK' if ok else 'FAIL'}",
                      flush=True)

    # ---- graph: two chained ARs, 200 replays, mutating inputs ----
    n = 262144
    dt = torch.bfloat16
    x1 = torch.randn((n,), dtype=dt, device="cuda")
    x2 = torch.randn((n,), dtype=dt, device="cuda")
    a = torch.empty_like(x1)
    b = torch.empty_like(x2)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        a.copy_(x1)
        o1 = st.all_reduce(a)
        b.copy_(x2)
        o2 = st.all_reduce(b)
    print(f"[r{rank}] captured", flush=True)

    bad = 0
    for it in range(200):
        x1.normal_()
        x1 += (it + 1) * 0.01
        x2.normal_()
        x2 -= (it + 1) * 0.02
        g.replay()
        r1 = x1.clone()
        dist.all_reduce(r1)
        r2 = x2.clone()
        dist.all_reduce(r2)
        torch.cuda.synchronize()
        if not check(f"graph it{it} ar1", o1, r1, 2e-2):
            bad += 1
        if not check(f"graph it{it} ar2", o2, r2, 2e-2):
            bad += 1
    if rank == 0:
        print(f"graph replay 200 rounds: {'OK' if bad == 0 else f'{bad} FAIL'}",
              flush=True)

    words = st.read_sync_words()
    if rank == 0:
        print(f"sync words [seq, packed, diag]: {words}", flush=True)

    triton_ar.dispose(st)
    dist.barrier()
    if rank == 0:
        print("ALL DONE " + ("PASS" if not FAILS else f"FAIL {FAILS[:5]}"),
              flush=True)
    dist.destroy_process_group()
    assert not FAILS, FAILS


main()
