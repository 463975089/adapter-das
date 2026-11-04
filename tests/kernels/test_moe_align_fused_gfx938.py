# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Wave10 parity gate: single-CTA fused small-shape moe_align vs the quartet.

The decode step calls moe_align_block_size 44x (40 main layers + 4 draft)
at tiny shapes (~96-576 lanes). Each call costs SIX kernel launches: the
aten fill_/zeros setup pair plus the hist/scan/fill/block_expert triton
quartet = 264 kernels/step anchored between _moe_align_hist neighbors in
the wave6 census. The fused kernel (_moe_align_fused_small_kernel, armed
for numel <= HCU_MOE_ALIGN_FUSED, default 2048) does all of it in ONE
launch with zero setup.

Contracts pinned here (must match the quartet bit-exactly except
intra-segment token order, which the fused_moe runner ignores):
  - bucket mapping: valid experts keep their id; sentinel -1 and stray
    out-of-range values fall into the LAST bucket (nb = E+1)
  - sorted_token_ids pad value = numel, pad slots inside block-aligned
    segments AND the whole tail beyond total
  - experts_ids: owned block -> expert id, sentinel-bucket block -> -1,
    tail block -> nb
  - num_tokens_post_pad = sum of block-aligned counts
  - cumsum_buffer = exclusive prefix of block-aligned counts (+ extra
    trailing total slot when the caller sized E+2)
  - CUDA-graph capture safe (static grid, no host syncs) and replay
    tracks in-place mutated topk_ids

Run (container, PYTHONPATH=/root/hcu_v0522:/sglang-v0522/python):
HIP_VISIBLE_DEVICES=2 python tests/kernels/test_moe_align_fused_gfx938.py
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import sgl_kernel  # noqa: E402

FAILS = []


def ok(tag, cond, detail=""):
    print(f"{tag}: {'OK' if cond else 'FAIL ' + detail}", flush=True)
    if not cond:
        FAILS.append(tag)


def _buffers(ids, E, B, extra):
    numel = ids.numel()
    nb = E + 1
    max_padded = numel + (E + 1) * (B - 1)
    dev = ids.device
    return dict(
        sorted=torch.full((max_padded,), -12345, device=dev, dtype=torch.int32),
        experts=torch.full(((max_padded + B - 1) // B,), -12345, device=dev,
                           dtype=torch.int32),
        post=torch.full((1,), -12345, device=dev, dtype=torch.int32),
        cumsum=torch.full((nb + (1 if extra else 0),), -12345, device=dev,
                          dtype=torch.int32),
    )


def _run(ids, E, B, extra, fused):
    os.environ["HCU_MOE_ALIGN_FUSED"] = "0" if not fused else "2048"
    buf = _buffers(ids, E, B, extra)
    sgl_kernel.moe_align_block_size(ids, E, B, buf["sorted"], buf["experts"],
                                    buf["post"], buf["cumsum"])
    return buf


def _bucket_counts(ids, E):
    flat = ids.reshape(-1).long()
    mapped = torch.where((flat >= 0) & (flat < E), flat, torch.full_like(flat, E))
    return torch.bincount(mapped, minlength=E + 1)


def _check_case(tag, ids, E, B, extra):
    g = _run(ids, E, B, extra, fused=False)
    f = _run(ids, E, B, extra, fused=True)
    numel = ids.numel()
    nb = E + 1
    ok(f"{tag}/post_pad", bool(torch.equal(g["post"], f["post"])),
       f'{g["post"].item()} vs {f["post"].item()}')
    ok(f"{tag}/cumsum", bool(torch.equal(g["cumsum"], f["cumsum"])),
       f'{g["cumsum"].cpu().tolist()[:8]} vs {f["cumsum"].cpu().tolist()[:8]}')
    ok(f"{tag}/experts_ids", bool(torch.equal(g["experts"], f["experts"])))

    counts = _bucket_counts(ids, E)
    aligned = ((counts + B - 1) // B) * B
    excl = torch.cumsum(aligned, 0) - aligned
    total = int(aligned.sum())
    ok(f"{tag}/total_sane", int(f["post"].item()) == total,
       f'{int(f["post"].item())} vs {total}')
    # whole-buffer pad fill first, then per-segment multiset compare
    for name, buf in (("gold", g), ("fused", f)):
        srt = buf["sorted"].cpu()
        pad_ok = bool((srt[total:] == numel).all()) if srt.numel() > total else True
        ok(f"{tag}/{name}_tail_pad", pad_ok)
    mism = []
    for b in range(nb):
        seg_g = g["sorted"][int(excl[b]):int(excl[b] + counts[b])].cpu()
        seg_f = f["sorted"][int(excl[b]):int(excl[b] + counts[b])].cpu()
        if not torch.equal(seg_g.sort().values, seg_f.sort().values):
            mism.append(b)
        pad_seg = f["sorted"][int(excl[b] + counts[b]):int(excl[b] + aligned[b])]
        if pad_seg.numel() and not bool((pad_seg == numel).all()):
            mism.append(f"{b}:pad")
    ok(f"{tag}/sorted_segments", not mism, str(mism[:4]))
    # sentinel bucket blocks carry -1 (spot-check via experts_ids diff done
    # above; explicit sentinel scan here for readability)
    if counts[E] > 0:
        blk = int(excl[E]) // B
        ok(f"{tag}/sentinel_block_-1", int(f["experts"][blk].item()) == -1,
           str(f["experts"][blk].item()))


def main():
    assert torch.cuda.is_available(), "needs GPU"
    torch.manual_seed(11)
    E, B = 48, 64  # DSV4.1F decode: EP8 local experts, fused_moe block

    # production-shaped: 96 lanes (bs16 x topk6) and 576 lanes
    for lanes, tag in ((96, "prod96"), (576, "prod576")):
        ids = torch.randint(0, E, (lanes // 6, 6), device="cuda",
                            dtype=torch.int32)
        _check_case(tag, ids, E, B, extra=False)
    # EP sentinels: ~30% -1 lanes, and all-sentinel
    ids = torch.randint(0, E, (16, 6), device="cuda", dtype=torch.int32)
    ids.view(-1)[:: 3] = -1
    _check_case("sent30pct", ids, E, B, extra=False)
    _check_case("all_sentinel", torch.full((8, 6), -1, device="cuda",
                                           dtype=torch.int32), E, B, extra=False)
    # extremes: empty, single lane, one-expert skew, boundary 2048 (fused)
    # and 2088 (quartet arm -- parity must still hold there)
    _check_case("numel0", torch.empty(0, 6, device="cuda", dtype=torch.int32),
                E, B, extra=False)
    _check_case("numel1", torch.randint(0, E, (1, 1), device="cuda",
                                        dtype=torch.int32), E, B, extra=False)
    _check_case("skew1expert", torch.zeros(32, 6, device="cuda",
                                           dtype=torch.int32), E, B, extra=False)
    _check_case("boundary2048", torch.randint(0, E, (342, 6), device="cuda",
                                              dtype=torch.int32), E, B, extra=False)
    _check_case("quartet2088", torch.randint(0, E, (348, 6), device="cuda",
                                             dtype=torch.int32), E, B, extra=False)
    # other block sizes + int64 ids + E+2 cumsum buffer (write_extra) +
    # large-E style
    _check_case("bs16", torch.randint(0, 300, (16, 6), device="cuda",
                                      dtype=torch.int32), 384, 16, extra=True)
    _check_case("bs128_i64",
                torch.randint(0, 48, (16, 6), device="cuda",
                              dtype=torch.int64), 48, 128, extra=True)
    _check_case("E8_tiny", torch.randint(0, 8, (5, 4), device="cuda",
                                         dtype=torch.int32), 8, 8, extra=False)
    # stray out-of-range id (defensive mapping, quartet parity)
    ids = torch.randint(0, E, (16, 6), device="cuda", dtype=torch.int32)
    ids.view(-1)[7] = E + 5  # out-of-range -> sentinel bucket in BOTH arms
    _check_case("oob_lane", ids, E, B, extra=False)

    # capture safety: fused arm under CUDA graph, replay after in-place
    # input mutation must track the quartet eager recompute
    ids = torch.randint(0, E, (16, 6), device="cuda", dtype=torch.int32)
    os.environ["HCU_MOE_ALIGN_FUSED"] = "2048"
    buf = _buffers(ids, E, B, extra=False)

    def call():
        sgl_kernel.moe_align_block_size(ids, E, B, buf["sorted"],
                                        buf["experts"], buf["post"],
                                        buf["cumsum"])

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    ids2 = torch.randint(0, E, (16, 6), device="cuda", dtype=torch.int32)
    ids2.view(-1)[::4] = -1
    ids.copy_(ids2)
    g.replay()
    torch.cuda.synchronize()
    ref = _run(ids2, E, B, extra=False, fused=False)
    ok("graph/replay_post", bool(torch.equal(buf["post"], ref["post"])))
    ok("graph/replay_cumsum", bool(torch.equal(buf["cumsum"], ref["cumsum"])))
    ok("graph/replay_experts", bool(torch.equal(buf["experts"], ref["experts"])))
    c2 = _bucket_counts(ids2, E)
    al2 = ((c2 + B - 1) // B) * B
    ex2 = torch.cumsum(al2, 0) - al2
    mism = [b for b in range(E + 1)
            if not torch.equal(buf["sorted"][int(ex2[b]):int(ex2[b] + c2[b])]
                               .cpu().sort().values,
                               ref["sorted"][int(ex2[b]):int(ex2[b] + c2[b])]
                               .cpu().sort().values)]
    ok("graph/replay_sorted", not mism, str(mism[:4]))

    # kernel-count fingerprint: fused 1 launch vs quartet 6
    from torch.profiler import ProfilerActivity, profile

    def count_kernels(call):
        for _ in range(2):
            call()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            call()
            torch.cuda.synchronize()
        prof.export_chrome_trace("/tmp/moe_align_gate.json")
        data = json.load(open("/tmp/moe_align_gate.json"))
        return [e.get("name", "") for e in data["traceEvents"]
                if e.get("cat") == "kernel"]

    os.environ["HCU_MOE_ALIGN_FUSED"] = "2048"
    kn_f = count_kernels(call)
    os.environ["HCU_MOE_ALIGN_FUSED"] = "0"
    kn_q = count_kernels(call)
    os.environ.pop("HCU_MOE_ALIGN_FUSED", None)
    print(f"kernels: fused={len(kn_f)} {kn_f[:3]}")
    print(f"kernels: quartet={len(kn_q)} {[n_[:40] for n_ in kn_q[:8]]}")
    ok("fused single launch", len(kn_f) == 1, str(kn_f))
    ok("quartet six launches", len(kn_q) == 6, str(len(kn_q)))
    ok("kernel-count win", len(kn_f) + 3 < len(kn_q))

    print("PASS" if not FAILS else f"FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
