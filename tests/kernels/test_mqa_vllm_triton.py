# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""vllm-face MQA logits triton arms vs the torch emu (golden) — parity gates.

2026-11-05: HCU_MQA_EMU_IMPL=triton (default) routes the two vllm-face
entries (sglang.kernels.ops.attention.dsa.aiter_paged_mqa_logits on the
legacy page_size=1 pool; aiter.ops.triton.fp8_mqa_logits.fp8_mqa_logits
ragged) to the homegrown kernels in hcu_adapter/mqa_logits_triton.py.
The emu keeps its golden/fallback role (HCU_MQA_EMU_IMPL=emu, or per-face
on fnuz / non-132B pools).

Gates: random pools with SHUFFLED token-id block tables (the paged arm
must gather by id, not assume contiguity), dual metrics cos+maxdiff, and
top-k set equality on the paged face (the DSA consumer contract). CPU
runs skip the GPU arms.

Run: python tests/kernels/test_mqa_vllm_triton.py   (GPU; skip on CPU)
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _impls():
    from hcu_adapter import mqa_logits_emu, mqa_logits_triton

    return mqa_logits_emu, mqa_logits_triton


def _rand_pool(T, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = (torch.randn(T, 128, generator=g) * 0.5).to(torch.float8_e4m3fn)
    sc = torch.rand(T, generator=g) * 0.02 + 0.005  # fp32 per-token scale
    pool = torch.cat(
        [k.view(torch.uint8), sc.view(torch.uint8).reshape(T, 4)], dim=1
    ).to(device)
    return pool, sc.to(device)


def _paged_case(N, H, T, L, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(N, H, 128, generator=g) * 0.7).to(torch.float8_e4m3fn)
    q = q.to(device)
    w = (torch.rand(N, H, generator=g) * 0.1 + 0.95).to(device)
    pool, _ = _rand_pool(T, device, seed + 1)
    # shuffled token-id block table: page_size=1 world, ids arbitrary
    bt = torch.randperm(T, generator=g)[:L].reshape(1, L).repeat(N, 1)
    bt = bt.to(device, dtype=torch.int32)
    lens = torch.full((N,), L, dtype=torch.int32, device=device)
    return q, pool, w, lens, bt


def _paged_gate(emu, tri):
    if not (torch.cuda.is_available() and torch.version.hip):
        print("SKIP: no HIP GPU")
        return True
    # refuse-arms stay refused (the fallback contract anchors on them)
    try:
        tri.triton_paged_ps1_mqa_logits(
            torch.zeros(1, 1, 128, device="cuda"), torch.zeros(2, 132, dtype=torch.uint8, device="cuda"),
            torch.ones(1, 1, device="cuda"), torch.ones(1, dtype=torch.int32, device="cuda"),
            torch.zeros(1, 1, dtype=torch.int32, device="cuda"), 1,
            preshuffle=True, kv_block_size=1)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("preshuffle pool must be refused by the ps1 arm")
    return False


def main():
    emu, tri = _impls()
    if _paged_gate(emu, tri):
        return 0
    failures = 0
    for label, N, H, T, L in (
        ("decode-ish", 8, 32, 4096, 3000),
        ("prefill-ish", 3, 16, 8192, 8192),
    ):
        q, pool, w, lens, bt = _paged_case(N, H, T, L, "cuda", seed=hash(label) % 2**31)
        a = emu._paged_mqa_logits(
            q, pool, w, lens, bt, L, preshuffle=False, kv_block_size=1)
        b = tri.triton_paged_ps1_mqa_logits(
            q, pool, w, lens, bt, L, preshuffle=False, kv_block_size=1)
        cos = torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0).item()
        rel = (a - b).abs().max().item() / (a.abs().max().item() + 1e-30)
        ka = torch.topk(a[0], 16).indices.sort().values
        kb = torch.topk(b[0], 16).indices.sort().values
        topk_eq = torch.equal(ka, kb)
        ok = cos > 0.99999 and rel < 5e-3 and topk_eq
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} paged-parity[{label}] cos={cos:.9f} rel={rel:.2e} topk16={topk_eq}")

    # ragged face: [ks, ke) windows, -inf outside
    g = torch.Generator(device="cpu").manual_seed(7)
    N, T, H = 4, 4096, 24
    q = (torch.randn(N, H, 128, generator=g) * 0.7).to(torch.float8_e4m3fn).cuda()
    pool, _ = _rand_pool(T, "cuda", 9)
    k_u8 = pool[:, :128]
    sc = pool[:, 128:].view(torch.float32).reshape(-1).contiguous()
    w = (torch.rand(N, H, generator=g) * 0.1 + 0.95).cuda()
    ks = torch.tensor([0, 128, 1000, 2048], dtype=torch.int32, device="cuda")
    ke = torch.tensor([2048, 4096, 3500, 4096], dtype=torch.int32, device="cuda")
    a = emu._ragged_fp8_mqa_logits(q, k_u8, sc, w, ks, ke)
    b = tri.triton_ragged_mqa_logits(q, k_u8, sc, w, ks, ke)
    fin = torch.isfinite(a)
    assert torch.isfinite(b[fin]).all(), "finite region mismatch"
    assert torch.isinf(b[~fin]).all(), "-inf region mismatch"
    cos = torch.nn.functional.cosine_similarity(
        a[fin], b[fin], dim=0).item()
    rel = (a[fin] - b[fin]).abs().max().item() / (a[fin].abs().max().item() + 1e-30)
    ok = cos > 0.99999 and rel < 5e-3
    failures += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'} ragged-parity cos={cos:.9f} rel={rel:.2e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
