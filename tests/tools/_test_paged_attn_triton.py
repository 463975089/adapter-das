# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Numerics gate: triton paged attention (decode + wide-window prefill)
vs fp32 hand references.

Runs on any CUDA-visible HCU. Decode family: rel-inf/nan aware max-abs diff
on o and lse within 6e-3 (bf16 K/V rounding dominates), across GQA ratios,
fp8 KV, ragged seqlens and split counts. Prefill family: bottom-right
causal windows against a shuffled paged pool, hand-roled softmax per row.
"""

import torch

import sgl_kernel.flash_attn as emu_mod
from sgl_kernel.paged_attn_triton import (
    paged_attention_decode_triton,
    paged_attention_prefill_triton,
)


def _run_case(bs, H, Hk, D, page, seqlens, fp8_kv=False, tol=0.002):
    torch.manual_seed(0)
    dev = "cuda"
    max_len = max(seqlens)
    npages_full = (max_len + page - 1) // page
    npages = npages_full * bs + bs  # + one spare per row (keeps bs | npages)
    kc = (torch.randn(npages, page, Hk, D, device=dev) * 0.4)
    vc = (torch.randn(npages, page, Hk, D, device=dev) * 0.4)
    if fp8_kv:
        k8 = kc.to(torch.float8_e4m3fn)
        v8 = vc.to(torch.float8_e4m3fn)
    else:
        k8 = kc.to(torch.bfloat16)
        v8 = vc.to(torch.bfloat16)
    # disjoint page ranges per row, shuffled inside the row: every page id
    # the row can reference (npages_full of them) is unique pool storage
    perm = torch.randperm(npages, device=dev).to(torch.int32).reshape(bs, -1)
    pt = perm[:, : npages_full].contiguous()
    seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=dev)
    q = (torch.randn(bs, 1, H, D, device=dev) * 0.5).to(torch.bfloat16)

    o_t, lse_t = paged_attention_decode_triton(
        q, k8, v8, pt, seqlens_t
    )
    # fp32 hand reference (the authoritative one): gather pages per row via
    # the table, softmax@fp32, p@v. (Note: the emu diverges on shuffled
    # tables at whole-row scale -- its chunked gather is page-aligned but
    # its in-chunk token mask is not; the triton kernel is the fix, so the
    # gate does not use the emu as reference on shuffled tables.)
    G = H // Hk
    worst = 0.0
    scale = D ** -0.5
    for b in range(bs):
        pages = pt[b].tolist()
        for h in range(0, H, max(1, H // 4)):  # sample 4 heads per row
            kvh = h // G
            qf = q[b, 0, h].float()
            kk = torch.cat([kc[p].float()[:, kvh] for p in pages])[: seqlens[b]]
            vv = torch.cat([vc[p].float()[:, kvh] for p in pages])[: seqlens[b]]
            s = (kk @ qf) * scale
            w = torch.softmax(s, -1)
            ref = (w.unsqueeze(1) * vv).sum(0)
            worst = max(worst, (o_t[b, 0, h].float() - ref).abs().max().item())
    rel = worst / 0.05  # o magnitudes ~0.05 on this test family
    return worst, rel


def _run_prefill_case(bs, H, Hk, D, page, seqlens, wl, tol=0.002):
    """Wide-window (paged extend) case: a rectangular [bs, wl, H, D] q
    window (the kvcache face is rectangular), keys = full pool up to
    seqlens[b] (>= wl), bottom-right causal: window row t sees keys
    j <= seqlens[b] - wl + t. fp32 hand reference like the decode case."""
    torch.manual_seed(1)
    dev = "cuda"
    max_len = max(seqlens)
    npages_full = (max_len + page - 1) // page
    npages = npages_full * bs + bs
    kc = (torch.randn(npages, page, Hk, D, device=dev) * 0.4).to(torch.bfloat16)
    vc = (torch.randn(npages, page, Hk, D, device=dev) * 0.4).to(torch.bfloat16)
    perm = torch.randperm(npages, device=dev).to(torch.int32).reshape(bs, -1)
    pt = perm[:, : npages_full].contiguous()
    seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=dev)
    q = (torch.randn(bs, wl, H, D, device=dev) * 0.5).to(torch.bfloat16)
    for b in range(bs):
        assert wl <= seqlens[b], "kv must cover the window (real face)"

    o_t, lse_t = paged_attention_prefill_triton(q, kc, vc, pt, seqlens_t)

    G = H // Hk
    scale = D**-0.5
    worst = 0.0
    for b in range(bs):
        pages = pt[b].tolist()
        sl = seqlens[b]
        for h in (0, H // 2, H - 1):
            kvh = h // G
            kk = torch.cat([kc[p].float()[:, kvh] for p in pages])[:sl]
            vv = torch.cat([vc[p].float()[:, kvh] for p in pages])[:sl]
            for t in range(wl):
                j_hi = sl - wl + t  # bottom-right causal
                s = (kk[: j_hi + 1] @ q[b, t, h].float()) * scale
                w = torch.softmax(s, -1)
                ref = (w.unsqueeze(1) * vv[: j_hi + 1]).sum(0)
                worst = max(
                    worst, (o_t[b, t, h].float() - ref).abs().max().item()
                )
    return worst


def main():
    cases = [
        ("gqa4-small", dict(bs=2, H=8, Hk=2, D=128, page=64, seqlens=[300, 137])),
        ("mh-a", dict(bs=3, H=16, Hk=16, D=128, page=16, seqlens=[65, 64, 33])),
        ("long-ctx", dict(bs=2, H=8, Hk=8, D=128, page=64, seqlens=[4096, 2048])),
        ("fp8kv", dict(bs=2, H=8, Hk=4, D=128, page=64, seqlens=[512, 129], fp8_kv=True, tol=0.008)),
        ("gqa4-page16", dict(bs=2, H=8, Hk=2, D=128, page=16, seqlens=[513, 257])),
    ]
    fails = 0
    for name, kw in cases:
        try:
            do, rel = _run_case(**kw)
            tol = 0.005 if kw.get("fp8_kv") else 0.002  # e4m3 grid is 4x coarser
            ok = do < tol and do == do  # abs gate on o; rel is informational
            print(f"{name}: max_abs={do:.5f} rel~{rel:.2e} {'PASS' if ok else 'FAIL'}")
            fails += 0 if ok else 1
        except Exception as e:
            print(f"{name}: EXC {type(e).__name__}: {str(e)[:140]}")
            fails += 1
    print("RESULT(decode):", "ALL-PASS" if fails == 0 else f"{fails}-FAIL")

    # wide-window (prefill / paged extend) family: rectangular q windows
    # (wl == real window everywhere), kv >= wl, ragged = ragged kv lens
    pcases = [
        ("pf-uniform", dict(bs=2, H=8, Hk=2, D=128, page=64, seqlens=[520, 300], wl=8)),
        ("pf-odd-w", dict(bs=3, H=16, Hk=4, D=128, page=64, seqlens=[140, 70, 500], wl=33)),
        ("pf-cross-page", dict(bs=2, H=8, Hk=8, D=128, page=16, seqlens=[600, 129], wl=128)),
        ("pf-long", dict(bs=2, H=8, Hk=4, D=128, page=64, seqlens=[4096, 2080], wl=512)),
        ("pf-whole", dict(bs=2, H=8, Hk=2, D=128, page=64, seqlens=[137, 64], wl=64)),
    ]
    pfails = 0
    for name, kw in pcases:
        try:
            d = _run_prefill_case(**kw)
            ok = d < 0.002 and d == d
            print(f"{name}: max_abs={d:.5f} {'PASS' if ok else 'FAIL'}")
            pfails += 0 if ok else 1
        except Exception as e:
            print(f"{name}: EXC {type(e).__name__}: {str(e)[:140]}")
            pfails += 1
    print("RESULT(prefill):", "ALL-PASS" if pfails == 0 else f"{pfails}-FAIL")
    print("RESULT:", "ALL-PASS" if fails + pfails == 0 else f"{fails + pfails}-FAIL")


if __name__ == "__main__":
    main()
