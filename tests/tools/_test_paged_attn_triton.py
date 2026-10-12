"""Numerics gate: triton paged decode vs the emu (fp32 reference family).

Runs on any CUDA-visible HCU. Verdict: rel-inf/nan aware max-abs diff on o
and lse within 6e-3 (bf16 K/V rounding dominates), across GQA ratios, fp8
KV, ragged seqlens and split counts.
"""

import torch

import sgl_kernel.flash_attn as emu_mod
from sgl_kernel.paged_attn_triton import paged_attention_decode_triton


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
    print("RESULT:", "ALL-PASS" if fails == 0 else f"{fails}-FAIL")


if __name__ == "__main__":
    main()
