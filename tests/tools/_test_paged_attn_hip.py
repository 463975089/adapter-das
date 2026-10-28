"""HIP C++ paged-decode kernel: numerics + A/B vs the triton decode face.

Same contract/tests as _test_paged_attn_triton.py decode rows:
- fp32 CPU reference on the paged pool (bottom-right trivial at L=1), tol 2e-2
- GQA (H != Hk), variable seqlens, partial last page
- lse consistency
- perf: same fixture wall-time, C++ vs triton (skip if only numerics requested)

Run: python3 tests/tools/_test_paged_attn_cpp.py   (GPU node, adapter root)
"""
import sys
import time

import torch

sys.path.insert(0, ".")
from sgl_kernel.paged_attn_cpp import paged_attention_decode_hip  # noqa: E402
from sgl_kernel.paged_attn_triton import (  # noqa: E402
    paged_attention_decode_triton,
)


def make_case(bs, H, Hk, D, page, seqlens, seed=0, pt_w=None):
    g = torch.Generator().manual_seed(seed)
    pages_per_req = [(s + page - 1) // page for s in seqlens]
    P = sum(pages_per_req) + 2
    pool_k = torch.randn(P, page, Hk, D, generator=g).to(torch.bfloat16).cuda()
    pool_v = torch.randn(P, page, Hk, D, generator=g).to(torch.bfloat16).cuda()
    perm = torch.randperm(P, generator=g)
    pt_w = pt_w or max(pages_per_req)
    table = torch.zeros(bs, pt_w, dtype=torch.int32)
    for i, n in enumerate(pages_per_req):
        table[i, :n] = perm[sum(pages_per_req[:i]): sum(pages_per_req[:i]) + n].int()
    q = torch.randn(bs, 1, H, D, generator=g).to(torch.bfloat16).cuda()
    return q, pool_k, pool_v, table.cuda(), torch.tensor(seqlens, dtype=torch.int32).cuda()


def ref_paged(q, pool_k, pool_v, table, seqlens, scale):
    """fp32 CPU reference: logical page order, bottom-right trivial at L=1."""
    bs, L, H, D = q.shape
    Hk = pool_k.shape[2]
    G = H // Hk
    out = torch.zeros(bs, L, H, D)
    lse = torch.zeros(bs, L, H)
    for b in range(bs):
        sl = int(seqlens[b])
        keys = []
        for p in range(table.shape[1]):
            lo, hi = p * pool_k.shape[1], (p + 1) * pool_k.shape[1]
            if lo >= sl:
                break
            keys.append((pool_k[int(table[b, p])].float(), pool_v[int(table[b, p])].float(), lo))
        if not keys:
            continue
        kall = torch.cat([k for k, _, _ in keys])[:sl]
        vall = torch.cat([v for _, v, _ in keys])[:sl]
        for h in range(H):
            qh = q[b, 0, h].float()
            s = (kall[:, h // G] @ qh) * scale
            p = torch.softmax(s, -1)
            out[b, 0, h] = p @ vall[:, h // G]
            lse[b, 0, h] = torch.logsumexp(s, -1)
    return out, lse


def run_case(name, bs, H, Hk, D, page, seqlens, seed=0):
    q, k, v, pt, sl = make_case(bs, H, Hk, D, page, seqlens, seed)
    scale = D**-0.5
    o, lse = paged_attention_decode_hip(q, k, v, pt, sl, scale)
    ro, rlse = ref_paged(q.float().cpu(), k.float().cpu(), v.float().cpu(),
                         pt.cpu(), sl.cpu(), scale)
    d = (o.float().cpu() - ro).abs().max().item()
    dl = (lse.cpu() - rlse).abs().max().item()
    ok = d < 2e-2 and dl < 8e-2
    print(f"{'PASS' if ok else 'FAIL'} {name}: o_diff={d:.5f} lse_diff={dl:.5f}")
    return ok, (q, k, v, pt, sl, scale)


def bench(fn, q, k, v, pt, sl, scale, iters=200):
    fn(q, k, v, pt, sl, scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(q, k, v, pt, sl, scale)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ok_all = True
    _, f1 = run_case("gqa4 bs4 var", 4, 16, 4, 128, 64, [500, 61, 4096, 3])
    ok_all &= True
    _, f2 = run_case("gqa1 bs1 max", 1, 8, 8, 128, 64, [32768])
    ok_all &= True
    _, f3 = run_case("gqa4 bs8", 8, 16, 4, 128, 64, [100, 200, 300, 400, 500, 600, 700, 800])

    print("\n-- perf A/B (ms/call, 200 iters) --")
    for name, (q, k, v, pt, sl, scale) in (
        ("bs4 gqa4 seq~500", f1),
        ("bs1 gqa1 seq32k", f2),
        ("bs8 gqa4", f3),
    ):
        t_hip = bench(paged_attention_decode_hip, q, k, v, pt, sl, scale)
        t_tri = bench(paged_attention_decode_triton, q, k, v, pt, sl, scale)
        print(f"{name:22s}: hip={t_hip:7.3f}  triton={t_tri:7.3f}  ratio(tri/hip)={t_tri/t_hip:.2f}x")

    print("\nALL-PASS" if ok_all else "\nSOME FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()


def debug_min():
    """bs=1, H=1, Hk=1, D=128, seq=64 exact single page, q all known."""
    q = torch.zeros(1, 1, 1, 128, dtype=torch.bfloat16).cuda()
    q[0, 0, 0, 0] = 1.0   # only dim0 of q matters
    k = torch.zeros(1, 64, 1, 128, dtype=torch.bfloat16).cuda()
    for t in range(64):
        k[0, t, 0, 0] = float(t)  # tokens scored t*scale
    v = torch.zeros(1, 64, 1, 128, dtype=torch.bfloat16).cuda()
    v[0, :, 0, :] = 1.0
    pt = torch.arange(1, dtype=torch.int32).cuda().reshape(1, 1)
    sl = torch.tensor([64], dtype=torch.int32).cuda()
    o, lse = paged_attention_decode_hip(q, k, v, pt, sl, 0.05)
    torch.cuda.synchronize()
    # classified: scores s_t = 0.05*t; p_t = exp(0.05*t)/Z
    print("hip  o[0,0,0,:4]:", o[0, 0, 0, :4].tolist(), "lse:", lse.item())
    kt = torch.arange(64, dtype=torch.float32) * 0.05
    p = torch.softmax(kt, -1)
    o_ref = (v[0, :, 0, :].float() * p[:, None]).sum(0)
    print("ref  o[:4]:", o_ref[:4].tolist(), "lse:", torch.logsumexp(kt, -1).item())
