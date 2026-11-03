"""flash_mla_sparse_fwd emu numerical parity test (DSA decode sparse path).

Builds a reference: for each query row pick topk token ids, gather from a
paged pool, dense-attention in fp32 with an eager inline implementation,
and compare against `sgl_kernel.flash_mla_sparse.flash_mla_sparse_fwd`.
Also exercises -1 tail padding, topk_length clipping, and attn_sink.

Run: python tests/kernels/test_flash_mla_sparse.py
"""

import sys

import torch

torch.manual_seed(7)


def _reference(q, kv_flat, idx, sm_scale, d_v, sink=None, topk_length=None):
    """Inline fp32 reference identical in semantics to the emu contract."""
    s_q, H, d_qk = q.shape
    idx64 = idx.to(torch.int64)
    topk = idx64.shape[1]
    valid = idx64 >= 0
    if topk_length is not None:
        ar = torch.arange(topk)[None, :]
        valid &= ar < topk_length.reshape(-1, 1).to(torch.int64)
    k_all = kv_flat[:, 0, :].to(torch.float32)
    qf = q.to(torch.float32)
    outs = torch.empty(s_q, H, d_v)
    for t in range(s_q):
        k = k_all[idx64[t]]  # [topk, d]
        s = (qf[t] @ k.T) * sm_scale  # [H, topk]
        s = s.masked_fill(~valid[t][None, :], float("-inf"))
        real_lse = torch.logsumexp(s, dim=-1)
        p = torch.softmax(s, dim=-1)
        if sink is not None:
            lse = torch.logaddexp(real_lse, sink)
            p = p * torch.exp(real_lse - lse).unsqueeze(-1)
        outs[t] = p @ k[:, :d_v]
    return outs


def _metric(o, ref):
    """Dual metric per test-suite.md: cos (layout bugs destroy it) + maxrel."""
    of, rf = o.to(torch.float32).flatten(), ref.flatten()
    denom = of.norm() * rf.norm()
    cos = (of @ rf / denom).item() if denom > 0 else 0.0
    scale = rf.abs().max().item() or 1.0
    maxrel = ((of - rf).abs() / scale).max().item()
    return cos, maxrel


def main():
    from sgl_kernel.flash_mla_sparse import flash_mla_sparse_fwd

    dev = torch.device("cpu")  # pure-torch emu: identical math on CPU
    s_q, H, d_qk, d_v = 8, 3, 576, 512
    pool_tokens, topk = 300, 16
    kv = torch.randn(pool_tokens, 1, d_qk, dtype=torch.bfloat16)
    q = torch.randn(s_q, H, d_qk, dtype=torch.bfloat16)

    # random distinct indices; last two rows get -1 tails
    idx = torch.stack(
        [torch.randperm(pool_tokens)[:topk] for _ in range(s_q)]
    ).to(torch.int32)
    idx[-1, 8:] = -1
    idx[-2, 12:] = -1
    topk_length = torch.tensor([topk] * (s_q - 2) + [8, 12], dtype=torch.int32)

    scale = d_qk**-0.5
    sink = torch.randn(H) * 1e-3

    fails = 0
    for label, sink_on, tkl in (
        ("plain", None, None),
        ("sink", sink.clone(), None),
        ("sink-sentinel", torch.full((H,), -1e30), None),
        ("tailpad", None, topk_length.clone()),
        ("sink+tailpad", sink.clone(), topk_length.clone()),
    ):
        o, lse, _ = flash_mla_sparse_fwd(
            q=q.to(dev),
            kv=kv.to(dev),
            indices=idx.to(dev),
            sm_scale=scale,
            d_v=d_v,
            attn_sink=sink_on,
            topk_length=tkl,
        )
        ref = _reference(q, kv, idx, scale, d_v, sink_on, tkl)
        cos, maxrel = _metric(o, ref)
        ok = cos >= 0.9999 and maxrel <= 5e-2
        status = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1
        print(
            f"[{status}] {label}: cos={cos:.6f} maxrel={maxrel:.2e}"
        )
    print(f"flash_mla_sparse emu: {'5/5 PASS' if fails == 0 else f'{5-fails}/5 PASS'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
