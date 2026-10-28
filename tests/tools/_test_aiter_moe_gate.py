"""HCU aiter fused-MoE bridge gate (GPU, standalone).

Validates, on any HCU node with the DTK aiter wheel:
  A. bridge picks the aiter kernel for a tuned blockwise-fp8 shape
     (E=256 / inter=512 / K=4096 / topk=8 -> ASM solution) and the result
     matches a dequantized bf16 reference within fp8-noise tolerance;
  B. bridge returns None (-> stock triton fallback) for a shape absent
     from the tuned tables (GLM-5.3-Flash TP8: E=288 inter 256) and for
     mode=triton, without touching aiter;
  C. mode knob round-trips through hcu_env.

Run:  HIP_VISIBLE_DEVICES=0 python tests/tools/_test_aiter_moe_gate.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

FAILS = []


class _Cfg:
    inplace = False
    activation = "silu"
    no_combine = False
    routed_scaling_factor = None
    gemm1_alpha = None
    gemm1_clamp_limit = None


def _make_blockwise(E, N1, N2, K, NI, M, TOPK, dev):
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234 + E)
    x = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) / 8)

    def make_q(n_e, n_out, n_in):
        s = torch.rand(n_e, n_out // 128, n_in // 128, device=dev,
                       generator=gen, dtype=torch.bfloat16).float() * 0.05 + 0.005
        q = (torch.rand(n_e, n_out, n_in, device=dev, generator=gen,
                        dtype=torch.bfloat16).float() * 240 - 120
             ).to(torch.float8_e4m3fn)
        return q, s

    w1q, w1s = make_q(E, N1, K)
    w2q, w2s = make_q(E, N2, NI)
    tw = torch.full((M, TOPK), 1.0 / TOPK, device=dev, dtype=torch.float32)
    ti = torch.stack(
        [torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]
    ).to(torch.int32)
    return x, w1q, w1s, w2q, w2s, tw, ti


def _ref_blockwise(x, w1q, w1s, w2q, w2s, tw, ti, NI, TOPK):
    M, K = x.shape
    E, N1, _ = w1q.shape
    N2 = w2q.shape[1]
    tok_ids = ti.reshape(-1).long()
    order = torch.argsort(tok_ids)
    h_all = x.repeat_interleave(TOPK, 0)[order]
    w_all = tw.reshape(-1, 1)[order]
    out_sorted = torch.zeros(M * TOPK, K, device=x.device, dtype=torch.float32)
    sorted_ids = tok_ids[order]
    ptr = 0
    while ptr < sorted_ids.numel():
        e = int(sorted_ids[ptr])
        end = ptr
        while end < sorted_ids.numel() and int(sorted_ids[end]) == e:
            end += 1
        h = h_all[ptr:end].float()
        w1f = (w1q[e].float().view(N1 // 128, 128, K // 128, 128)
               * w1s[e].view(N1 // 128, 1, K // 128, 1)).view(N1, K).to(torch.bfloat16)
        aa = (h @ w1f.float().t()).to(torch.bfloat16).float()
        a = torch.nn.functional.silu(aa[:, :NI]) * aa[:, NI:]
        w2f = (w2q[e].float().view(N2 // 128, 128, NI // 128, 128)
               * w2s[e].view(N2 // 128, 1, NI // 128, 1)).view(N2, NI).to(torch.bfloat16)
        out_sorted[ptr:end] = a @ w2f.float().t()
        ptr = end
    out_sorted *= w_all
    unorder = torch.empty_like(order)
    unorder[order] = torch.arange(M * TOPK, device=x.device)
    return out_sorted[unorder].view(M, TOPK, K).sum(1)


def main():
    from hcu_adapter import aiter_moe
    from hcu_adapter.hcu_env import resolve

    dev = "cuda"
    M, TOPK = 512, 8

    # A: tuned blockwise shape -> aiter kernel, numerics vs bf16 ref
    E, NI, K = 256, 512, 4096
    N1, N2 = 2 * NI, K
    x, w1q, w1s, w2q, w2s, tw, ti = _make_blockwise(E, N1, N2, K, NI, M, TOPK, dev)
    out = aiter_moe.try_aiter_fused_moe(
        hidden_states=x, w1=w1q, w2=w2q, topk_output=(tw, ti, None),
        moe_runner_config=_Cfg(), use_fp8_w8a8=True, use_int8_w8a8=False,
        w1_scale=w1s, w2_scale=w2s, a1_scale=None, block_shape=[128, 128],
    )
    if out is None:
        FAILS.append("A-noaiter (kernel declined tuned shape)")
    else:
        ref = _ref_blockwise(x, w1q, w1s, w2q, w2s, tw, ti, NI, TOPK)
        d = (out.float() - ref).abs()
        rel = (d.sum() / ref.abs().sum()).item()
        print(f"A-aiter-asm: rel={rel:.3f}")
        if not (rel < 0.05):
            FAILS.append(f"A-rel-{rel:.3f}")

    # B1: GLM TP8 shape absent from tuned tables -> None (triton fallback)
    E2, NI2 = 288, 256
    N1b, N2b = 2 * NI2, K
    x2, w1q2, w1s2, w2q2, w2s2, tw2, ti2 = _make_blockwise(
        E2, N1b, N2b, K, NI2, M, TOPK, dev)
    out2 = aiter_moe.try_aiter_fused_moe(
        hidden_states=x2, w1=w1q2, w2=w2q2, topk_output=(tw2, ti2, None),
        moe_runner_config=_Cfg(), use_fp8_w8a8=True, use_int8_w8a8=False,
        w1_scale=w1s2, w2_scale=w2s2, a1_scale=None, block_shape=[128, 128],
    )
    print(f"B1-glm-shape-fallback: {'OK' if out2 is None else 'USED-AITER'}")
    if out2 is not None:
        FAILS.append("B1-should-fallback")

    # B2: mode=triton -> None without importing aiter
    os.environ["HCU_FUSED_MOE_IMPL"] = "triton"
    out3 = aiter_moe.try_aiter_fused_moe(
        hidden_states=x, w1=w1q, w2=w2q, topk_output=(tw, ti, None),
        moe_runner_config=_Cfg(), use_fp8_w8a8=True, use_int8_w8a8=False,
        w1_scale=w1s, w2_scale=w2s, a1_scale=None, block_shape=[128, 128],
    )
    print(f"B2-mode-triton: {'OK' if out3 is None else 'USED-AITER'}")
    if out3 is not None:
        FAILS.append("B2-mode-ignored")
    del os.environ["HCU_FUSED_MOE_IMPL"]

    # C: knob resolves through hcu_env
    v, src = resolve("HCU_FUSED_MOE_IMPL")
    print(f"C-resolve: {v}({src})")
    if v != "auto":
        FAILS.append(f"C-resolve-{v}")

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
