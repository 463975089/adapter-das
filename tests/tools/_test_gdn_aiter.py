# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Numerics gate: GDN aiter route vs the upstream triton fla kernels.

Findings baked in (2026-10-13): h and v_new are bit-exact (dh=dv=0) at
every probed shape on gfx936 (bw7); gfx938 (nmz26) shows fp32-ordering
divergence growing with accumulation depth — dh up to ~2e-3 and dw up to
~3e-5 at bs=4/T=256 (same kernel family, different wave scheduling).
State pools are fp32 on BOTH sides (sglang ssm pool is fp32); a/b/A_log/
dt_bias are fp32; H must equal HV in the packed-decode layout on these
tests.
"""

import torch

from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.chunk_fwd import (
    chunk_gated_delta_rule_fwd_intra,
)
from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
import aiter.ops.fla as afla
from aiter.ops.triton.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode as aiter_decode,
)


def _run_chunk_case(bs, T, H, K=128, V=128):
    torch.manual_seed(5)
    k = torch.randn(bs, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.3
    v = torch.randn(bs, T, H, V, device="cuda", dtype=torch.bfloat16) * 0.3
    beta = torch.rand(bs, T, H, device="cuda") * 0.9 + 0.05
    g = -torch.rand(bs, T, H, device="cuda").exp() * 0.1
    gc = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=None)
    w, u, _ = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=gc, beta=beta, cu_seqlens=None, chunk_size=64)

    idx = torch.arange(bs, device="cuda", dtype=torch.int32)
    p0 = torch.randn(bs, H, V, K, device="cuda", dtype=torch.float32) * 0.2
    p1 = p0.clone()
    h0, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=gc, gk=None,
        initial_state=p0, initial_state_indices=idx,
        save_new_value=True, cu_seqlens=None,
    )
    h1, _ = afla.chunk_gated_delta_rule_fwd_sglang_hip_blockdim64(
        k=k, w=w, u=u, g=gc, gk=None,
        initial_state=p1, initial_state_indices=idx,
        output_final_state=True, chunk_size=64, save_new_value=True,
        cu_seqlens=None, transpose_state_layout=True,
    )
    dh = (h0.float() - h1.float()).abs().max().item()
    dw = (p0 - p1).abs().max().item()
    return dh, dw


def _run_decode_case(bs, H, K=128, V=128):
    torch.manual_seed(11)
    HV = H
    qkv_dim = 2 * H * K + HV * V
    mixed_qkv = torch.randn(bs, qkv_dim, device="cuda",
                            dtype=torch.bfloat16) * 0.5
    a = torch.randn(bs, HV, device="cuda") * 0.1
    b = torch.randn(bs, HV, device="cuda") * 0.1
    A_log = torch.randn(HV, device="cuda") * 0.1
    dt_bias = torch.randn(HV, device="cuda") * 0.1
    scale = K**-0.5
    slots = bs * 3
    pool = torch.randn(slots, HV, V, K, device="cuda",
                       dtype=torch.float32) * 0.3
    idx = torch.randperm(slots, device="cuda")[:bs].to(torch.int32)

    o0 = torch.empty(bs, 1, HV, V, device="cuda", dtype=torch.bfloat16)
    p0 = pool.clone()
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv, a, b, A_log, dt_bias, scale, p0, o0, idx,
        use_qk_l2norm_in_kernel=True,
    )
    p1 = pool.clone()
    o1 = torch.empty(bs, 1, HV, V, device="cuda", dtype=torch.bfloat16)
    aiter_decode(
        mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        scale=scale, initial_state=p1, out=o1, ssm_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
    )
    do = (o0.float() - o1.float()).abs().max().item()
    ds = (p0[idx.long()].float() - p1[idx.long()].float()).abs().max().item()
    return do, ds


def main():
    fails = 0
    for name, kw in [
        ("chunk-b1", dict(bs=1, T=64, H=8)),
        ("chunk-b2", dict(bs=2, T=192, H=8)),
        ("chunk-h32", dict(bs=2, T=128, H=32)),
        ("chunk-gh32", dict(bs=4, T=256, H=32)),
    ]:
        try:
            dh, dw = _run_chunk_case(**kw)
            # gfx936 is bit-exact; gfx938 diverges at fp32-ordering level
            # (dh~2e-3, dw~3e-5 worst probed) — one gate, card-aware band.
            ok = dh < 4e-3 and dw < 1e-4 and dh == dh
            print(f"{name}: dh={dh:.6f} dw={dw:.2e} "
                  f"{'PASS' if ok else 'FAIL'}")
            fails += 0 if ok else 1
        except Exception as e:
            print(f"{name}: EXC {type(e).__name__}: {str(e)[:140]}")
            fails += 1
    for name, kw in [
        ("decode-b1", dict(bs=1, H=8)),
        ("decode-b8", dict(bs=8, H=32)),
    ]:
        try:
            do, ds = _run_decode_case(**kw)
            ok = do < 6e-3 and ds < 3e-2 and do == do
            print(f"{name}: dout={do:.5f} dstate={ds:.5f} "
                  f"{'PASS' if ok else 'FAIL'}")
            fails += 0 if ok else 1
        except Exception as e:
            print(f"{name}: EXC {type(e).__name__}: {str(e)[:140]}")
            fails += 1
    print("RESULT:", "ALL-PASS" if fails == 0 else f"{fails}-FAIL")


if __name__ == "__main__":
    main()
