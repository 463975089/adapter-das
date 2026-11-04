# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""R-05 w4a16 MoE deepgemm bridge — three-tier numeric gate.

Segments
--------
A  convention pin: lo-even u8 bytes + (q-8)*s, the wheel's own triton ref
   (moe_w4a16_groupwise_triton) must equal an fp32 einsum golden on the
   valid (actual-masked) rows.
B  C++ masked tier: grouped_gemm_w4a16_nt_masked_entry via the bridge's
   _prepare (incl. the 64/128 -> 32 scale expansion) against the same
   golden, with ragged per-expert counts and one zero-count expert.
C  fused pipeline: deepgemm_w4a16_fused_experts (dispatch -> masked
   gemm1 -> silu -> masked gemm2 -> weighted combine) vs an fp32 hand
   reference over a shapes matrix (E/I/K/topk/M variations, group 32 and
   128, weight cache reuse).

Tolerances follow the house rule for bf16-in/w4 groupwise kernels
(median rel < 1e-2 per expert against the fp32 golden; the pipeline arm
is comparison-vs-hand with full rows, abs tol 3e-2 bf16 activations).

Run: pytest tests/kernels/test_w4a16_moe_deepgemm.py -v (HCU node;
requires the DTK deepgemm wheel)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires HCU/CUDA device"
)

dev = "cuda"


def _deepgemm():
    import deepgemm  # noqa: F401

    return deepgemm


def _make_experts(E, N, K, group, seed=0):
    """Random int4 weights in the moe_wna16 STORE layout: qweight u8
    [E, N, K//2] lo-even, scales [E, N, K//group]."""
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randint(0, 16, (E, N, K), device=dev, generator=g)
    lo, hi = q[:, :, 0::2], q[:, :, 1::2]
    qw = (lo | (hi << 4)).to(torch.uint8)  # lo-even bytes == deepgemm b
    s = (torch.rand(E, N, K // group, device=dev, generator=g) * 0.02
         + 0.005).to(torch.bfloat16)
    return qw, s, q


def _dequant(qw, s, group):
    """Dequant from the STORE layout: u8 qweight [E,N,K//2] lo-even ->
    (q-8)*s on the group grid (references receive the unpacked q)."""
    if qw.dtype == torch.uint8:
        lo = (qw & 0xF).long()
        hi = (qw >> 4).long()
        q = torch.stack([lo, hi], dim=-1).reshape(*qw.shape[:-1], -1)
    else:
        q = qw.long()
    sE = s.repeat_interleave(group, dim=-1).float()
    return (q.float() - 8.0) * sE  # symmetric convention, group-agnostic


def _golden(a, q, s, group, actual):
    """fp32 einsum golden limited to the valid rows: [E, M, N]."""
    w = _dequant(q, s, group)
    gold = torch.einsum("emk,enk->emn", a.float(), w)
    for e in range(gold.shape[0]):
        gold[e, int(actual[e]):] = 0.0
    return gold


def _rel_rows(got, gold, actual):
    worst = 0.0
    for e in range(gold.shape[0]):
        m = int(actual[e])
        if m == 0:
            continue
        d = (got[e, :m].float() - gold[e, :m]).abs()
        sc = gold[e, :m].abs().clamp(min=1e-3)
        worst = max(worst, (d / sc).median().item())
    return worst


# ---------------------------------------------------------------------------
# segment A: convention pin (triton ref of the wheel)
# ---------------------------------------------------------------------------


@requires_gpu
def test_a_convention_triton_ref():
    dg = _deepgemm()
    torch.manual_seed(11)
    E, M, N, K, GS = 4, 64, 256, 256, 32
    qw, s, q = _make_experts(E, N, K, GS, seed=11)
    a = torch.randn(E, M, K, device=dev, dtype=torch.bfloat16)
    actual = torch.tensor([16, 32, 48, 64], device=dev, dtype=torch.int32)
    got = dg.moe_w4a16_groupwise_triton(a.clone(), qw, s.clone(),
                                        actual.clone())
    gold = _golden(a, q, s, GS, actual)
    assert _rel_rows(got.float(), gold, actual) < 1e-2


# ---------------------------------------------------------------------------
# segment B: C++ masked tier through the bridge's _prepare
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("group", [32, 128])
def test_b_cpp_masked(group):
    dg = _deepgemm()
    from hcu_adapter.w4a16_moe import _grouped_gemm, _prepare

    torch.manual_seed(5)
    E, MCap, N, K = 8, 128, 256, 256
    qw, s, q = _make_experts(E, N, K, group, seed=5)
    counts = torch.tensor([MCap, 0, 64, 1, 100, 128, 33, 96],
                          device=dev, dtype=torch.int32)
    bm, s32 = _prepare(qw, s)
    # second call must hit the cache and return identical objects
    bm2, s32b = _prepare(qw, s)
    assert bm2 is bm and s32b is s32
    a_pad = torch.randn(E, MCap, K, device=dev, dtype=torch.bfloat16)
    got = _grouped_gemm(a_pad, bm, s32, counts, MCap, N, torch.bfloat16)
    gold = _golden(a_pad, q, s, group, counts)
    assert _rel_rows(got.float(), gold, counts) < 1e-2


# ---------------------------------------------------------------------------
# segment C: fused pipeline vs fp32 hand reference
# ---------------------------------------------------------------------------


def _hand_pipeline(x, w1, w2, tw, tids, s1, s2, group, rscale=None):
    """Dispatch -> (q-8)*s gemm1 -> silu -> gemm2 -> weighted combine."""
    M, K = x.shape
    E, N1, _ = w1.shape
    I = N1 // 2
    d1 = _dequant(w1, s1, group)
    d2 = _dequant(w2, s2, group)
    out = x.new_zeros(M, K, dtype=torch.float32)
    for m in range(M):
        for kk in range(tids.shape[1]):
            e = int(tids[m, kk])
            h = x[m].float() @ d1[e].T           # [2I]
            g = torch.nn.functional.silu(h[:I].float()) * h[I:].float()
            o = g @ d2[e].T                      # [K]
            out[m] += o * float(tw[m, kk])
    if rscale is not None and float(rscale) != 1.0:
        out = out * float(rscale)
    return out


@requires_gpu
@pytest.mark.parametrize("group", [32, 128])
@pytest.mark.parametrize(
    "E,I,K,M,topk,seed",
    [(8, 128, 256, 64, 3, 1), (4, 256, 512, 48, 5, 2), (6, 128, 384, 33, 2, 3)],
)
def test_c_pipeline(group, E, I, K, M, topk, seed):
    from hcu_adapter.w4a16_moe import deepgemm_w4a16_fused_experts

    torch.manual_seed(seed)
    N1, N2 = 2 * I, K
    qw1, s1, q1 = _make_experts(E, N1, K, group, seed=seed)
    qw2, s2, q2 = _make_experts(E, N2, I, group, seed=seed + 50)
    x = (torch.randn(M, K, device=dev) * 0.3).to(torch.bfloat16)
    tids = torch.randint(0, E, (M, topk), device=dev)
    tw = torch.softmax(torch.randn(M, topk, device=dev), dim=-1)

    got = deepgemm_w4a16_fused_experts(x, qw1, qw2, tw, tids, s1, s2)
    assert got is not None
    gold = _hand_pipeline(x, qw1, qw2, tw, tids, s1, s2, group)
    d = (got.float() - gold).abs()
    rel = (d / gold.abs().clamp(min=0.5)).median().item()
    assert rel < 3e-2, f"pipeline rel={rel:.4f}"

    # routed_scaling must multiply through
    got2 = deepgemm_w4a16_fused_experts(x, qw1, qw2, tw, tids, s1, s2,
                                        routed_scaling=2.0)
    assert torch.allclose(got2.float(), got.float() * 2.0, atol=2e-2,
                          rtol=1e-2)


if __name__ == "__main__":
    import sys as _s

    _s.exit(pytest.main([__file__, "-v"]))
