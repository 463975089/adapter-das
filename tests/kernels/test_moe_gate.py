# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MoE gate kernels (topk_softmax / moe_fused_gate), GPU contracts vs torch.

Run on a node with a working GPU: `pytest tests/kernels/test_moe_gate.py -v`.
Skipped automatically when no GPU is visible.
"""

import pytest
import torch

import sgl_kernel

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)


@requires_gpu
def test_topk_softmax_gpu_matches_cpu():
    torch.manual_seed(6)
    M, E, K = 64, 256, 8
    logits = (torch.randn(M, E, device="cuda") * 4).to(torch.bfloat16)
    bias = torch.randn(E, device="cuda", dtype=torch.float32)
    w = torch.empty(M, K, device="cuda", dtype=torch.float32)
    ids = torch.empty(M, K, device="cuda", dtype=torch.int32)
    sgl_kernel.topk_softmax(w, ids, logits, renormalize=True,
                            moe_softcapping=0.0, correction_bias=bias)

    lf = logits.float()
    act = torch.softmax(lf, dim=-1)
    _, rids = torch.topk(lf + bias, k=K, dim=-1, sorted=False)
    rw = act.gather(1, rids)
    rw = rw / rw.sum(-1, keepdim=True)
    torch.testing.assert_close(
        torch.sort(w, dim=-1).values, torch.sort(rw, dim=-1).values, rtol=1e-3, atol=1e-4
    )


@requires_gpu
def test_moe_fused_gate_gpu_grouped():
    torch.manual_seed(8)
    M, E, K, G, KG = 32, 64, 8, 8, 4
    logits = torch.randn(M, E, device="cuda")
    bias = torch.randn(E, device="cuda")
    w, ids = sgl_kernel.moe_fused_gate(
        logits, bias, topk=K, scoring_func="sigmoid", renormalize=True,
        num_expert_group=G, topk_group=KG,
    )
    assert w.shape == (M, K) and ids.shape == (M, K)
    scores = torch.sigmoid(logits)
    biased = scores + bias
    gs = biased.view(M, G, E // G).topk(2, dim=-1).values.sum(-1)
    keep = gs.topk(KG, dim=-1, sorted=False).indices
    gm = torch.zeros_like(gs, dtype=torch.bool).scatter(1, keep, True)
    em = gm.unsqueeze(-1).expand(M, G, E // G).reshape(M, E)
    masked = biased.masked_fill(~em, float("-inf"))
    _, rids = torch.topk(masked, k=K, dim=-1, sorted=False)
    rw = scores.gather(1, rids)
    rw = rw / rw.sum(-1, keepdim=True)
    torch.testing.assert_close(
        torch.sort(w, dim=-1).values, torch.sort(rw, dim=-1).values, rtol=1e-4, atol=1e-5
    )
    assert torch.equal(torch.sort(ids, dim=-1).values, torch.sort(rids, dim=-1).values)
