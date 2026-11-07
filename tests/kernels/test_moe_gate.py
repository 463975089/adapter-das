# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MoE gate contracts, including finite expert/group ties on CPU and HCU/CUDA.

Run on a node with a working GPU: `pytest tests/kernels/test_moe_gate.py -v`.
Device-only cases are skipped automatically when no GPU is visible.
"""

import math

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


def _tied_logits(dtype):
    """Finite router rows with ties both within and across the top-k boundary."""
    rows = torch.zeros(4, 256, dtype=dtype)
    view = rows[:, ::2]  # Exercise the public strided-input contract as well.
    view[1, 96:104] = 3
    view[1, 28] = view[1, 67] = 2
    view[2, 3:10] = 4
    view[2, 28] = view[2, 67] = 2
    view[3, ::3] = -1
    view[3, 19] = 2
    return view


def _softmax_tie_oracle(logits, renormalize):
    # Independent scalar ordering: descending logit, then lowest expert ID.
    # Both the community Triton router and power-of-two AOT CUDA kernel use
    # this tie rule; arbitrary torch.topk tie winners are not the oracle.
    expert_ids, weights = [], []
    for row in logits.float().cpu().tolist():
        selected = sorted(range(len(row)), key=lambda i: (-row[i], i))[:8]
        maximum = max(row)
        numerators = [math.exp(x - maximum) for x in row]
        denominator = math.fsum(
            numerators[i] for i in selected
        ) if renormalize else math.fsum(numerators)
        expert_ids.append(selected)
        weights.append([numerators[i] / denominator for i in selected])
    return torch.tensor(expert_ids, dtype=torch.int32), torch.tensor(weights)


def _softmax_gate(entry, logits, weights, ids, renormalize):
    if entry == "topk_softmax":
        sgl_kernel.topk_softmax(weights, ids, logits, renormalize=renormalize)
    else:
        result_weights, result_ids = sgl_kernel.moe_fused_gate(
            logits, None, topk=8, scoring_func="softmax",
            renormalize=renormalize,
        )
        weights.copy_(result_weights)
        ids.copy_(result_ids)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_gpu)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("entry", ["topk_softmax", "moe_fused_gate"])
def test_softmax_ties_choose_lowest_expert_id(device, dtype, renormalize, entry):
    cpu_logits = _tied_logits(dtype)
    storage = torch.empty(4, 256, device=device, dtype=dtype)
    logits = storage[:, ::2]
    logits.copy_(cpu_logits)
    weights = torch.empty(4, 8, device=device)
    ids = torch.empty(4, 8, device=device, dtype=torch.int32)
    expected_ids, expected_weights = _softmax_tie_oracle(cpu_logits, renormalize)
    for _ in range(3):
        _softmax_gate(entry, logits, weights, ids, renormalize)
        assert torch.equal(ids.cpu(), expected_ids)
        torch.testing.assert_close(
            weights.cpu(), expected_weights, rtol=1e-6, atol=1e-7
        )


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("entry", ["topk_softmax", "moe_fused_gate"])
def test_softmax_ties_follow_changed_graph_inputs(dtype, entry):
    cpu_logits = _tied_logits(dtype)
    logits = cpu_logits.to("cuda")
    weights = torch.empty(4, 8, device="cuda")
    ids = torch.empty(4, 8, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _softmax_gate(entry, logits, weights, ids, True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _softmax_gate(entry, logits, weights, ids, True)
    for shift in [0, 17, 49, 0]:
        changed = cpu_logits.roll(shift, dims=1)
        logits.copy_(changed)
        graph.replay()
        expected_ids, expected_weights = _softmax_tie_oracle(changed, True)
        assert torch.equal(ids.cpu(), expected_ids)
        torch.testing.assert_close(
            weights.cpu(), expected_weights, rtol=1e-6, atol=1e-7
        )


def _nonsoftmax_inputs(dtype, grouped):
    experts = 64 if grouped else 72
    storage = torch.zeros(4, experts * 2, dtype=dtype)
    logits = storage[:, ::2]
    logits[1, ::8] = 2
    logits[2, 3:7] = 2
    logits[3, ::3] = -2
    bias = torch.zeros(experts)
    bias[::8] = .125
    return logits, bias


def _nonsoftmax_oracle(logits, bias, scoring, grouped, renormalize, shared):
    weights, indices = [], []
    for row in logits.float().cpu().tolist():
        act = [(1 / (1 + math.exp(-x)) if scoring == "sigmoid"
                else math.sqrt(math.log1p(math.exp(x)))) for x in row]
        rank = [a + b for a, b in zip(act, bias.tolist())]
        eligible = list(range(len(row)))
        if grouped:
            width = len(row) // 4
            groups = [list(range(g * width, (g + 1) * width)) for g in range(4)]
            scores = [math.fsum(sorted((rank[i] for i in group), reverse=True)[:2])
                      for group in groups]
            keep = sorted(range(4), key=lambda g: (-scores[g], g))[:2]
            eligible = [i for g in keep for i in groups[g]]
        selected = sorted(eligible, key=lambda i: (-rank[i], i))[:6]
        values = [act[i] for i in selected]
        total = math.fsum(values)
        if shared:
            selected.append(len(row))
            values.append(total / 1.7)
        weights.append([v / (total if renormalize else 1) * 1.7 for v in values])
        indices.append(selected)
    return torch.tensor(weights), torch.tensor(indices, dtype=torch.int32)


def _nonsoftmax_gate(entry, logits, bias, weights, ids, renormalize, shared):
    if entry == "topk_sigmoid":
        sgl_kernel.topk_sigmoid(
            weights, ids, logits, renormalize=renormalize, correction_bias=bias,
            routed_scaling_factor=1.7, num_fused_shared_experts=int(shared),
        )
    else:
        result_weights, result_ids = sgl_kernel.moe_fused_gate(
            logits, bias, topk=6 + int(shared),
            scoring_func="sqrtsoftplus" if entry == "fused_sqrtsoftplus" else "sigmoid",
            renormalize=renormalize, num_fused_shared_experts=int(shared),
            num_expert_group=4 if entry == "fused_grouped_sigmoid" else 1,
            topk_group=2 if entry == "fused_grouped_sigmoid" else 1,
            routed_scaling_factor=1.7, apply_routed_scaling_factor_on_output=True,
        )
        weights.copy_(result_weights)
        ids.copy_(result_ids)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_gpu)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("entry", [
    "topk_sigmoid", "fused_sigmoid", "fused_grouped_sigmoid", "fused_sqrtsoftplus",
])
def test_nonsoftmax_ties_choose_lowest_expert_and_group(
        device, dtype, renormalize, shared, entry):
    grouped = entry == "fused_grouped_sigmoid"
    cpu_logits, cpu_bias = _nonsoftmax_inputs(dtype, grouped)
    logits = torch.empty(4, cpu_logits.shape[1] * 2, device=device, dtype=dtype)[:, ::2]
    logits.copy_(cpu_logits)
    bias = cpu_bias.to(device)
    weights = torch.empty(4, 6 + int(shared), device=device)
    ids = torch.empty_like(weights, dtype=torch.int32)
    expected_w, expected_ids = _nonsoftmax_oracle(
        cpu_logits, cpu_bias, "sqrtsoftplus" if entry == "fused_sqrtsoftplus" else "sigmoid",
        grouped, renormalize, shared,
    )
    for _ in range(3):
        _nonsoftmax_gate(entry, logits, bias, weights, ids, renormalize, shared)
        assert torch.equal(ids.cpu(), expected_ids)
        torch.testing.assert_close(weights.cpu(), expected_w, rtol=1e-6, atol=1e-7)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("entry", [
    "topk_sigmoid", "fused_sigmoid", "fused_grouped_sigmoid", "fused_sqrtsoftplus",
])
def test_nonsoftmax_ties_follow_changed_graph_inputs(dtype, entry):
    grouped = entry == "fused_grouped_sigmoid"
    cpu_logits, cpu_bias = _nonsoftmax_inputs(dtype, grouped)
    logits, bias = cpu_logits.to("cuda"), cpu_bias.to("cuda")
    weights = torch.empty(4, 7, device="cuda")
    ids = torch.empty_like(weights, dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _nonsoftmax_gate(entry, logits, bias, weights, ids, True, True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _nonsoftmax_gate(entry, logits, bias, weights, ids, True, True)
    for shift in [0, 11, 29, 0]:
        changed = cpu_logits.roll(shift, dims=1)
        logits.copy_(changed)
        graph.replay()
        expected_w, expected_ids = _nonsoftmax_oracle(
            changed, cpu_bias,
            "sqrtsoftplus" if entry == "fused_sqrtsoftplus" else "sigmoid",
            grouped, True, True,
        )
        assert torch.equal(ids.cpu(), expected_ids)
        torch.testing.assert_close(weights.cpu(), expected_w, rtol=1e-6, atol=1e-7)
