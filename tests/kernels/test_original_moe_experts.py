# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Community BF16 routed experts at VL2's real dimensions and token shapes."""

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU and community SGLang"
)


@pytest.fixture
def runtime_context():
    from sglang.test.test_utils import published_topology

    with published_topology():
        yield


def _reference(inputs, w1, w2, ids, weights):
    # Independent CPU arithmetic, including the weighted-down storage boundary.
    pieces = torch.empty(inputs.shape[0], 6, 2560, dtype=torch.float32)
    for expert in ids.unique().tolist():
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        packed = F.linear(inputs[rows].float(), w1[expert]).bfloat16().float()
        gate, up = packed.chunk(2, dim=-1)
        activation = (F.silu(gate) * up).bfloat16().float()
        down = F.linear(activation, w2[expert])
        pieces[rows, slots] = (
            down * weights[rows, slots, None]
        ).bfloat16().float()
    combined = torch.zeros(inputs.shape, dtype=torch.float32)
    for slot in range(6):
        combined += pieces[:, slot]
    # Scale applies after the FP32 sum, before the final BF16 store.
    return (combined * 2).bfloat16().float()


@pytest.mark.parametrize("tokens", [1, 437])
def test_vl2_routed_experts_staged_reference_and_fresh_graph(tokens, runtime_context):
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_experts_impl,
    )

    torch.manual_seed(1891)
    w1 = torch.empty(72, 3072, 2560, device="cuda", dtype=torch.bfloat16)
    w2 = torch.empty(72, 2560, 1536, device="cuda", dtype=torch.bfloat16)
    w1.normal_(std=2560**-0.5)
    w2.normal_(std=1536**-0.5)
    # An unselected expert must not contaminate outputs, including padded tiles.
    w1[-1].fill_(float("nan"))
    w2[-1].fill_(float("nan"))
    cpu_w1, cpu_w2 = w1.cpu().float(), w2.cpu().float()
    storage = torch.empty(tokens * 2560 + 32, device="cuda", dtype=torch.bfloat16)
    inputs = storage[16:-16].view(tokens, 2560)
    ids = torch.empty(tokens, 6, device="cuda", dtype=torch.int32)
    weights = torch.empty(tokens, 6, device="cuda", dtype=torch.float32)

    def fill(phase):
        generator = torch.Generator().manual_seed(1901 + phase)
        values = torch.randn(tokens * 2560 + 32, generator=generator).bfloat16()
        # Unique IDs within each row, with expert 71 deliberately empty.
        cpu_ids = torch.stack([
            torch.randperm(71, generator=generator)[:6] for _ in range(tokens)
        ]).to(torch.int32)
        cpu_weights = torch.rand(tokens, 6, generator=generator) + .1
        cpu_weights /= cpu_weights.sum(-1, keepdim=True)
        storage.copy_(values)
        ids.copy_(cpu_ids)
        weights.copy_(cpu_weights)
        return values, cpu_ids, cpu_weights

    def run():
        # In-place output avoids requiring distributed symmetric allocation.
        # The clone preserves the independently supplied input and sentinels.
        return fused_experts_impl(
            inputs.clone(), w1, w2, weights, ids,
            inplace=True, gate_up_interleaved=False, routed_scaling_factor=2.,
        )

    with torch.inference_mode():
        fill(0)
        run()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_output = run()

        initial = None
        for phase in range(3):
            values, cpu_ids, cpu_weights = fill(phase)
            expected = _reference(
                values[16:-16].view(tokens, 2560),
                cpu_w1, cpu_w2, cpu_ids.long(), cpu_weights,
            )
            eager = run().cpu().float()
            graph.replay()
            observed = captured_output.cpu().float()
            assert torch.equal(eager, observed), "graph used different or stale data"
            assert torch.equal(storage.cpu(), values), "experts mutated input storage"
            assert torch.equal(ids.cpu(), cpu_ids), "experts mutated routing IDs"
            assert torch.equal(weights.cpu(), cpu_weights), "experts mutated weights"
            assert observed.shape == expected.shape and torch.isfinite(observed).all()
            assert (observed - expected).norm() / expected.norm() < 1e-2
            assert F.cosine_similarity(
                observed.flatten(), expected.flatten(), dim=0
            ) > .9999
            assert (observed - expected).abs().max() / expected.abs().max() < 2e-2
            if initial is None:
                initial = observed.clone()
            else:
                assert not torch.equal(observed, initial), "output ignored new inputs"
