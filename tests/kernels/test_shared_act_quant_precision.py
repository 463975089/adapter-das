# SPDX-License-Identifier: Apache-2.0
"""Byte comparisons against the actual native activation and FP8 providers."""
import pytest
import torch

from hcu_adapter.features.dsv41.shared_act_quant import silu_clamp_bf16_quant

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU required")


def reference(x, limit, lightop):
    from vllm import _custom_ops as ops

    y = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=x.dtype)
    torch.ops._C.silu_and_mul_with_clamp(y, x, limit, 1.0, 0.0)
    if lightop:
        from lightop.quant import per_token_quant_fp8

        q, scale = per_token_quant_fp8(y)
        return q, scale.reshape(x.shape[0], 1)
    return ops.scaled_fp8_quant(y, use_per_token_if_dynamic=True)


def assert_bytes(actual, expected):
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.view(torch.uint8), e.view(torch.uint8),
                                   rtol=0, atol=0)


@pytest.mark.parametrize("n", [7, 16, 120, 128, 2304])
@pytest.mark.parametrize("lightop", [False, True])
@pytest.mark.parametrize("limit", [10.0, 1.00390625, 0.1, -1.00390625])
@pytest.mark.parametrize("gate,up", [(2.0, 1.0), (1.0, float("nan")),
                                    (float("-inf"), 1.0),
                                    (float("nan"), float("inf"))])
def test_clamp_special_values_and_bf16_boundaries(n, lightop, limit, gate, up):
    x = torch.empty((2, 2 * n), device="cuda", dtype=torch.bfloat16)
    x[:, :n] = gate
    x[:, n:] = up
    if lightop and n % 16:
        with pytest.raises(ValueError, match="vector-aligned"):
            silu_clamp_bf16_quant(x, limit, lightop=True)
        return
    # A constant row exposes scale changes; the second mixes special values
    # with finite values and signed zero instead of relying only on saturation.
    x[1, 0] = 0.0625
    x[1, n] = -0.0
    before = x.clone()
    assert_bytes(silu_clamp_bf16_quant(x, limit, lightop=lightop),
                 reference(x, limit, lightop))
    assert torch.equal(x.view(torch.uint8), before.view(torch.uint8))


@pytest.mark.parametrize("n", [128, 2304])
@pytest.mark.parametrize("lightop", [False, True])
def test_non_bf16_limit_and_nan_inputs_in_changing_graph(n, lightop):
    limit = 1.00390625
    x = torch.ones((7, 2 * n), device="cuda", dtype=torch.bfloat16)
    silu_clamp_bf16_quant(x, limit, lightop=lightop)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = silu_clamp_bf16_quant(x, limit, lightop=lightop)
    for gate, up in [(2.0, 1.0), (1.0, float("nan")), (0.0625, 0.0009765625)]:
        x[:, :n] = gate
        x[:, n:] = up
        graph.replay()
        assert_bytes(actual, reference(x, limit, lightop))
