from types import SimpleNamespace

import pytest
import torch

from hcu_adapter import aiter_moe


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")
@pytest.mark.parametrize("experts,width,inner,rows,topk", [
    (8, 256, 128, 3, 2), (384, 5120, 288, 3, 6), (128, 5120, 288, 3, 3),
])
def test_channel_fp8_swiglu_clamp_eager_and_graph(monkeypatch, experts, width, inner, rows, topk):
    monkeypatch.setenv("HCU_FUSED_MOE_IMPL", "aiter")
    x = torch.empty((rows, width), dtype=torch.bfloat16, device="cuda")
    w1 = torch.ones((experts, 2 * inner, width), device="cuda").to(torch.float8_e4m3fn)
    w2 = torch.ones((experts, width, inner), device="cuda").to(torch.float8_e4m3fn)
    s1 = torch.full((experts, 2 * inner, 1), 1 / 32, device="cuda")
    s1[:, inner:] = 1 / 16
    s2 = torch.full((experts, width, 1), 1 / inner, device="cuda")
    weights = torch.full((rows, topk), 1 / topk, device="cuda")
    ids = torch.arange(rows * topk, dtype=torch.int32, device="cuda").reshape(rows, topk) % experts

    def run(limit):
        config = SimpleNamespace(
            inplace=False, no_combine=False, activation="silu",
            routed_scaling_factor=1.0, gemm1_alpha=None,
            gemm1_clamp_limit=None, swiglu_limit=limit,
        )
        result = aiter_moe.try_aiter_fused_moe(
            hidden_states=x, w1=w1, w2=w2, topk_output=(weights, ids, None),
            moe_runner_config=config, use_fp8_w8a8=True, use_int8_w8a8=False,
            w1_scale=s1, w2_scale=s2, a1_scale=None, block_shape=None,
        )
        assert result is not None, "AITER must execute for numeric coverage"
        return result

    def reference(limit):
        total = x.float().sum(dim=-1, keepdim=True)
        gate, up = total / 32, total / 16
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        return (torch.nn.functional.silu(gate) * up).expand_as(x)

    def fill(values):
        x.copy_((torch.tensor(values, dtype=torch.float32, device=x.device) * (256 / width))[:, None].expand_as(x))

    fill([4.0, 0.25, -0.25])
    unclamped = run(None)
    clamped = run(10.0)
    torch.testing.assert_close(unclamped.float(), reference(None), rtol=0.02, atol=0.02)
    torch.testing.assert_close(clamped.float(), reference(10.0), rtol=0.02, atol=0.02)
    assert (unclamped[0].float() - clamped[0].float()).abs().min().item() > 100

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run(10.0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = run(10.0)
    for values in ([0.25, 4.0, 0.5], [6.0, -0.25, 0.25], [0.5, 0.25, 8.0]):
        fill(values)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output.float(), reference(10.0), rtol=0.02, atol=0.02)
