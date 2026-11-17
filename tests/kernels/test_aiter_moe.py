from types import SimpleNamespace

import pytest
import torch

from hcu_adapter import aiter_moe

# Which shapes the DTK aiter registry will actually serve with a validated
# asm/moe_c solution drifts with the image's tune tables (2026-11-16 image:
# E=8/K=256 and E=128/K=5120/inner=288 dropped out, E=128/K=4096/inner=192
# and the moe_c shape E=192/K=4096/inner=128 are in). Numeric coverage runs
# wherever the stack engages; the canary below keeps the bridge from
# silently going dead on every shape.
SHAPES = [
    (384, 5120, 288, 3, 6),
    (128, 4096, 192, 3, 8),
    (192, 4096, 128, 3, 8),
]


def _shape_engages(experts, width, inner, rows, topk):
    """Mirror the bridge's engagement gate: a validated config whose
    solution is asm/moe_c (hcu_adapter/aiter_moe.py returns None on
    anything else, so the numeric contract below would never run)."""
    try:
        from aiter.moe import MoeQuantType, get_aiter_moe_config
    except ImportError:
        return False
    quant = getattr(MoeQuantType, "fp8_w8a8_channel", None)
    if quant is None:
        quant = getattr(MoeQuantType, "FP8_W8A8", None)
    if quant is None:
        return False
    try:
        status, cfg = get_aiter_moe_config(
            M=rows, E=experts, N1=2 * inner, N2=width, K=width, top_k=topk,
            block_size=0, dtype=torch.bfloat16, quant_type=quant,
            activation="silu",
        )
    except Exception:
        return False
    return bool(status) and getattr(cfg, "solution_type", None) in ("asm", "moe_c")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")
def test_at_least_one_shape_engages_on_this_stack():
    assert any(_shape_engages(*s) for s in SHAPES), (
        "no aiter MoE shape engages on this stack -- the aiter bridge is "
        "dead here and this file's numeric coverage is silently gone"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")
@pytest.mark.parametrize("experts,width,inner,rows,topk", SHAPES)
def test_channel_fp8_swiglu_clamp_eager_and_graph(monkeypatch, experts, width, inner, rows, topk):
    if not _shape_engages(experts, width, inner, rows, topk):
        pytest.skip("DTK aiter has no validated asm/moe_c config for this shape")
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
