# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Community Dense MLP composition at VL2's actual prefill/decode shapes."""

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU and community SGLang"
)


@pytest.fixture
def replicated_allocation_context():
    from sglang.srt.runtime_context import get_parallel

    # Replicated linears read the allocation context even without collectives.
    with get_parallel().override(tp_group=None):
        yield


def _reference(x, gate_up, down, dtype):
    # CPU FP32 accumulation with an explicit store at each dtype boundary.
    packed = F.linear(x.float(), gate_up).to(dtype).float()
    gate, up = packed.chunk(2, dim=-1)
    activation = (F.silu(gate) * up).to(dtype).float()
    return F.linear(activation, down).to(dtype).float()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 437])
def test_community_dense_mlp_staged_reference_and_fresh_graph(
    dtype, tokens, replicated_allocation_context
):
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP

    torch.manual_seed(1771)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device("cuda"):
            # The public replicated mode needs no distributed process group.
            module = DeepseekV2MLP(
                hidden_size=2560,
                intermediate_size=12288,
                hidden_act="silu",
                parallel_group="replicated",
            ).eval()
    finally:
        torch.set_default_dtype(previous_dtype)

    with torch.no_grad():
        module.gate_up_proj.weight.normal_(std=2560**-0.5)
        module.down_proj.weight.normal_(std=12288**-0.5)
    gate_up = module.gate_up_proj.weight.detach().cpu().float()
    down = module.down_proj.weight.detach().cpu().float()
    assert gate_up.shape == (24576, 2560) and down.shape == (2560, 12288)
    assert module.gate_up_proj.weight.dtype == module.down_proj.weight.dtype == dtype
    storage = torch.empty((tokens * 2560 + 32,), device="cuda", dtype=dtype)
    inputs = storage[16:-16].view(tokens, 2560)

    def fill(phase):
        generator = torch.Generator().manual_seed(1781 + phase)
        values = torch.randn(tokens * 2560 + 32, generator=generator).to(dtype)
        storage.copy_(values)
        return values

    with torch.inference_mode():
        fill(0)
        module(inputs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            module(inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_output = module(inputs)

        initial = None
        for phase in range(3):
            values = fill(phase)
            original = values[16:-16].view(tokens, 2560)
            expected = _reference(original, gate_up, down, dtype)
            eager = module(inputs).cpu().float()
            graph.replay()
            observed = captured_output.cpu().float()
            assert torch.equal(observed, eager), "graph used different or stale data"
            assert torch.equal(storage.cpu(), values), "MLP mutated its input storage"
            assert observed.shape == (tokens, 2560) and torch.isfinite(observed).all()
            relative_l2 = (observed - expected).norm() / expected.norm()
            cosine = F.cosine_similarity(
                observed.flatten(), expected.flatten(), dim=0
            )
            scaled_max = (observed - expected).abs().max() / expected.abs().max()
            if dtype == torch.bfloat16:
                assert relative_l2 < 1e-2 and cosine > 0.9999
                assert scaled_max < 2e-2
            else:
                assert relative_l2 < 2e-3 and cosine > 0.999998
                assert scaled_max < 3e-3
            if initial is None:
                initial = observed.clone()
            else:
                assert not torch.equal(observed, initial), "output ignored new inputs"
