"""Original fused SiLU storage boundary and fresh-input graph contract."""

import pytest
import torch


def _batch(phase, tokens, dtype):
    generator = torch.Generator().manual_seed(1731 + phase)
    values = (torch.randn(tokens * 6144 + 32, generator=generator) * 3).to(dtype)
    inputs = values[16:-16].view(tokens, 6144)
    inputs[0, :5] = torch.tensor([-20, -5, 0, 5, 20], dtype=dtype)
    inputs[0, 3072:3077] = torch.tensor(
        [-7 + phase, 3, -2, 1 + phase, -4], dtype=dtype)
    return values


def _reference(inputs):
    gate, up = inputs.double().chunk(2, dim=-1)
    return gate * torch.sigmoid(gate) * up


def _within_reference(actual, expected):
    # FP32 sigmoid/exp and the two products precede one final dtype store.
    # Apply the arithmetic interval before that store, so premature BF16/FP16
    # intermediate rounding cannot hide inside a dtype-wide relative tolerance.
    bound = expected.abs() * (64 * 2**-24)
    lower = (expected - bound).to(actual.dtype).double()
    upper = (expected + bound).to(actual.dtype).double()
    observed = actual.double()
    return torch.isfinite(observed) & (observed >= lower) & (observed <= upper)


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="Requires a real HIP device",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [2, 47])
def test_original_silu_fused_store_fresh_input_graph(dtype, tokens):
    from sglang.kernels.ops.activation.activation import silu_and_mul

    batches = [_batch(phase, tokens, dtype) for phase in range(3)]
    storage = batches[0].cuda()
    inputs = storage[16:-16].view(tokens, 6144)
    output_storage = torch.full(
        (tokens * 3072 + 32,), 29, dtype=dtype, device="cuda")
    output = output_storage[16:-16].view(tokens, 3072)
    assert inputs.is_contiguous() and inputs.storage_offset() == 16
    assert output.is_contiguous() and output.storage_offset() == 16

    def execute():
        result = silu_and_mul(inputs, output)
        assert result is output

    execute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    torch.cuda.synchronize()

    for values in batches:
        original = values[16:-16].view(tokens, 6144)
        expected = _reference(original)
        # A deliberately premature activation store must be rejected by this
        # fixture, rather than merely comparing two implementations that agree.
        gate, up = original.double().chunk(2, dim=-1)
        premature = ((gate * torch.sigmoid(gate)).to(dtype).double() * up).to(dtype)
        assert not _within_reference(premature, expected).all()
        eager = None
        for mode in ("eager", "graph"):
            storage.copy_(values)
            output_storage.fill_(29)
            if mode == "eager":
                execute()
            else:
                graph.replay()
            observed = output.cpu()
            assert _within_reference(observed, expected).all()
            relative_l2 = (observed.double() - expected).norm() / expected.norm()
            assert relative_l2 <= (0.003 if dtype == torch.bfloat16 else 0.0005)
            assert torch.equal(storage.cpu(), values)
            guards = output_storage.cpu()
            assert (guards[:16] == 29).all() and (guards[-16:] == 29).all()
            if eager is None:
                eager = observed.clone()
            else:
                assert torch.equal(observed, eager)
