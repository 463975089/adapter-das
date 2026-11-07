"""Original RoPE JIT with packed views and an independent rounding envelope."""

import pytest
import torch


def _batch(phase, tokens, dtype):
    generator = torch.Generator().manual_seed(1241 + phase)
    packed = torch.randn(tokens, 16 + 4096 + 16, generator=generator).to(dtype)
    positions = ((torch.arange(tokens) * 7 + phase * 13) % 4096).to(
        torch.int32 if dtype == torch.float16 else torch.int64)
    positions[0] = (0, 2048, 4095)[phase]
    inverse = (1000000 + phase * 1000) ** (-torch.arange(0, 128, 2).double() / 128)
    angles = torch.arange(4096).double()[:, None] * inverse
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).float()
    # Equal coefficients and inputs cancel in FP64. An FP32 multiply followed
    # by FMA may leave a small nonzero result, many BF16 steps away from zero.
    cache[positions[0]] = torch.tensor(0.5).sqrt()
    packed[0, 16:16 + 128] = 1.0078125
    return packed, positions, cache


def _check_rotation(actual, inputs, cache_rows, is_neox):
    values = inputs.double()
    cosine, sine = cache_rows.double().chunk(2, dim=-1)
    cosine, sine = cosine[:, None], sine[:, None]
    if is_neox:
        first, second = values.chunk(2, dim=-1)
    else:
        first, second = values[..., 0::2], values[..., 1::2]

    def join(left, right):
        if is_neox:
            return torch.cat((left, right), dim=-1)
        return torch.stack((left, right), dim=-1).flatten(-2)

    exact = join(first * cosine - second * sine,
                 second * cosine + first * sine)
    # Roundoff scales with the products, not their possibly cancelling sum.
    # Eight FP32 unit roundoffs cover separate products/add and either FMA
    # contraction order; round this interval only at the output storage edge.
    bound = join(first.abs() * cosine.abs() + second.abs() * sine.abs(),
                 second.abs() * cosine.abs() + first.abs() * sine.abs()) * (8 * 2**-24)
    lower = (exact - bound).to(actual.dtype).double()
    upper = (exact + bound).to(actual.dtype).double()
    observed = actual.double()
    assert torch.isfinite(observed).all()
    assert ((observed >= lower) & (observed <= upper)).all()
    relative_l2 = (observed - exact).norm() / exact.norm()
    assert relative_l2 <= (0.003 if actual.dtype == torch.bfloat16 else 0.0005)


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="Requires a real HIP device",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize("tokens", [2, 47])
def test_original_rope_packed_fresh_input_graph(dtype, is_neox, tokens):
    from sglang.kernels.ops.attention.rope import apply_rope_with_cos_sin_cache_inplace

    batches = [_batch(phase, tokens, dtype) for phase in range(3)]
    packed, positions, cache = [tensor.cuda() for tensor in batches[0]]
    q_end, k_end = 16 + 2048, 16 + 3072
    q = packed[:, 16:q_end].view(tokens, 16, 128)
    k = packed[:, q_end:k_end].view(tokens, 8, 128)
    assert q.stride(0) > 16 * 128 and k.stride(0) > 8 * 128

    def execute():
        apply_rope_with_cos_sin_cache_inplace(
            q=q, k=k, cos_sin_cache=cache, positions=positions, is_neox=is_neox)

    execute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    torch.cuda.synchronize()

    for inputs, indices, coefficients in batches:
        selected = coefficients[indices.long()]
        eager = None
        for mode in ("eager", "graph"):
            packed.copy_(inputs)
            positions.copy_(indices)
            cache.copy_(coefficients)
            if mode == "eager":
                execute()
            else:
                graph.replay()
            observed = packed.cpu()
            _check_rotation(
                observed[:, 16:q_end].reshape(tokens, 16, 128),
                inputs[:, 16:q_end].reshape(tokens, 16, 128), selected, is_neox)
            _check_rotation(
                observed[:, q_end:k_end].reshape(tokens, 8, 128),
                inputs[:, q_end:k_end].reshape(tokens, 8, 128), selected, is_neox)
            assert torch.equal(observed[:, :16], inputs[:, :16])
            assert torch.equal(observed[:, k_end:], inputs[:, k_end:])
            assert torch.equal(positions.cpu(), indices)
            assert torch.equal(cache.cpu(), coefficients)
            if eager is None:
                eager = observed.clone()
            else:
                assert torch.equal(observed, eager)
