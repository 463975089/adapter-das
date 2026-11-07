"""Original fused QK norm: independent oracle, packed views and graph replay."""

import pytest
import torch


def _batch(phase, tokens, q_heads, k_heads, head_dim, dtype):
    generator = torch.Generator().manual_seed(972 + phase)
    width = 16 + (q_heads + 2 * k_heads) * head_dim + 16
    packed = torch.randn(tokens, width, generator=generator).to(dtype)
    q_weight = torch.randn(head_dim, generator=generator).to(dtype)
    k_weight = torch.randn(head_dim, generator=generator).to(dtype)
    q = packed[:, 16:16 + q_heads * head_dim].view(tokens, q_heads, head_dim)
    k_start = 16 + q_heads * head_dim
    k = packed[:, k_start:k_start + k_heads * head_dim].view(
        tokens, k_heads, head_dim)
    # Exercise epsilon-dominated rows, exact zeros and ordinary signed weights.
    q[0].mul_(0.0001 * (phase + 1))
    k[0].zero_()

    def reference(x, weight):
        values = x.double()
        return (values * torch.rsqrt(
            values.square().mean(-1, keepdim=True) + 1e-6
        ) * weight.double()).to(dtype)

    return packed, q_weight, k_weight, reference(q, q_weight), reference(k, k_weight)


def _check_output(actual, expected):
    assert torch.isfinite(actual).all()
    # Independent FP64 reduction can round across a storage midpoint. Bound
    # that difference to an adjacent representable value, rather than masking
    # layout or normalization errors with a large absolute tolerance.
    adjacent_low = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
    adjacent_high = torch.nextafter(expected, torch.full_like(expected, torch.inf))
    assert ((actual >= adjacent_low) & (actual <= adjacent_high)).all()
    relative_l2 = (actual.double() - expected.double()).norm() / expected.double().norm()
    assert relative_l2 <= (0.003 if actual.dtype == torch.bfloat16 else 0.0005)


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="Requires a real HIP device",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tokens,q_heads,k_heads,head_dim",
    [(2, 16, 8, 128), (47, 16, 8, 128), (3, 4, 2, 512)],
)
def test_fused_qknorm_packed_fresh_input_graph(
    dtype, tokens, q_heads, k_heads, head_dim,
):
    from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm

    batches = [
        _batch(phase, tokens, q_heads, k_heads, head_dim, dtype)
        for phase in range(3)
    ]
    packed = batches[0][0].cuda()
    q_weight = batches[0][1].cuda()
    k_weight = batches[0][2].cuda()
    q_end = 16 + q_heads * head_dim
    k_end = q_end + k_heads * head_dim
    q = packed[:, 16:q_end].view(tokens, q_heads, head_dim)
    k = packed[:, q_end:k_end].view(tokens, k_heads, head_dim)
    assert q.stride(0) > q_heads * head_dim
    assert k.stride(0) > k_heads * head_dim

    def execute():
        fused_inplace_qknorm(
            q, k, q_weight, k_weight, eps=1e-6, head_dim=head_dim)

    execute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    torch.cuda.synchronize()

    for inputs, qw, kw, q_ref, k_ref in batches:
        eager = None
        for mode in ("eager", "graph"):
            packed.copy_(inputs)
            q_weight.copy_(qw)
            k_weight.copy_(kw)
            if mode == "eager":
                execute()
            else:
                graph.replay()
            actual = packed.cpu()
            q_out = actual[:, 16:q_end].reshape(tokens, q_heads, head_dim)
            k_out = actual[:, q_end:k_end].reshape(tokens, k_heads, head_dim)
            _check_output(q_out, q_ref)
            _check_output(k_out, k_ref)
            # Prefix/suffix guards and V share the original packed allocation.
            assert torch.equal(actual[:, :16], inputs[:, :16])
            assert torch.equal(actual[:, k_end:], inputs[:, k_end:])
            assert torch.equal(q_weight.cpu(), qw)
            assert torch.equal(k_weight.cpu(), kw)
            if eager is None:
                eager = actual.clone()
            else:
                assert torch.equal(actual, eager)
