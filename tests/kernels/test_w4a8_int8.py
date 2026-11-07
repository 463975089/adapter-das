"""Candidate CT channel-INT4/token-INT8 kernels: CPU oracle and fresh graph.

Reference: independent CPU FP32 activation QDQ and unpacked channel-INT4 GEMM.
Require exact quantized codes/scales, relative L2 < 2e-5 and cosine > .99999.
Run: pytest tests/kernels/test_w4a8_int8.py -q (HCU node).
"""

import pytest
import torch

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires HCU/CUDA device"
)


def _weights(experts, outputs, hidden):
    rng = torch.Generator().manual_seed(721)
    unsigned = torch.randint(0, 16, (experts, outputs, hidden), generator=rng)
    byte = (unsigned[..., ::2] | (unsigned[..., 1::2] << 4)).to(torch.uint8)
    words = sum(unsigned[..., shift::8].long() << (4 * shift) for shift in range(8))
    scales = torch.rand(experts, outputs, 1, generator=rng) * .125 + .01
    return byte.cuda(), scales.cuda(), words.to(torch.int32)


@requires_gpu
@pytest.mark.parametrize("capacity,hidden,outputs", [(17, 160, 80), (1024, 512, 128)])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.bfloat16])
def test_ct_w4a8_masked_numeric_and_fresh_graph(
    capacity, hidden, outputs, input_dtype, scale_dtype,
):
    from hcu_adapter.kernels.w4a8_int8_triton import (
        token_int8_masked, w4a8_int8_masked_gemm,
    )
    from tools.ct_w4a8_reference import channel_int4, token_int8

    experts = 4
    byte, weight_scales, words = _weights(experts, outputs, hidden)
    weight_scales = weight_scales.to(scale_dtype)
    counts = torch.tensor([capacity, 0, 1, min(7, capacity)], device="cuda",
                          dtype=torch.int32)
    value = torch.empty(experts, capacity, hidden, dtype=input_dtype, device="cuda")
    rng = torch.Generator(device="cuda").manual_seed(722)

    def change_input():
        value.normal_(generator=rng)
        value[0, 0].zero_()
        value[2].mul_(.03125)
        active = torch.arange(capacity, device="cuda")[None, :] < counts[:, None]
        value.masked_fill_(~active[..., None], float("nan"))

    def run():
        codes, scales = token_int8_masked(value, counts)
        output = w4a8_int8_masked_gemm(codes, scales, byte, weight_scales, counts)
        return codes, scales, output

    def check(result):
        codes, scales, output = result
        cpu_counts = counts.cpu()
        valid = torch.arange(capacity)[None, :] < cpu_counts[:, None]
        source = value.cpu().float().masked_fill(~valid[..., None], 0)
        q, s, qdq = token_int8(source)
        assert torch.equal(codes.cpu(), q)
        torch.testing.assert_close(scales.cpu()[valid], s[valid], rtol=0, atol=0)
        assert torch.equal(scales.cpu()[~valid], torch.ones_like(s[~valid]))
        decoded = torch.stack([
            channel_int4(words[e], weight_scales[e].cpu(), hidden)
            for e in range(experts)
        ])
        expected = torch.einsum("emk,enk->emn", qdq, decoded)
        expected.masked_fill_(~valid[..., None], 0)
        actual = output.cpu()
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual[~valid]) == 0
        torch.testing.assert_close(actual, expected, rtol=1e-4,
                                   atol=1e-5 * max(1., expected.abs().max().item()))
        denominator = torch.linalg.vector_norm(expected)
        if denominator == 0:
            assert torch.count_nonzero(actual) == 0
            return
        assert (torch.linalg.vector_norm(actual - expected) / denominator) < 2e-5
        assert torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ) > .99999

    change_input()
    check(run())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = run()
    graph.replay()
    torch.cuda.synchronize()
    check(result)
    counts.zero_()
    change_input()
    graph.replay()
    torch.cuda.synchronize()
    check(result)
    counts.copy_(torch.tensor([capacity, 0, 1, min(7, capacity)],
                              device="cuda", dtype=torch.int32))
    change_input()
    graph.replay()
    torch.cuda.synchronize()
    check(result)
    counts.copy_(torch.tensor([0, capacity, min(3, capacity), 1],
                              device="cuda", dtype=torch.int32))
    change_input()
    weight_scales.mul_(.75)
    graph.replay()
    torch.cuda.synchronize()
    check(result)


@requires_gpu
def test_ct_token_int8_halfway_rounding():
    from hcu_adapter.kernels.w4a8_int8_triton import token_int8_masked
    from tools.ct_w4a8_reference import token_int8

    row = torch.tensor([127.5, -127.5, .5, -.5, 1.5, -1.5, 2.5, -2.5])
    value = row[None, None, :].repeat(1, 4, 1)
    value *= torch.tensor([1., 2., .125, 0.])[None, :, None]
    expected_q, expected_s, _ = token_int8(value)
    actual_q, actual_s = token_int8_masked(
        value.cuda(), torch.tensor([4], device="cuda", dtype=torch.int32)
    )
    assert torch.equal(actual_q.cpu(), expected_q)
    assert torch.equal(actual_s.cpu(), expected_s)
