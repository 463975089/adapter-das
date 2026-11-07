"""CT native-dtype token INT8: exact codes/scales and changed-input graphs.

Run: pytest tests/kernels/test_w4a8_int8_native.py -q (HCU node).
Reference is independently checked against official CT on CPU.
"""

import pytest
import torch

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires HCU/CUDA device",
)


@requires_gpu
def test_expert_batching_reduces_peak_memory(monkeypatch):
    from hcu_adapter.kernels import w4a8_int8_native_triton as native

    experts, capacity, hidden, intermediate = 16, 128, 256, 128
    rng = torch.Generator(device="cuda").manual_seed(113)
    value = torch.randn(experts, capacity, hidden, generator=rng,
                        device="cuda", dtype=torch.bfloat16)
    gate = torch.randint(0, 256, (experts, 2 * intermediate, hidden // 2),
                         generator=rng, device="cuda", dtype=torch.uint8)
    down = torch.randint(0, 256, (experts, hidden, intermediate // 2),
                         generator=rng, device="cuda", dtype=torch.uint8)
    gs = torch.full((experts, 2 * intermediate, 1), .01,
                    device="cuda", dtype=torch.bfloat16)
    ds = torch.full((experts, hidden, 1), .01,
                    device="cuda", dtype=torch.bfloat16)
    counts = torch.full((experts,), capacity, device="cuda", dtype=torch.int32)

    def run():
        return native.w4a8_int8_native_masked_experts(
            value, gate, down, gs, ds, counts,
        )

    def measured(budget):
        monkeypatch.setattr(native, "_EXPERT_SCRATCH_BYTES", budget)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        result = run()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        cpu = result.cpu()
        del result
        return cpu, peak

    # Compile both shapes before measuring live tensor memory.
    for budget in (1, 2**60):
        monkeypatch.setattr(native, "_EXPERT_SCRATCH_BYTES", budget)
        run()
    unbatched, full_peak = measured(2**60)
    batched, small_peak = measured(1)
    assert torch.equal(batched, unbatched)
    assert small_peak < full_peak * .6, (small_peak, full_peak)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("capacity,hidden", [(17, 160), (129, 513)])
def test_native_token_quantization_fresh_graph(dtype, capacity, hidden):
    from hcu_adapter.kernels.w4a8_int8_native_triton import token_int8_native_masked
    from tools.ct_w4a8_reference import token_int8_native

    counts = torch.tensor([capacity, 0, 3], device="cuda", dtype=torch.int32)
    # Strided input exercises the wrapper's contiguity conversion under capture.
    backing = torch.empty(3, capacity, hidden * 2, device="cuda", dtype=dtype)
    value = backing[..., ::2]
    rng = torch.Generator(device="cuda").manual_seed(826)

    def change():
        value.normal_(generator=rng)
        value[0, 0].zero_()
        value[2].mul_(.03125)
        value[0, 1, :8] = torch.tensor(
            [127.5, -127.5, .5, -.5, 1.5, -1.5, 2.5, -2.5],
            device="cuda", dtype=dtype,
        )
        valid = torch.arange(capacity, device="cuda")[None, :] < counts[:, None]
        value.masked_fill_(~valid[..., None], float("nan"))

    def run():
        return token_int8_native_masked(value, counts)

    def check(result):
        q, s = (t.cpu() for t in result)
        valid = torch.arange(capacity)[None, :] < counts.cpu()[:, None]
        source = value.cpu().masked_fill(~valid[..., None], 0)
        expected_q, expected_s, _ = token_int8_native(source)
        assert q.dtype == torch.int8 and s.dtype == dtype
        assert torch.equal(q, expected_q)
        assert torch.equal(s[valid], expected_s[valid])
        assert torch.equal(s[~valid], torch.ones_like(s[~valid]))

    change()
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
    value.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check(result)
    counts.copy_(torch.tensor([0, capacity, 1], device="cuda", dtype=torch.int32))
    change()
    graph.replay()
    torch.cuda.synchronize()
    check(result)
    counts.copy_(torch.tensor([capacity, 0, 3], device="cuda", dtype=torch.int32))
    change()
    graph.replay()
    torch.cuda.synchronize()
    check(result)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_native_token_quantization_empty_capacity(dtype):
    from hcu_adapter.kernels.w4a8_int8_native_triton import token_int8_native_masked

    value = torch.empty(3, 0, 160, dtype=dtype, device="cuda")
    counts = torch.zeros(3, dtype=torch.int32, device="cuda")
    q, s = token_int8_native_masked(value, counts)
    assert q.shape == value.shape and q.dtype == torch.int8
    assert s.shape == (3, 0, 1) and s.dtype == dtype


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_native_quantization_integer_gemm_fresh_graph(dtype):
    from hcu_adapter.kernels.w4a8_int8_native_triton import token_int8_native_masked
    from hcu_adapter.kernels.w4a8_int8_triton import w4a8_int8_masked_gemm
    from tools.ct_w4a8_reference import token_int8_native

    experts, capacity, hidden, outputs = 3, 17, 160, 80
    rng = torch.Generator().manual_seed(827)
    unsigned = torch.randint(0, 16, (experts, outputs, hidden), generator=rng)
    signed = unsigned - 8
    packed = (unsigned[..., ::2] | (unsigned[..., 1::2] << 4)).byte().cuda()
    ws = (torch.rand(experts, outputs, 1, generator=rng) * .125 + .01).bfloat16().cuda()
    value = torch.randn(experts, capacity, hidden, generator=rng).to(dtype).cuda()
    counts = torch.tensor([capacity, 0, 3], device="cuda", dtype=torch.int32)

    def run():
        q, s = token_int8_native_masked(value, counts)
        return w4a8_int8_masked_gemm(q, s.float(), packed, ws, counts)

    def check(result):
        valid = torch.arange(capacity)[None, :] < counts.cpu()[:, None]
        source = value.cpu().masked_fill(~valid[..., None], 0)
        q, s, _ = token_int8_native(source)
        # FP64 integer products/sums are exact for these dimensions.
        dot = torch.bmm(q.double(), signed.double().transpose(1, 2)).float()
        expected = (dot * s.float()) * ws.cpu().float().transpose(1, 2)
        actual = result.cpu()
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual[~valid]) == 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

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


    value.mul_(.25)
    counts.copy_(torch.tensor([0, capacity, 1], device="cuda", dtype=torch.int32))
    ws.mul_(.5)
    graph.replay()
    torch.cuda.synchronize()
    check(result)
    counts.zero_()
    value.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check(result)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("limit", [None, .75])
@pytest.mark.parametrize("scratch_bytes", [1, 64 * 1024 * 1024])
def test_native_experts_fresh_graph(dtype, limit, scratch_bytes, monkeypatch):
    from hcu_adapter.kernels import w4a8_int8_native_triton as native
    from hcu_adapter.kernels.w4a8_int8_native_triton import w4a8_int8_native_masked_experts
    from tools.ct_w4a8_reference import int32_native_expert

    monkeypatch.setattr(native, "_EXPERT_SCRATCH_BYTES", scratch_bytes)
    experts, capacity, hidden, intermediate = 3, 17, 160, 80
    rng = torch.Generator().manual_seed(842)

    def weight(outputs, columns):
        unsigned = torch.randint(0, 16, (experts, outputs, columns), generator=rng)
        byte = (unsigned[..., ::2] | (unsigned[..., 1::2] << 4)).byte()
        words = sum(unsigned[..., i::8].long() << (4 * i) for i in range(8)).int()
        scale = (torch.rand(experts, outputs, 1, generator=rng) * .125 + .01).bfloat16()
        return byte.cuda(), scale.cuda(), words

    gw, gs, gwords = weight(2 * intermediate, hidden)
    dw, ds, dwords = weight(hidden, intermediate)
    counts = torch.tensor([capacity, 0, 3], device="cuda", dtype=torch.int32)
    value = torch.empty(experts, capacity, hidden, device="cuda", dtype=dtype)
    device_rng = torch.Generator(device="cuda").manual_seed(843)

    def change():
        value.normal_(generator=device_rng)
        value[:, 0].zero_()
        value[2].mul_(.03125)
        valid = torch.arange(capacity, device="cuda")[None, :] < counts[:, None]
        value.masked_fill_(~valid[..., None], float("nan"))

    def run():
        return w4a8_int8_native_masked_experts(
            value, gw, dw, gs, ds, counts, swiglu_limit=limit,
        )

    def check(result):
        expected = torch.zeros_like(value, device="cpu")
        for e, count in enumerate(counts.cpu().tolist()):
            if count:
                expected[e, :count] = int32_native_expert(
                    value[e:e + 1, :count].cpu(),
                    (gwords[e, :intermediate], gs[e, :intermediate].cpu()),
                    (gwords[e, intermediate:], gs[e, intermediate:].cpu()),
                    (dwords[e], ds[e].cpu()), swiglu_limit=limit,
                )[0]
        actual = result.cpu()
        assert actual.dtype == dtype and torch.isfinite(actual).all()
        valid = torch.arange(capacity)[None, :] < counts.cpu()[:, None]
        assert torch.count_nonzero(actual[~valid]) == 0
        delta = actual.float() - expected.float()
        norm = torch.linalg.vector_norm(expected.float())
        if norm:
            assert torch.linalg.vector_norm(delta) / norm < 2e-5
            assert torch.nn.functional.cosine_similarity(
                actual.float().flatten(), expected.float().flatten(), dim=0,
            ) > .99999
        else:
            assert torch.count_nonzero(actual) == 0

    change()
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
    value.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check(result)
    counts.copy_(torch.tensor([0, capacity, 1], device="cuda", dtype=torch.int32))
    change()
    gs.mul_(.5)
    ds.mul_(.75)
    graph.replay()
    torch.cuda.synchronize()
    check(result)
    counts.copy_(torch.tensor([capacity, 0, 3], device="cuda", dtype=torch.int32))
    change()
    graph.replay()
    torch.cuda.synchronize()
    check(result)
