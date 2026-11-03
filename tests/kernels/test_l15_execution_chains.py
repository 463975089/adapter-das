"""Actual upstream execution chains; GPU-only, with independent FP32 oracles."""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def runtime():
    if not torch.cuda.is_available():
        pytest.skip("requires one HCU")
    from sglang.test.test_utils import published_topology
    with published_topology():
        yield


def check(actual, expected, *, atol=0.02):
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.05, atol=atol)
    cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    assert cosine.item() >= 0.9999


def replay_three(run, inputs, reference):
    """Capture once, mutate the same storage, and check three distinct waves."""
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    previous = reference()
    for wave in range(1, 4):
        for tensor in inputs:
            tensor.copy_(torch.randn_like(tensor) * (0.2 + wave * 0.1))
        expected = reference()
        assert not torch.allclose(previous.float(), expected.float(), atol=0.02, rtol=0.05)
        graph.replay()
        torch.cuda.synchronize()
        check(output, expected)
        previous = expected.clone()


@pytest.mark.parametrize("tokens,topk", [(1, 1), (7, 2), (33, 2)])
def test_triton_moe_expert_chain(runtime, tokens, topk):
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts

    torch.manual_seed(430 + tokens)
    experts, hidden, intermediate = 4, 128, 128
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.3
    logits = torch.randn(tokens, experts, device="cuda", dtype=torch.float32)
    w1 = torch.randn(experts, 2 * intermediate, hidden, device="cuda",
                     dtype=x.dtype) * 0.1
    w2 = torch.randn(experts, hidden, intermediate, device="cuda",
                     dtype=x.dtype) * 0.1

    def run():
        routing = select_experts(x, logits, TopKConfig(top_k=topk, renormalize=True))
        # The real Triton up/activation/down/weighted-combine path. In-place
        # output avoids requiring a distributed symmetric allocator in this
        # single-device component test; no computation is replaced.
        return fused_experts_impl(
            x.clone(), w1, w2, routing.topk_weights, routing.topk_ids,
            inplace=True, gate_up_interleaved=False,
        )

    def reference():
        weights, ids = torch.topk(logits.float().softmax(-1), topk, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        # Evaluate every expert independently, then gather routed outputs.
        outputs = []
        for expert in range(experts):
            gate, up = F.linear(x.float(), w1[expert].float()).chunk(2, dim=-1)
            outputs.append(F.linear(F.silu(gate) * up, w2[expert].float()))
        all_outputs = torch.stack(outputs, dim=1)
        selected = all_outputs.gather(1, ids[..., None].expand(-1, -1, hidden))
        return (selected * weights[..., None]).sum(1)

    check(run(), reference())
    replay_three(run, [x, logits], reference)


@pytest.mark.parametrize("kind", ["int8", "fp8"])
@pytest.mark.parametrize("tokens", [1, 17, 33])
def test_quantized_linear_chain(runtime, monkeypatch, kind, tokens):
    from compressed_tensors.quantization import QuantizationStrategy
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
        CompressedTensorsW8A8Int8,
    )
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

    torch.manual_seed(851 + tokens)
    hidden, output = 128, 256
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(output, device="cuda", dtype=x.dtype) * 0.1
    scale = torch.linspace(0.003, 0.02, output, device="cuda").view(-1, 1)
    layer = torch.nn.Module()
    dtype = torch.int8 if kind == "int8" else torch.float8_e4m3fn
    weight = torch.randint(-20, 21, (output, hidden), device="cuda").to(dtype)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    if kind == "int8":
        scheme = CompressedTensorsW8A8Int8(QuantizationStrategy.CHANNEL, False, True)
        scheme.process_weights_after_loading(layer)

        def run():
            return scheme.apply_weights(layer, x, bias)
    else:
        # Keep the real platform dispatch: claiming CUTLASS support on HIP
        # would exercise an unreachable NVIDIA branch.
        monkeypatch.setenv("SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE", "0")

        def run():
            return apply_fp8_linear(
                x, weight.t(), scale, bias=bias,
                pad_output=False, use_per_token_if_dynamic=True,
            )

    def reference():
        maximum = 127 if kind == "int8" else torch.finfo(dtype).max
        activation_scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-10) / maximum
        values = x.float() / activation_scale
        if kind == "int8":
            # libdevice.round uses half-away-from-zero (not torch's ties-even).
            values = values.sign() * torch.floor(values.abs() + 0.5)
        q = values.to(dtype).float()
        result = (q @ weight.float().t()) * activation_scale * scale.view(1, -1)
        return result.to(x.dtype) + bias

    check(run(), reference())
    replay_three(run, [x], reference)


@pytest.mark.parametrize("disable_update", [False, True])
def test_gdn_state_ring_and_graph(runtime, disable_update):
    from sglang.kernels.ops.attention.cutedsl_gdn_mtp_ring import gated_delta_rule_mtp

    torch.manual_seed(939)
    batch, steps, heads, width, value_width, slots = 2, 3, 2, 32, 64, 4
    q = torch.randn(batch, steps, heads, width, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, steps, heads, value_width, device="cuda", dtype=q.dtype)
    a = torch.randn(batch, steps, heads, device="cuda")
    b = torch.randn_like(a)
    A_log = torch.randn(heads, device="cuda") * 0.1
    dt_bias = torch.randn_like(A_log) * 0.1
    pool = torch.randn(slots, 2, heads, value_width, width, device="cuda") * 0.1
    state = pool[:, 1]  # real envelope stride: adjacent slots span both layers
    untouched_layer = pool[:, 0].clone()
    initial = state.clone()
    indices = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    rawv = torch.zeros(slots, heads, steps, value_width, device="cuda", dtype=v.dtype)
    rawk = torch.zeros(slots, heads, steps, width, device="cuda", dtype=k.dtype)
    gates = torch.zeros(slots, heads, steps, device="cuda")
    betas = torch.zeros_like(gates)

    def run():
        return gated_delta_rule_mtp(
            A_log, a, dt_bias, q=q, k=k, v=v, b=b,
            initial_state_source=state, initial_state_indices=indices,
            use_qk_l2norm_in_kernel=True, disable_state_update=disable_update,
            cache_ring=True, replayssm_rawv=rawv, replayssm_rawk=rawk,
            replayssm_g=gates, replayssm_beta=betas,
        )

    def reference(before):
        expected_state = before.clone()
        expected = torch.empty_like(v, dtype=torch.float32)
        for row, slot in enumerate([1, 3]):
            h = before[slot].transpose(-1, -2).clone()
            for step in range(steps):
                query, key = q[row, step].float(), k[row, step].float()
                query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
                key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
                g = -A_log.exp() * F.softplus(a[row, step] + dt_bias)
                beta = b[row, step].sigmoid()
                h = h * g.exp()[:, None, None]
                delta = (v[row, step].float() - (h * key[..., None]).sum(-2)) * beta[:, None]
                h = h + key[..., None] * delta[:, None, :]
                expected[row, step] = (h * (query / width**0.5)[..., None]).sum(-2)
            if not disable_update:
                expected_state[slot] = h.transpose(-1, -2)
        return expected, expected_state

    expected, expected_state = reference(initial)
    check(run(), expected)
    torch.testing.assert_close(state, expected_state, rtol=1e-4, atol=1e-5)
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    state.copy_(initial)
    oracle_state = initial.clone()
    previous = expected
    for wave in range(3):
        for tensor in (q, k, v, a, b):
            tensor.copy_(torch.randn_like(tensor) * (wave + 1))
        expected, oracle_state = reference(oracle_state)
        assert not torch.allclose(previous, expected, atol=0.02, rtol=0.05)
        graph.replay()
        torch.cuda.synchronize()
        check(output, expected)
        torch.testing.assert_close(state, oracle_state, rtol=1e-4, atol=1e-5)
        for row, slot in enumerate([1, 3]):
            torch.testing.assert_close(rawv[slot], v[row].transpose(0, 1))
            torch.testing.assert_close(rawk[slot], k[row].transpose(0, 1))
            torch.testing.assert_close(
                gates[slot], (-A_log.exp() * F.softplus(a[row] + dt_bias)).t(),
                rtol=1e-5, atol=1e-6,
            )
            torch.testing.assert_close(betas[slot], b[row].sigmoid().t())
        assert torch.count_nonzero(rawv[[0, 2]]).item() == 0
        torch.testing.assert_close(pool[:, 0], untouched_layer, rtol=0, atol=0)
        previous = expected


@pytest.mark.parametrize("rank,rope", [(256, 32), (512, 64)])
@pytest.mark.parametrize("mode", ["prefill", "decode", "graph"])
def test_mla_nonzero_rope_contract(runtime, rank, rope, mode):
    """The upstream tiny default uses rope=0 and misses real MLA KV geometry."""
    import unittest

    if not torch.cuda.is_available():
        pytest.skip("requires one HCU")
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
        MLAAttentionCase, build_mla_attention_fixture, expected_mla_fixture_output,
        run_mla_attention_case, run_mla_fixture_eager,
    )

    case = MLAAttentionCase(
        name=f"l15_nonzero_rope_{rank}_{rope}_{mode}", backend="triton",
        forward_mode=ForwardMode.EXTEND if mode == "prefill" else ForwardMode.DECODE,
        num_heads=4, page_size=16, prefix_lens=(7, 15, 16),
        extend_lens=(2, 3, 1) if mode == "prefill" else (),
    )
    options = dict(kv_lora_rank=rank, qk_rope_head_dim=rope, dtype=torch.bfloat16)
    if mode != "graph":
        run_mla_attention_case(unittest.TestCase(), case, **options)
        return
    fixture = build_mla_attention_fixture(unittest.TestCase(), case, **options)
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        run_mla_fixture_eager(fixture)  # initialize static metadata outside capture

        def run():
            return fixture.actual_module(fixture.input_hidden, fixture.forward_batch)

        for _ in range(2):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        previous = expected_mla_fixture_output(fixture)
        for wave in range(1, 4):
            fixture.input_hidden.copy_(torch.randn_like(fixture.input_hidden) * wave)
            expected = expected_mla_fixture_output(fixture)
            assert not torch.allclose(previous, expected, atol=0.03, rtol=0.03)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, atol=0.03, rtol=0.03)
            previous = expected
