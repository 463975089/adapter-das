"""Actual upstream execution chains; GPU-only, with independent FP32 oracles."""

import pytest
import torch
import torch.nn.functional as F


def _mamba2_decode_reference(mixer, case, hidden, conv, ssm, slots):
    """CPU FP32 recurrence, with BF16 storage at the public layer boundaries.

    No production projection, convolution, scan or normalization function is
    called. Return the entire updated pools so writes to unrelated slots can
    be checked independently of the model output.
    """
    def cpu(tensor):
        return tensor.detach().cpu().float()

    projected = F.linear(cpu(hidden), cpu(mixer.in_proj.weight)).bfloat16()
    width = case.intermediate_size
    conv_width = width + 2 * case.n_groups * case.state_size
    gate, xbc, dt = projected.split([width, conv_width, case.num_heads], -1)
    conv = conv.clone()
    ssm = ssm.clone()
    outputs = []
    for row, slot in enumerate(slots):
        window = torch.cat([conv[slot].float(), xbc[row].float()[:, None]], -1)
        preactivation = (
            window * cpu(mixer.conv1d.weight).squeeze(1)
        ).sum(-1) + cpu(mixer.conv1d.bias)
        convolved = F.silu(preactivation).bfloat16().float()
        conv[slot] = window[:, 1:].to(conv.dtype)
        x, b, c = convolved.split(
            [width, case.n_groups * case.state_size,
             case.n_groups * case.state_size]
        )
        x = x.reshape(case.num_heads, case.head_dim)
        b = b.reshape(case.n_groups, case.state_size)
        c = c.reshape(case.n_groups, case.state_size)
        delta = F.softplus(dt[row].float() + cpu(mixer.dt_bias))
        y = torch.empty_like(x)
        for head in range(case.num_heads):
            group = head * case.n_groups // case.num_heads
            state = ssm[slot, head] * torch.exp(delta[head] * cpu(mixer.A)[head])
            state = state + delta[head] * torch.outer(x[head], b[group])
            ssm[slot, head] = state
            y[head] = state @ c[group] + cpu(mixer.D)[head] * x[head]
        # The SSM output is stored in activation dtype before gated RMSNorm.
        gated = y.flatten().bfloat16().float() * F.silu(gate[row].float())
        grouped = gated.reshape(case.n_groups, -1)
        normalized = grouped * torch.rsqrt(
            grouped.square().mean(-1, keepdim=True) + mixer.norm.variance_epsilon
        )
        normed = (normalized.flatten() * cpu(mixer.norm.weight)).bfloat16()
        outputs.append(F.linear(normed.float(), cpu(mixer.out_proj.weight)).bfloat16())
    return torch.stack(outputs), conv, ssm


@pytest.mark.parametrize("batch,page_size,groups", [(1, 1, 1), (3, 16, 2)])
def test_mamba2_decode_fresh_input_graph(runtime, batch, page_size, groups):
    """Real MambaMixer2 eager/graph; three changing inputs and remapped state slots."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.test.kits.attention_unittest.attention_methods.mamba2_attention import (
        Mamba2AttentionCase, _conv_states, _ssm_states,
        build_mamba2_attention_fixture,
    )

    class RequiredBackend:
        def skipTest(self, reason):
            pytest.fail(f"required Mamba2 backend unavailable: {reason}")

    case = Mamba2AttentionCase(
        name="adapter_mamba2_fresh_graph", backend="triton",
        forward_mode=ForwardMode.DECODE, num_heads=2, head_dim=16,
        state_size=16, n_groups=groups, conv_kernel=4,
        mamba_chunk_size=16, hidden_size=32, page_size=page_size,
        prefix_lens=(14, 15, 16)[:batch],
    )
    fixture = build_mamba2_attention_fixture(
        RequiredBackend(), case, disable_cuda_graph=False, runner_batch_size=8,
    )
    backend, fb = fixture.backend, fixture.forward_batch
    conv, ssm = _conv_states(fixture), _ssm_states(fixture)
    assert conv.shape[-1] == case.conv_kernel - 1
    assert ssm.shape[1:] == (case.num_heads, case.head_dim, case.state_size)
    conv.copy_(torch.randn_like(conv) * 0.1)
    ssm.copy_(torch.randn_like(ssm) * 0.1)
    initial_conv, initial_ssm = conv.clone(), ssm.clone()
    expected_conv, expected_ssm = conv.cpu().clone(), ssm.cpu().clone()
    backend.init_cuda_graph_state(max_bs=batch, max_num_tokens=batch)
    backend.init_forward_metadata_out_graph(fb, in_capture=True)

    def run():
        with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
            return fixture.actual_module(fb, fixture.hidden_states)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    conv.copy_(initial_conv)
    ssm.copy_(initial_ssm)
    mapping = fixture.runner.req_to_token_pool.req_index_to_mamba_index_mapping
    previous = None
    for wave in range(3):
        # The captured graph must read refreshed metadata rather than the slots
        # used at capture. Selected slots are unique, non-monotonic and nonzero.
        slots = [(2 * row + wave) % 7 + 1 for row in range(batch)][::-1]
        mapping[fb.req_pool_indices] = torch.tensor(slots, device="cuda", dtype=torch.int32)
        fixture.hidden_states.copy_(torch.randn_like(fixture.hidden_states) * (wave + 1) * 0.3)
        backend.init_forward_metadata_out_graph(fb)
        before_conv, before_ssm = conv.clone(), ssm.clone()
        expected, expected_conv, expected_ssm = _mamba2_decode_reference(
            fixture.actual_module.mixer, case, fixture.hidden_states,
            expected_conv, expected_ssm, slots,
        )
        if previous is not None:
            assert not torch.allclose(expected.float(), previous.float(), atol=0.02, rtol=0.05)
        previous = expected.clone()
        eager = run()
        check(eager.cpu(), expected)
        eager_conv, eager_ssm = conv.clone(), ssm.clone()
        conv.copy_(before_conv)
        ssm.copy_(before_ssm)
        graph.replay()
        torch.cuda.synchronize()
        check(captured.cpu(), expected)
        # Both output and recurrent state are checked; an output-only oracle
        # misses corruption that first becomes visible on the next token.
        torch.testing.assert_close(conv, eager_conv, rtol=0, atol=0)
        torch.testing.assert_close(ssm, eager_ssm, rtol=0, atol=0)
        torch.testing.assert_close(conv.cpu(), expected_conv, rtol=0.05, atol=0.002)
        torch.testing.assert_close(ssm.cpu(), expected_ssm, rtol=0.05, atol=0.002)
        untouched = [slot for slot in range(conv.shape[0]) if slot not in slots]
        assert torch.equal(conv[untouched], before_conv[untouched])
        assert torch.equal(ssm[untouched], before_ssm[untouched])


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


@pytest.mark.parametrize("tokens,topk", [(1, 1), (7, 2), (33, 2)])
def test_w8a8_int8_moe_chain(runtime, tokens, topk):
    from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts
    from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config, W8A8Int8MoEMethod

    torch.manual_seed(983 + tokens)
    experts, hidden, intermediate = 4, 128, 128
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * .3
    logits = torch.randn(tokens, experts, device="cuda", dtype=torch.float32)
    layer = torch.nn.Module()
    layer.moe_tp_size = layer.moe_ep_size = 1
    layer.moe_tp_rank = layer.moe_ep_rank = 0
    method = W8A8Int8MoEMethod(W8A8Int8Config())
    with torch.device("cuda"):
        method.create_weights(layer, experts, hidden, intermediate, x.dtype)
    for weight in (layer.w13_weight, layer.w2_weight):
        weight.copy_(torch.randint(-20, 21, weight.shape, device="cuda", dtype=torch.int8))
    for scale in (layer.w13_weight_scale, layer.w2_weight_scale):
        scale.copy_(torch.linspace(.003, .02, scale.numel(), device="cuda").reshape_as(scale))
    method.process_weights_after_loading(layer)
    method.create_moe_runner(layer, MoeRunnerConfig(
        num_experts=experts, num_local_experts=experts, hidden_size=hidden,
        intermediate_size_per_partition=intermediate, top_k=topk, params_dtype=x.dtype,
        inplace=True, gate_up_interleaved=False, routed_scaling_factor=1., layer=layer,
    ))

    def run():
        routing = select_experts(x, logits, TopKConfig(top_k=topk, renormalize=True))
        # The runner writes its output in-place; preserve the oracle inputs
        # and the storage mutated between graph replays.
        dispatch = StandardDispatchOutput(x.clone(), None, routing)
        return method.apply(layer, dispatch).hidden_states

    def quantize(value):
        # Validate the public quantizer independently in FP64, then use its
        # validated integers for the two independent GEMM/activation oracles.
        # Triton's approximate reciprocal can straddle exact half-integers;
        # require exact half-away rounding everywhere outside that FP32 bound.
        quantized, scale = per_token_quant_int8(value.contiguous())
        maximum = value.double().abs().amax(-1, keepdim=True).clamp_min(1e-10)
        scaled = value.double() * (127 / maximum)
        rounded = scaled.sign() * torch.floor(scaled.abs() + .5)
        uncertainty = 4 * torch.finfo(torch.float32).eps * 127
        ambiguous = (scaled.abs().frac() - .5).abs() <= uncertainty
        assert torch.all(ambiguous | (quantized.double() == rounded))
        assert torch.all((quantized.double() - scaled).abs() <= .5 + uncertainty)
        torch.testing.assert_close(scale.double(), maximum / 127, rtol=5e-7, atol=0)
        return quantized.float(), scale

    def reference():
        routing_weights, ids = torch.topk(logits.float().softmax(-1), topk, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(-1, keepdim=True)
        quantized, input_scale = quantize(x)
        outputs = []
        for expert in range(experts):
            gate_up = ((quantized @ layer.w13_weight[expert].float().t())
                       * (input_scale * layer.w13_weight_scale[expert].view(1, -1)))
            gate, up = gate_up.to(x.dtype).float().chunk(2, -1)
            activated = (F.silu(gate) * up).to(x.dtype)
            act_quantized, act_scale = quantize(activated)
            down = ((act_quantized @ layer.w2_weight[expert].float().t())
                    * (act_scale * layer.w2_weight_scale[expert].view(1, -1)))
            outputs.append(down)
        all_outputs = torch.stack(outputs, 1)
        selected = all_outputs.gather(1, ids[..., None].expand(-1, -1, hidden))
        # The down-GEMM epilogue applies routing in FP32 before storing each
        # expert contribution in BF16, then the combine sums those values.
        weighted = (selected * routing_weights[..., None]).to(x.dtype)
        return weighted.float().sum(1).to(x.dtype)

    check(run(), reference())
    replay_three(run, [x, logits], reference)


@pytest.mark.parametrize("kind", ["int8", "fp8", "ct-fp8"])
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
    elif kind == "ct-fp8":
        from compressed_tensors.quantization import QuantizationArgs
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
            CompressedTensorsW8A8Fp8,
        )

        # The real checkpoint route: channel weights, dynamic token
        # activations, process-after-loading and default output padding.
        scheme = CompressedTensorsW8A8Fp8(
            QuantizationArgs(num_bits=8, type="float", strategy="channel"),
            is_static_input_scheme=False,
        )
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
        if kind == "ct-fp8":
            return (result + bias.float()).to(x.dtype)
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
