"""Checkpoint-declared W4A8 layer oracle, checked against official CPU CT."""

import copy

import pytest

torch = pytest.importorskip("torch")

from tools.ct_w4a8_reference import (
    channel_int4, expert, int32_expert, int32_linear, linear, token_int8, validate_group,
)


def group():
    return {
        "format": "pack-quantized",
        "weights": dict(num_bits=4, type="int", symmetric=True, strategy="channel",
                        dynamic=False, group_size=-1),
        "input_activations": dict(num_bits=8, type="int", symmetric=True,
                                 strategy="token", dynamic=True, group_size=None),
        "output_activations": None,
    }


def official():
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.quantization import QuantizationArgs
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize, quantize
    from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp
    from compressed_tensors.compressors.pack_quantized.helpers import (
        pack_to_int32, unpack_from_int32,
    )
    return (QuantizationArgs(**group()["input_activations"]),
            compute_dynamic_scales_and_zp, fake_quantize, quantize,
            pack_to_int32, unpack_from_int32)


def test_token_codes_scales_and_qdq_match_official_ct(request):
    from _cpu_process import rerun_if_host_loaded
    if rerun_if_host_loaded(request):
        return
    args, dynamic, fake, quantize, _, _ = official()
    # Binary-exact scales include halfway cases, saturation, and zero rows.
    row = torch.tensor([127.5, -127.5, .5, -.5, 1.5, -1.5, 2.5, -2.5])
    value = row[None, None, :] * torch.tensor([1., 2., .125, 0.])[None, :, None]
    original = value.clone()
    codes, scales, qdq = token_int8(value)
    expected_scale, zp = dynamic(value, args, torch.nn.Linear(8, 2))
    assert torch.equal(scales, expected_scale)
    assert torch.equal(codes, quantize(value, expected_scale, zp, args, dtype=torch.int8))
    assert torch.equal(qdq, fake(value, expected_scale, zp, args))
    assert codes[0, 0].tolist() == [127, -128, 0, 0, 2, -2, 2, -2]
    assert scales[0, -1, 0] == torch.finfo(torch.float32).eps
    assert torch.equal(value, original)
    # These conventions are numerically different; do not substitute /127.
    assert not torch.equal(scales[0, 0], value[0, 0].abs().max() / 127)


@pytest.mark.parametrize("columns", [1, 7, 8, 9, 128])
def test_packed_channel_weights_match_official_ct(columns, request):
    from _cpu_process import rerun_if_host_loaded
    if rerun_if_host_loaded(request):
        return
    *_, pack, unpack = official()
    signed = (torch.arange(3 * columns).reshape(3, columns) % 16 - 8).to(torch.int8)
    packed = pack(signed, 4)
    scale = torch.tensor([[.5], [2.], [.125]], dtype=torch.bfloat16)
    original = packed.clone()
    weight = channel_int4(packed, scale, columns)
    decoded = unpack(packed, 4, torch.Size((3, columns)))
    assert torch.equal(decoded, signed)
    assert torch.equal(weight, decoded.float() * scale.float())
    assert torch.equal(packed, original)
    # The highest nibble sets the int32 sign bit; arithmetic shifts must mask it.
    word = torch.tensor([[-1]], dtype=torch.int32)
    assert torch.equal(channel_int4(word, torch.ones(1, 1), 8),
                       torch.full((1, 8), 7.))


@pytest.mark.parametrize("limit", [None, .75])
def test_expert_chain_matches_official_qdq_at_both_gemms(limit, request):
    from _cpu_process import rerun_if_host_loaded
    if rerun_if_host_loaded(request):
        return
    args, dynamic, fake, _, pack, unpack = official()
    rng = torch.Generator().manual_seed(194)
    value = torch.randn(2, 3, 16, generator=rng)
    value[0, 0].zero_()
    value[1] *= .03125

    def weights(rows, columns):
        signed = torch.randint(-8, 8, (rows, columns), generator=rng, dtype=torch.int8)
        return pack(signed, 4), torch.rand(rows, 1, generator=rng) * .25 + .01

    gate, up, down = weights(24, 16), weights(24, 16), weights(16, 24)

    def official_linear(x, params):
        scale, zp = dynamic(x, args, torch.nn.Identity())
        qdq = fake(x, scale, zp, args)
        weight = unpack(params[0], 4, torch.Size((params[0].shape[0], x.shape[-1])))
        return torch.nn.functional.linear(qdq, weight.float() * params[1])

    g, u = official_linear(value, gate), official_linear(value, up)
    if limit is not None:
        g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    expected = official_linear(torch.nn.functional.silu(g) * u, down)
    actual = expert(value, gate, up, down, swiglu_limit=limit)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Dequantizing only the weights omits an observable part of the contract.
    weight_only = torch.nn.functional.linear(value, channel_int4(*gate, 16))
    assert not torch.equal(linear(value, *gate), weight_only)


@pytest.mark.parametrize("fault", ["fp8", "group", "asymmetric", "static", "output",
                                    "scale-dtype", "actorder"])
def test_unvalidated_contracts_are_refused(fault):
    config = group()
    if fault == "fp8":
        config["input_activations"]["type"] = "float"
    elif fault == "group":
        config["weights"]["group_size"] = 128
    elif fault == "asymmetric":
        config["weights"]["symmetric"] = False
    elif fault == "static":
        config["input_activations"]["dynamic"] = False
    elif fault == "output":
        config["output_activations"] = copy.deepcopy(config["input_activations"])
    elif fault == "scale-dtype":
        config["input_activations"]["scale_dtype"] = "float16"
    else:
        config["weights"]["actorder"] = "group"
    with pytest.raises(ValueError):
        validate_group(config)


def test_valid_group_is_not_mutated():
    config = group()
    original = copy.deepcopy(config)
    validate_group(config)
    assert config == original


@pytest.mark.parametrize("value", [
    torch.ones(2, 8), torch.ones(1, 2, 8, dtype=torch.bfloat16),
    torch.empty(1, 0, 8), torch.full((1, 2, 8), float("nan")),
    torch.ones(1, 2, 8, device="meta"),
])
def test_invalid_activation_inputs_are_refused(value):
    with pytest.raises(ValueError):
        token_int8(value)


@pytest.mark.parametrize("fault", ["dtype", "shape", "zero-scale", "nan-scale", "columns"])
def test_invalid_packing_or_scales_are_refused(fault):
    packed, scale, columns = torch.zeros(2, 2, dtype=torch.int32), torch.ones(2, 1), 9
    if fault == "dtype":
        packed = packed.long()
    elif fault == "shape":
        scale = torch.ones(2, 2)
    elif fault == "zero-scale":
        scale.zero_()
    elif fault == "nan-scale":
        scale.fill_(float("nan"))
    else:
        columns = 8
    with pytest.raises(ValueError):
        channel_int4(packed, scale, columns)


@pytest.mark.parametrize("columns", [1, 7, 8, 9, 160])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.bfloat16])
def test_integer_linear_matches_official_codes_and_exact_fp64_dot(
    columns, scale_dtype, request,
):
    from _cpu_process import rerun_if_host_loaded
    if rerun_if_host_loaded(request):
        return
    args, dynamic, _, quantize, pack, _ = official()
    rng = torch.Generator().manual_seed(225)
    value = torch.randn(2, 3, columns, generator=rng) * 4
    value[..., -1] = 127.5
    value[0, 0, -1] = -127.5
    value[:, 1] *= .03125
    value[:, 2].zero_()
    signed = torch.randint(-8, 8, (4, columns), generator=rng, dtype=torch.int8)
    packed = pack(signed, 4)
    channel_scale = (torch.rand(4, 1, generator=rng) * .25 + .01).to(scale_dtype)
    activation_scale, zp = dynamic(value, args, torch.nn.Identity())
    codes = quantize(value, activation_scale, zp, args, dtype=torch.int8)
    # Integer inputs and this bound make every FP64 product/sum exact;
    # no rounded dequantized operands participate in the dot.
    exact = torch.matmul(codes.double(), signed.double().t())
    python_sum = sum(int(a) * int(b) for a, b in zip(codes[0, 0], signed[0]))
    assert exact[0, 0, 0] == python_sum
    expected = (exact.float() * activation_scale) * channel_scale.float().t()
    torch.testing.assert_close(
        int32_linear(value, packed, channel_scale), expected, rtol=0, atol=0,
    )


def test_integer_linear_rounds_only_after_full_dot_above_fp32_exact_range(request):
    from _cpu_process import rerun_if_host_loaded
    if rerun_if_host_loaded(request):
        return
    *_, pack, _ = official()
    columns = 32769
    value = torch.full((1, 1, columns), -127.5)
    value[..., -1] = 127.
    signed = torch.full((1, columns), -8, dtype=torch.int8)
    signed[..., -1] = 7
    packed = pack(signed, 4)
    scale = torch.tensor([[.03125]], dtype=torch.bfloat16)
    exact_integer = 32768 * 1024 + 127 * 7
    # The exact sum is odd and larger than 2**24: its FP32 cast rounds.
    rounded = torch.tensor(exact_integer, dtype=torch.float32)
    assert float(rounded) != exact_integer
    expected = (rounded * scale.float()).reshape(1, 1, 1)
    torch.testing.assert_close(int32_linear(value, packed, scale), expected,
                               rtol=0, atol=0)


@pytest.mark.parametrize("limit", [None, .75])
def test_integer_expert_matches_official_quantization_and_fp64_integer_dots(
    limit, request,
):
    from _cpu_process import rerun_if_host_loaded
    if rerun_if_host_loaded(request):
        return
    args, dynamic, _, quantize, pack, _ = official()
    rng = torch.Generator().manual_seed(226)
    value = torch.randn(2, 3, 16, generator=rng)
    value[:, 0].zero_()

    def make_weight(rows, columns):
        signed = torch.randint(-8, 8, (rows, columns), generator=rng,
                               dtype=torch.int8)
        scale = (torch.rand(rows, 1, generator=rng) * .25 + .01).bfloat16()
        return (pack(signed, 4), scale), signed

    gate, gi = make_weight(24, 16)
    up, ui = make_weight(24, 16)
    down, di = make_weight(16, 24)

    def exact_linear(x, signed, scale):
        activation_scale, zp = dynamic(x, args, torch.nn.Identity())
        q = quantize(x, activation_scale, zp, args, dtype=torch.int8)
        dot = torch.matmul(q.double(), signed.double().t()).float()
        return (dot * activation_scale) * scale.float().t()

    g = exact_linear(value, gi, gate[1])
    u = exact_linear(value, ui, up[1])
    if limit is not None:
        g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    expected = exact_linear(torch.nn.functional.silu(g) * u, di, down[1])
    actual = int32_expert(value, gate, up, down, swiglu_limit=limit)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_integer_linear_refuses_possible_int32_overflow():
    value = torch.ones(1, 1, 2097152)
    with pytest.raises(ValueError, match="INT32"):
        int32_linear(value, torch.zeros(1, 1, dtype=torch.int32), torch.ones(1, 1))


@pytest.mark.parametrize("limit", [0, -1., float("inf"), float("nan")])
def test_integer_expert_refuses_invalid_clamp(limit):
    value = torch.ones(1, 1, 8)
    weight = torch.zeros(8, 1, dtype=torch.int32), torch.ones(8, 1)
    with pytest.raises(ValueError, match="swiglu_limit"):
        int32_expert(value, weight, weight, weight, swiglu_limit=limit)
