"""CPU FP32 layer oracle for CT channel-INT4/dynamic-token-INT8.

This describes the checkpoint's declared quantization, independently of the
community W4AFP8 scheme and the adapter's W4A16 fallback. It is not a model
loader, a serving implementation, or a model-accuracy acceptance result.
The activation convention is checked against compressed-tensors in tests.
"""

import torch


def validate_group(group):
    """Accept only the audited channel-INT4, token-INT8 contract."""
    if (not isinstance(group, dict) or group.get("format") != "pack-quantized"
            or group.get("output_activations") is not None):
        raise ValueError("requires pack-quantized weights without output quantization")
    for label, bits, strategy, dynamic, size in (
        ("weights", 4, "channel", False, -1),
        ("input_activations", 8, "token", True, None),
    ):
        args = group.get(label)
        if (not isinstance(args, dict) or args.get("num_bits") != bits
                or args.get("type") != "int" or args.get("symmetric") is not True
                or args.get("strategy") != strategy or args.get("dynamic") is not dynamic
                or args.get("group_size") != size
                or any(args.get(key) is not None for key in (
                    "actorder", "block_structure", "scale_dtype", "zp_dtype",
                ))
                or args.get("observer_kwargs") not in (None, {})):
            raise ValueError(f"unsupported CT W4A8 {label} contract")


def token_int8(value):
    """Return integer codes, FP32 scales, and QDQ for [batch, token, hidden].

    The signed range is [-128, 127], with absmax / 127.5 and ties-to-even
    rounding. A zero row uses FP32 epsilon, matching CT 0.17.0 FP32 QDQ.
    Require explicit batch/token axes: that library's token reducer assumes
    3D input, and a 2D input would instead select one tensor-wide scale.
    """
    if (value.device.type != "cpu" or value.dtype != torch.float32
            or value.ndim != 3 or value.numel() == 0):
        raise ValueError("reference activations must be nonempty CPU FP32 [B, T, K]")
    if not torch.isfinite(value).all():
        raise ValueError("reference activations must be finite")
    scale = value.abs().amax(-1, keepdim=True) / 127.5
    scale = torch.where(scale == 0, torch.finfo(torch.float32).eps, scale)
    codes = (value / scale).clamp(-128, 127).round().to(torch.int8)
    return codes, scale, codes.float() * scale


def token_int8_native(value):
    """CT token INT8 QDQ with the input's floating point storage dtype.

    Unlike ``token_int8``, scales, division and QDQ retain FP32/BF16/FP16.
    Rounding the quotient to the input dtype before ties-to-even is part
    of this contract; promoting the whole operation to FP32 is different.
    """
    if (value.device.type != "cpu"
            or value.dtype not in (torch.float32, torch.bfloat16, torch.float16)
            or value.ndim != 3 or value.numel() == 0):
        raise ValueError("requires nonempty CPU FP32/BF16/FP16 [B, T, K]")
    if not torch.isfinite(value).all():
        raise ValueError("reference activations must be finite")
    scale = value.abs().amax(-1, keepdim=True) / 127.5
    scale = torch.where(scale == 0, torch.finfo(value.dtype).eps, scale)
    codes = (value / scale).clamp(-128, 127).round().to(torch.int8)
    return codes, scale, codes.to(value.dtype) * scale


def _channel_codes(packed, scale, columns):
    if (packed.device.type != "cpu" or packed.dtype != torch.int32
            or packed.ndim != 2 or packed.shape[0] == 0
            or type(columns) is not int or columns < 1
            or packed.shape[1] != (columns + 7) // 8):
        raise ValueError("requires CPU int32 [N, ceil(K/8)] packed weights")
    if (scale.device.type != "cpu"
            or scale.dtype not in (torch.bfloat16, torch.float32)
            or scale.shape != (packed.shape[0], 1)
            or not torch.isfinite(scale).all() or not (scale > 0).all()):
        raise ValueError("requires finite positive CPU channel scales [N, 1]")
    shifts = torch.arange(8, dtype=torch.int64) * 4
    unsigned = ((packed.long().unsqueeze(-1) >> shifts) & 15).flatten(1)
    return unsigned[:, :columns] - 8


def channel_int4(packed, scale, columns):
    """Decode CT int32 words, low nibble first, subtracting the offset 8."""
    return _channel_codes(packed, scale, columns).float() * scale.float()


def linear(value, packed, scale):
    """Apply input INT8 QDQ and a channel-INT4 Linear in CPU FP32."""
    _, _, activation = token_int8(value)
    weight = channel_int4(packed, scale, value.shape[-1])
    return torch.nn.functional.linear(activation, weight)


def int32_linear(value, packed, scale):
    """CPU oracle for integer accumulation followed by FP32 rescaling.

    INT64 accumulation independently models an overflow-free INT32 dot.
    Cast that sum to FP32, then multiply token and channel scales in order.
    This is a separate execution contract from ``linear``: dequantizing
    operands before an FP32 GEMM can round differently, and a later
    quantization boundary can amplify that difference.
    """
    if value.ndim == 3 and value.shape[-1] > 2097151:
        raise ValueError("K exceeds the safe INT32 accumulation bound")
    codes, activation_scale, _ = token_int8(value)
    weights = _channel_codes(packed, scale, value.shape[-1])
    accumulated = torch.matmul(codes.to(torch.int64), weights.t())
    return (accumulated.float() * activation_scale) * scale.float().t()


def expert(value, gate, up, down, *, swiglu_limit=None):
    """One gated expert, without routing/combine, in the FP32 reference dtype.

    Each weight is a (packed, channel_scale) pair. Activation QDQ is applied
    at both Linear boundaries. This does not reproduce BF16 intermediate
    rounding or establish the community FP8 route's numerical semantics.
    """
    if swiglu_limit is not None and (
        not isinstance(swiglu_limit, (float, int))
        or not 0 < swiglu_limit < float("inf")
    ):
        raise ValueError("swiglu_limit must be finite and positive")
    gate_value = linear(value, *gate)
    up_value = linear(value, *up)
    if gate_value.shape != up_value.shape:
        raise ValueError("gate and up projections must have the same shape")
    if down[0].shape[0] != value.shape[-1]:
        raise ValueError("down projection must restore the hidden dimension")
    if swiglu_limit is not None:
        gate_value = gate_value.clamp(max=swiglu_limit)
        up_value = up_value.clamp(min=-swiglu_limit, max=swiglu_limit)
    intermediate = torch.nn.functional.silu(gate_value) * up_value
    return linear(intermediate, *down)


def int32_expert(value, gate, up, down, *, swiglu_limit=None):
    """One integer-accumulating expert, with FP32 activation intermediates.

    Quantization still follows CT's audited FP32 token convention. This
    supplements, rather than replaces, the official fake-quant/FP32 Linear
    oracle in ``expert``. Neither function establishes serving or model
    accuracy acceptance, including BF16 intermediate storage.
    """
    if swiglu_limit is not None and (
        not isinstance(swiglu_limit, (float, int))
        or not 0 < swiglu_limit < float("inf")
    ):
        raise ValueError("swiglu_limit must be finite and positive")
    gate_value = int32_linear(value, *gate)
    up_value = int32_linear(value, *up)
    if gate_value.shape != up_value.shape:
        raise ValueError("gate and up projections must have the same shape")
    if down[0].shape[0] != value.shape[-1]:
        raise ValueError("down projection must restore the hidden dimension")
    if swiglu_limit is not None:
        gate_value = gate_value.clamp(max=swiglu_limit)
        up_value = up_value.clamp(min=-swiglu_limit, max=swiglu_limit)
    intermediate = torch.nn.functional.silu(gate_value) * up_value
    return int32_linear(intermediate, *down)


def int32_native_linear(value, packed, scale):
    """Integer execution with native token quantization and output storage.

    This explicitly stores the rescaled projection in the input dtype.
    It is distinct from both FP32-intermediate references and from a
    fake-quantized floating point Linear.
    """
    if value.ndim == 3 and value.shape[-1] > 2097151:
        raise ValueError("K exceeds the safe INT32 accumulation bound")
    codes, activation_scale, _ = token_int8_native(value)
    weights = _channel_codes(packed, scale, value.shape[-1])
    accumulated = torch.matmul(codes.to(torch.int64), weights.t())
    output = (accumulated.float() * activation_scale.float()) * scale.float().t()
    return output.to(value.dtype)


def int32_native_expert(value, gate, up, down, *, swiglu_limit=None):
    """Explicit integer expert with native projection and fused-product storage.

    CT supplies the activation quantization convention, not an assertion
    that integer GEMM matches its floating point fake-quantized Linear.
    No routing, combine or model acceptance is established here.
    """
    if swiglu_limit is not None and (
        type(swiglu_limit) not in (float, int)
        or not 0 < swiglu_limit < float("inf")
    ):
        raise ValueError("swiglu_limit must be finite and positive")
    gate_value = int32_native_linear(value, *gate)
    up_value = int32_native_linear(value, *up)
    if gate_value.shape != up_value.shape:
        raise ValueError("gate and up projections must have the same shape")
    if down[0].shape[0] != value.shape[-1]:
        raise ValueError("down projection must restore the hidden dimension")
    if swiglu_limit is not None:
        gate_value = gate_value.clamp(max=swiglu_limit)
        up_value = up_value.clamp(min=-swiglu_limit, max=swiglu_limit)
    intermediate = (
        torch.nn.functional.silu(gate_value.float()) * up_value.float()
    ).to(value.dtype)
    return int32_native_linear(intermediate, *down)
