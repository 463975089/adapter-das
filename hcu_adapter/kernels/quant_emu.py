# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Finite FP8/FP4 inference subset of the pinned tile-kernels cast contracts.

Anchors (tile_kernels 2.1.0): quant/common.py::get_sf_and_inv (UE8M0 bin via
reciprocal multiply then ceil(log2); BF16 multiplier for BF16 inputs),
quant/per_token_cast_cuda.py, quant/per_token_cast_and_cast_back_cuda.py,
torch/cast.py (E2M1 round-to-nearest-even; low nibble = even element) and
config.py amax clamps. deepinfer tilelang_ops/cuda/fused_transform.py uses
the same per-group quantizer on its BF16 stage output.
"""

import torch

_MAX = {"e4m3": 448.0, "e2m1": 6.0}
# get_amax_clamp_for_quant(fmt, use_e4m3_sf=False)
_AMAX_CLAMP = {"e4m3": 1e-4, "e2m1": 6.0 * 2.0 ** -126}
FLOAT8E4M3_MIN_POSITIVE = 2.0 ** -9


def e2m1_codes(value):
    """FP4 E2M1 round-to-nearest-even with saturation at 6 (NaN -> 6)."""
    magnitude = torch.nan_to_num(value.abs(), nan=6.0, posinf=6.0).clamp(max=6.0)
    code = torch.where(
        magnitude < 2, (magnitude * 2).round(),
        torch.where(magnitude < 4, magnitude.round() + 2, (magnitude * 0.5).round() + 4)).to(torch.uint8)
    return code | (torch.signbit(value).to(torch.uint8) << 3)


def decode_e2m1(codes):
    # Computed on device (no host lookup table, so it is graph-capture safe):
    # code e|m -> e == 0 ? m * 0.5 : (1 + m * 0.5) * 2^(e - 1).
    exponent = ((codes >> 1) & 3).float()
    mantissa = (codes & 1).float()
    magnitude = torch.where(exponent == 0, mantissa * 0.5, (1 + mantissa * 0.5) * torch.exp2(exponent - 1))
    return torch.where((codes & 8) != 0, -magnitude, magnitude)


def unpack_e2m1(packed):
    raw = packed.view(torch.uint8)
    return decode_e2m1(torch.stack((raw & 15, raw >> 4), -1).flatten(-2))


def _narrow(scaled, fmt):
    if fmt == "e4m3":
        return scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    codes = e2m1_codes(scaled)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).view(torch.int8)


def _widen(q, fmt):
    return q.float() if fmt == "e4m3" else unpack_e2m1(q)


def quantize_groups(x, fmt, group, sf_format):
    """Quantize the last dim of BF16/FP32 x in groups of `group`.

    sf_format: "ue8m0" (power-of-two, round_sf), "fp32" (round_sf=False),
    "fp32_rounded" (round_sf, float scales) or "e4m3" (FP8 scales, FP4 data).
    Returns (data, sf, sf_float): data is E4M3 or packed E2M1 int8; sf is uint8
    exponent bytes, float32 or E4M3; sf_float is the FP32 dequant scale.
    """
    if fmt not in _MAX or x.shape[-1] % group:
        raise ValueError("Unsupported cast format or group size")
    finfo_max = _MAX[fmt]
    blocks = x.unflatten(-1, (-1, group))
    amax = blocks.abs().amax(-1).float().clamp_min(_AMAX_CLAMP[fmt])
    if sf_format in ("ue8m0", "fp32_rounded"):
        bits = (amax * (1.0 / finfo_max)).view(torch.int32)
        biased = ((bits - 1) >> 23) + 1
        multiplier = ((254 - biased) << 23).view(torch.float32)
        if x.dtype == torch.bfloat16:
            scaled = (blocks * multiplier.to(torch.bfloat16).unsqueeze(-1)).float()
        else:
            scaled = blocks.float() * multiplier.unsqueeze(-1)
        sf_float = (biased << 23).view(torch.float32)
        sf = biased.to(torch.uint8) if sf_format == "ue8m0" else sf_float
    elif sf_format == "e4m3":
        sf = (amax.clamp_min(FLOAT8E4M3_MIN_POSITIVE * finfo_max) * (1.0 / finfo_max)).clamp_max(
            448.0).to(torch.float8_e4m3fn)
        sf_float = sf.float()
        scaled = blocks.float() * (1.0 / sf_float).unsqueeze(-1)
    elif sf_format == "fp32":
        sf_float = sf = amax / finfo_max
        scaled = blocks.float() * (finfo_max / amax).unsqueeze(-1)
    else:
        raise ValueError("Unknown scale format: " + sf_format)
    return _narrow(scaled.flatten(-2), fmt), sf, sf_float


def dequantize_groups(data, sf_float, fmt, group, dtype=torch.bfloat16):
    values = _widen(data, fmt).unflatten(-1, (-1, group))
    return (values * sf_float.unsqueeze(-1)).flatten(-2).to(dtype)


def per_token_cast(x, fmt, num_per_channels, x_block_size=None,
                   use_tma_aligned_col_major_sf=False, round_sf=False,
                   use_packed_ue8m0=False, use_e4m3_sf=False,
                   stochastic_cast=False, out=None):
    """Quantize plain BF16/FP32 rows, preserving GPU packed UE8M0 layout."""
    if x_block_size is not None or isinstance(x, tuple) or use_e4m3_sf or stochastic_cast:
        raise NotImplementedError("HCU per_token_cast supports deterministic plain BF16/FP32 input only")
    if fmt not in _MAX:
        raise ValueError("per_token_cast format must be e4m3 or e2m1")
    if x.ndim != 2 or not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("per_token_cast requires a BF16/FP32 matrix on HCU")
    tokens, hidden = x.shape
    if (num_per_channels not in (16, 32, 64, 128) or hidden <= 0
            or hidden % 64 or hidden % num_per_channels or x.stride(-1) != 1):
        raise ValueError("per_token_cast requires dense rows and aligned quantization blocks")
    if use_packed_ue8m0 and not round_sf:
        raise ValueError("Packed UE8M0 scales require power-of-two rounding")
    groups = hidden // num_per_channels
    sf_width = (groups + 3) // 4 if use_packed_ue8m0 else groups
    sf_dtype = torch.int32 if use_packed_ue8m0 else torch.float32
    q_shape = (tokens, hidden if fmt == "e4m3" else hidden // 2)
    q_dtype = torch.float8_e4m3fn if fmt == "e4m3" else torch.int8
    if out is None:
        q = torch.empty(q_shape, dtype=q_dtype, device=x.device)
        if use_tma_aligned_col_major_sf:
            aligned_tokens = (tokens + 3) // 4 * 4
            sf = torch.empty_strided((tokens, sf_width), (1, aligned_tokens),
                                     dtype=sf_dtype, device=x.device)
        else:
            sf = torch.empty((tokens, sf_width), dtype=sf_dtype, device=x.device)
    else:
        if use_tma_aligned_col_major_sf:
            raise NotImplementedError("Pinned per_token_cast does not accept column-major out scales")
        q, sf = out
        if (q.device != x.device or q.shape != q_shape or q.dtype != q_dtype
                or not q.is_contiguous() or sf.device != x.device
                or sf.shape != (tokens, sf_width) or sf.dtype != sf_dtype or sf.stride(-1) != 1):
            raise ValueError("per_token_cast output buffers do not match the pinned layout")
    if tokens == 0:
        return q, sf

    sf_format = ("ue8m0" if use_packed_ue8m0 else "fp32_rounded") if round_sf else "fp32"
    data, scales, _ = quantize_groups(x, fmt, num_per_channels, sf_format)
    q.copy_(data)
    if use_packed_ue8m0:
        if use_tma_aligned_col_major_sf:
            raw = sf.T.view(torch.uint8)
            for lane in range(4):
                rows = scales[:, lane::4]
                raw[:rows.shape[1], lane::4].copy_(rows.T)
        else:
            sf.view(torch.uint8)[:, :groups].copy_(scales)
    else:
        sf.copy_(scales)
    return q, sf


def per_token_cast_and_cast_back(x, fmt, num_per_channels, round_sf=False, use_e4m3_sf=False):
    """In-place lossy round trip of x[tokens, hidden] (stride(1) == 1)."""
    if use_e4m3_sf or x.ndim != 2 or x.stride(1) != 1:
        raise NotImplementedError("HCU cast-and-cast-back supports 2D rows without E4M3 scales")
    if num_per_channels not in (16, 32, 64, 128) or x.shape[1] % num_per_channels:
        raise ValueError("cast-and-cast-back requires aligned quantization blocks")
    data, _, sf_float = quantize_groups(x, fmt, num_per_channels, "fp32_rounded" if round_sf else "fp32")
    x.copy_(dequantize_groups(data, sf_float, fmt, num_per_channels, x.dtype))


def swiglu_forward_and_per_token_cast(x, fmt, num_per_channels, psum_num_tokens_per_expert=None,
                                      topk_weights=None, routed_scaling_factor=None,
                                      use_tma_aligned_col_major_sf=False, round_sf=False,
                                      use_packed_ue8m0=False, clamp_value=None, clamped_count=None):
    """Pinned tile_kernels 2.1.0 swiglu_forward_and_per_token_cast (FP32 math).

    out = clamp(xl) / (1 + exp(-clamp(xl))) * clamp(xr) [* w], then optional
    per-token cast. Rows in PSUM alignment gaps are unspecified, as in the
    vendor kernel (its output is torch.empty); here they are computed anyway.
    """
    if clamped_count is not None:
        raise NotImplementedError("HCU SwiGLU does not implement clamp counting")
    if (fmt, num_per_channels) not in (("e4m3", 128), ("e4m3", 32), ("bf16", None), ("fp32", None)):
        raise ValueError("Unsupported SwiGLU output format")
    if x.ndim != 2 or not x.is_contiguous() or x.shape[1] % 2:
        raise ValueError("SwiGLU expects contiguous [tokens, 2 * hidden]")
    hidden = x.shape[1] // 2
    values = x.float()
    left, right = values[:, :hidden], values[:, hidden:]
    if clamp_value is not None:
        left = left.clamp(max=clamp_value)
        right = right.clamp(min=-clamp_value, max=clamp_value)
    out = left / (1 + torch.exp(-left)) * right
    if topk_weights is not None:
        weight = topk_weights.float()
        if routed_scaling_factor is not None:
            weight = weight * routed_scaling_factor
        out = out * weight[:, None]
    if fmt in ("bf16", "fp32"):
        return out.to(torch.bfloat16 if fmt == "bf16" else torch.float32)
    return per_token_cast(out, fmt, num_per_channels, use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
                          round_sf=round_sf, use_packed_ue8m0=use_packed_ue8m0)


def swiglu_forward(x, fmt, psum_num_tokens_per_expert=None, topk_weights=None,
                   routed_scaling_factor=None, clamp_value=None, clamped_count=None):
    """Pinned standalone BF16/FP32 SwiGLU, sharing the fused provider's math."""
    if fmt not in ("bf16", "fp32"):
        raise ValueError("swiglu_forward supports bf16 or fp32 output only")
    return swiglu_forward_and_per_token_cast(
        x, fmt, None, psum_num_tokens_per_expert, topk_weights,
        routed_scaling_factor=routed_scaling_factor, clamp_value=clamp_value,
        clamped_count=clamped_count)
