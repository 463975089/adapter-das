# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Byte-preserving FlashMLA projection permutations, pinned to vendor 2e5429f."""

import torch


def _validate(weight, scale, ndim):
    if weight.ndim != ndim or scale.ndim != ndim:
        raise ValueError("FlashMLA projection and scale ranks must match the API")
    if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.int32:
        raise TypeError("FlashMLA projection requires E4M3FN weight and packed INT32 scales")
    if weight.device.type != "cuda" or scale.device != weight.device:
        raise ValueError("FlashMLA weight and scale must share a CUDA/HIP device")
    if weight.stride(-1) != 1 or scale.stride(-2) != 1:
        raise ValueError("Projection K must be contiguous; scales must be MN-major")


def _scale_output(values):
    mn, pk = values.shape[-2:]
    aligned = (mn + 3) // 4 * 4
    stride = ((aligned * pk,) if values.ndim == 3 else ()) + (1, aligned)
    output = torch.empty_strided(values.shape, stride, device=values.device, dtype=torch.int32)
    output.copy_(values)
    return output


def permute_q_b_proj(weight_and_sf, h_q, d_q):
    """Interleave 16-row head-dimension chunks across Q heads, including scales."""
    weight, scale = weight_and_sf
    _validate(weight, scale, 2)
    if h_q <= 0 or d_q <= 0 or d_q % 16 or weight.shape[0] != h_q * d_q:
        raise ValueError("Q projection requires positive heads and a head dimension divisible by 16")
    rows, k = weight.shape
    if scale.shape[0] != rows or scale.shape[1] == 0:
        raise ValueError("Q projection scale shape mismatch")
    gran = k // (4 * scale.shape[1])
    if gran not in (32, 128) or k % (4 * gran):
        raise ValueError("Q projection requires aligned per32/per128 packed scales")
    # Move bytes rather than numerically casting the FP8 tensor.
    moved = weight.view(torch.uint8).reshape(h_q, d_q // 16, 16, k)
    result = moved.permute(1, 0, 2, 3).contiguous().reshape(rows, k).view(weight.dtype)
    moved_sf = scale.reshape(h_q, d_q // 16, 16, scale.shape[1])
    values = moved_sf.permute(1, 0, 2, 3).reshape(scale.shape)
    return result, _scale_output(values)


def permute_wv_proj(weight_and_sf, wv_group_size, d_o):
    """Interleave 32-column head chunks and repack their individual scale bytes."""
    weight, scale = weight_and_sf
    _validate(weight, scale, 3)
    if d_o != 512 or wv_group_size <= 0:
        raise ValueError("Pinned FlashMLA WV permutation supports d_o=512 and positive group size")
    groups, rows, k = weight.shape
    if k != wv_group_size * d_o or tuple(scale.shape) != (groups, rows, k // 128):
        raise ValueError("WV projection requires per32 scales with matching grouped dimensions")
    moved = weight.view(torch.uint8).reshape(groups, rows, wv_group_size, d_o // 32, 32)
    result = moved.permute(0, 1, 3, 2, 4).contiguous().reshape(weight.shape).view(weight.dtype)
    exponents = torch.stack([(scale >> (8 * byte)) & 255 for byte in range(4)], dim=-1)
    exponents = exponents.flatten(-2).reshape(groups, rows, wv_group_size, d_o // 32)
    moved_sf = exponents.permute(0, 1, 3, 2).reshape(groups, rows, k // 32)
    packed = moved_sf[..., 0::4].clone()
    for byte in range(1, 4):
        packed.bitwise_or_(moved_sf[..., byte::4] << (8 * byte))
    return result, _scale_output(packed)
