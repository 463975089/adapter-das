# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Torch implementation of the existing norm contract."""

def norm(x, out, weight, residual, residual_out, sf, extra_sf, origin, *,
                has_residual, eps, out_scale, do_fp8_cast, num_per_channels,
                col_packed, round_sf, use_packed_ue8m0,
                return_origin_out, shared_sf_block_m):
    import torch

    value = x + residual if has_residual else x
    if has_residual:
        residual_out.copy_(value)
    value = value.float()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    result = normalized * (weight.float() * out_scale)
    if return_origin_out:
        origin.copy_(result)
    if not do_fp8_cast:
        out.copy_(result)
        return
    # Upstream casts the norm result back to input dtype BEFORE measuring amax.
    q = result.to(x.dtype).float().reshape(*x.shape[:-1], -1, num_per_channels)
    amax = q.abs().amax(-1).clamp_min(1e-4)
    if round_sf:
        scale_bits = (amax / 448.0).view(torch.int32)
        exponent_byte = ((scale_bits - 1) >> 23) + 1
        multiplier = ((254 - exponent_byte) << 23).view(torch.float32)
        scale = (exponent_byte << 23).view(torch.float32)
        packed = exponent_byte.to(torch.uint8)
    else:
        multiplier = 448.0 / amax
        scale = amax / 448.0
    out.copy_((q * multiplier[..., None]).reshape(x.shape).to(out.dtype))
    if use_packed_ue8m0:
        if col_packed:
            # The final incomplete int32 pack has unspecified padding bytes.
            for lane in range(4):
                count = packed[..., lane::4].shape[-1]
                sf[..., :count, lane].copy_(packed[..., lane::4])
        else:
            sf.copy_(packed)
    else:
        sf.copy_(scale)
    if shared_sf_block_m:
        row = torch.arange(x.shape[1], device=x.device)
        local = row % shared_sf_block_m
        mapped = (row // shared_sf_block_m * ((shared_sf_block_m + 127) // 128 * 128) +
                  local // 128 * 128 + local % 32 * 4 + local % 128 // 32)
        extra_sf.index_copy_(0, mapped, packed.reshape(x.shape[1], -1, 4))
