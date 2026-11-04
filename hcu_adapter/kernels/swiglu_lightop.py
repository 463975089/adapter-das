# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""swiglu lightop implementation; upstream rounding/layout contracts are unchanged."""

import torch

from .lightop_usage import _count
from .quant_lightop import per_token_cast


def swiglu_forward_and_per_token_cast(x, fmt, num_per_channels, psum_num_tokens_per_expert=None,
                                      topk_weights=None, routed_scaling_factor=None,
                                      use_tma_aligned_col_major_sf=False, round_sf=False,
                                      use_packed_ue8m0=False, clamp_value=None, clamped_count=None):
    import lightop

    if clamped_count is not None or fmt not in ('e4m3', 'bf16', 'fp32'):
        raise NotImplementedError('Unsupported LightOp SwiGLU format or clamp counting')
    if x.ndim != 2 or not x.is_contiguous() or x.shape[1] % 2:
        raise ValueError('SwiGLU requires a contiguous [M,2H] matrix')
    values = x.float()
    h = x.shape[1] // 2
    if clamp_value is not None:
        values = torch.cat((values[:, :h].clamp(max=clamp_value),
                            values[:, h:].clamp(-clamp_value, clamp_value)), -1)
    result = torch.empty((x.shape[0], h), device=x.device, dtype=torch.float32)
    if x.shape[0]:
        lightop.fuse_silu_and_mul(values, result)
        _count('swiglu.native')
    if topk_weights is not None:
        weights = topk_weights.float()
        if routed_scaling_factor is not None:
            weights = weights * routed_scaling_factor
        result *= weights[:, None]
    if fmt != 'e4m3':
        return result.to(torch.bfloat16 if fmt == 'bf16' else torch.float32)
    return per_token_cast(result, fmt, num_per_channels,
                          use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
                          round_sf=round_sf, use_packed_ue8m0=use_packed_ue8m0)


def swiglu_forward(x, fmt, psum_num_tokens_per_expert=None, topk_weights=None,
                   routed_scaling_factor=None, clamp_value=None, clamped_count=None):
    return swiglu_forward_and_per_token_cast(
        x, fmt, None, psum_num_tokens_per_expert, topk_weights, routed_scaling_factor,
        clamp_value=clamp_value, clamped_count=clamped_count)
