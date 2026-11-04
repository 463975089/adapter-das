# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""norm lightop implementation; upstream rounding/layout contracts are unchanged."""

import torch

from .lightop_usage import _count
from .quant_lightop import per_token_cast


def norm(x, out, weight, residual, residual_out, sf, extra_sf, origin, *,
         has_residual, eps, out_scale, do_fp8_cast, num_per_channels,
         col_packed, round_sf, use_packed_ue8m0, return_origin_out, shared_sf_block_m):
    import lightop

    if shared_sf_block_m:
        raise NotImplementedError('LightOp norm candidate does not cover mega-MoE shared SF')
    value = x + residual if has_residual else x
    if has_residual:
        residual_out.copy_(value)
    # The upstream residual sum rounds to input dtype before RMSNorm. Native
    # fused_add_rms_norm has a different FP32-sum policy, so use plain RMSNorm.
    matrix = value.float().reshape(-1, x.shape[-1]).contiguous()
    result = torch.empty_like(matrix)
    lightop.rms_norm(matrix, (weight.float() * out_scale).contiguous(), eps, out=result)
    result = result.view_as(x)
    if return_origin_out:
        origin.copy_(result)
    if not do_fp8_cast:
        out.copy_(result)
    else:
        q, packed = per_token_cast(result.to(x.dtype).reshape(-1, x.shape[-1]), 'e4m3',
                                  num_per_channels, use_tma_aligned_col_major_sf=col_packed,
                                  round_sf=round_sf, use_packed_ue8m0=use_packed_ue8m0)
        out.copy_(q.view_as(out))
        if use_packed_ue8m0:
            raw = packed.contiguous().view(torch.uint8).reshape(*x.shape[:-1], -1)
            if col_packed:
                sf.copy_(raw.view_as(sf))
            else:
                sf.copy_(raw[..., :sf.shape[-1]])
        else:
            sf.copy_(packed.view_as(sf))
    _count('norm.native')
