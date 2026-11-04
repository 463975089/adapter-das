# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""mhc lightop implementation; upstream rounding/layout contracts are unchanged."""

import torch

from .lightop_usage import _count


def hc_projection(x, weight, eps):
    import lightop

    result = lightop.hc_mix_stats(x.contiguous(), weight.contiguous(), eps)
    if result is None:
        _count('mhc.compatibility')
        value = x.float()
        return (value @ weight.float().T) * torch.rsqrt(value.square().mean(-1) + eps)[:, None]
    _count('mhc.native')
    return result


def mega_mhc(*args, **kwargs):
    from .mhc_emu import mega_mhc as implementation

    return implementation(*args, **kwargs, mix_backend='lightop')
