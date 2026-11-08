# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Framework-free LightOp implementation of tensor/token-channel scaled MM."""
from __future__ import annotations

import torch


def _scale_shape(scale, count):
    return (scale.ndim <= 2 and (scale.numel() == 1 or scale.shape == (count,)
                                 or scale.shape == (count, 1)))


def supported(A, B, As, Bs, out_dtype, bias, architecture):
    """Metadata-only vendor admission; unsupported contracts stay upstream."""
    if architecture not in ('gfx936', 'gfx938'):
        return False
    allowed = (torch.int8, torch.float8_e4m3fn) if architecture == 'gfx938' else (torch.int8,)
    if A.dtype not in allowed or B.dtype != A.dtype or A.ndim != 2 or B.ndim != 2:
        return False
    m, k = A.shape
    n = B.shape[1]
    if not m or not n or not k or k != B.shape[0] or k % 16 or n % 16:
        return False
    if not A.is_cuda or B.device != A.device or not A.is_contiguous() or not B.t().is_contiguous():
        return False
    if out_dtype not in (torch.float16, torch.bfloat16):
        return False
    if (As.dtype != torch.float32 or Bs.dtype != torch.float32
            or As.device != A.device or Bs.device != A.device
            or not _scale_shape(As, m) or not _scale_shape(Bs, n)):
        return False
    return (bias is None or (bias.dtype == out_dtype and bias.device == A.device
                            and bias.shape == (n,) and bias.is_contiguous()))


def scaled_mm(A, B, As, Bs, out_dtype, bias, *, gemm):
    """Caller checks admission. Expand scalar scales without host synchronization."""
    m, k = A.shape
    n = B.shape[1]
    sa = As.reshape(-1, 1).expand(m, 1).contiguous()
    sb = Bs.reshape(-1, 1).expand(n, 1).contiguous()
    status, out = gemm(A, B.t(), sa, sb, m, n, k, 'NT', out_dtype, bias)
    if status is not True:
        raise RuntimeError('LightOp scaled-MM reported failure')
    if (not isinstance(out, torch.Tensor) or out.shape not in ((m,n), (1,m,n))
            or out.dtype != out_dtype or out.device != A.device):
        raise RuntimeError('LightOp scaled-MM returned incompatible shape/dtype/device')
    return out.view(m, n)
