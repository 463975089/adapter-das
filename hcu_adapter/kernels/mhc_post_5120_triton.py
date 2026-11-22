# SPDX-License-Identifier: Apache-2.0
"""H5120 Post with the official Torch FP32 sum and BF16 storage boundary."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def supported(x, residual, post, comb):
    tensors = (x, residual, post, comb)
    # "highest" does not disable autocast of the Torch reference's FP32 bmm.
    # Keep autocast callers on that reference instead of fixed FP32 FMA.
    if (not all(isinstance(t, torch.Tensor) for t in tensors)
            or residual.ndim != 3 or residual.shape[1:] != (4, 5120)
            or residual.shape[0] <= 0
            or x.shape != (residual.shape[0], 5120)
            or post.shape not in ((residual.shape[0], 4), (residual.shape[0], 4, 1))
            or comb.shape != (residual.shape[0], 4, 4)
            or x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16
            or post.dtype != torch.float32 or comb.dtype != torch.float32
            or not all(t.is_cuda and t.device == x.device and t.is_contiguous() for t in tensors)
            or torch.get_float32_matmul_precision() != 'highest'
            or torch.is_autocast_enabled('cuda')):
        return False
    return getattr(torch.cuda.get_device_properties(x.device), 'gcnArchName', '').split(':')[0] == 'gfx938'


@triton.jit
def _mhc_post_5120(X, R, P, C, O, BLOCK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    head = tl.arange(0, 4)[:, None]
    r0 = tl.load(R + token * 20480 + col, col < 5120, other=0).to(tl.float32)[None, :]
    r1 = tl.load(R + token * 20480 + 5120 + col, col < 5120, other=0).to(tl.float32)[None, :]
    r2 = tl.load(R + token * 20480 + 10240 + col, col < 5120, other=0).to(tl.float32)[None, :]
    r3 = tl.load(R + token * 20480 + 15360 + col, col < 5120, other=0).to(tl.float32)[None, :]
    c0 = tl.load(C + token * 16 + head)
    c1 = tl.load(C + token * 16 + 4 + head)
    c2 = tl.load(C + token * 16 + 8 + head)
    c3 = tl.load(C + token * 16 + 12 + head)
    # Match the gfx938 Torch FP32 bmm's forward K-order, including its +0
    # accumulator. Other reduction trees change BF16 tie decisions.
    mixed = tl.fma(r0, c0, 0.)
    mixed = tl.fma(r1, c1, mixed)
    mixed = tl.fma(r2, c2, mixed)
    mixed = tl.fma(r3, c3, mixed)
    post = tl.load(P + token * 4 + head)
    x = tl.load(X + token * 5120 + col, col < 5120, other=0).to(tl.float32)[None, :]
    # Torch materializes the FP32 post product before the final addition.
    # Keep that boundary separate from the sum's explicit FMA instructions.
    result = mixed + post * x
    tl.store(O + token * 20480 + head * 5120 + col[None, :], result, col[None, :] < 5120)


def post_5120(x, residual, post, comb):
    if not supported(x, residual, post, comb):
        raise ValueError('H5120 Post requires contiguous gfx938 BF16, FP32 mixes, '
                         'highest precision and CUDA autocast disabled')
    out = torch.empty_like(residual)
    rows = residual.shape[0]
    # Keep the exact per-element arithmetic; reduce CTA overhead at decode
    # buckets while preserving the existing launch for larger/prefill batches.
    block, warps = (1024, 2) if 32 <= rows <= 64 else (
        (256, 8) if rows <= 16 else (256, 4))
    _mhc_post_5120[(rows, triton.cdiv(5120, block))](
        x, residual, post, comb, out, block,
        num_warps=warps, enable_fp_fusion=False)
    return out
