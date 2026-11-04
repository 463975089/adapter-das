# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Rotary embedding with the pinned TileKernels arithmetic (no deepinfer imports).

Anchor: tile_kernels 2.1.0 transform/rope_cuda.py and deepinfer
tilelang_ops/cuda/fused_transform.py: FP32 math, y0 = fma(x0, cos, -(x1*sin)),
y1 = fma(x0, sin, x1*cos) (conjugate swaps the sin signs), rounded once to
the input dtype. Triton uses tl.fma so results are bitwise reproducible.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope(X, Pos, CS, T, H, XS_T, XS_H, CS_S, POS_OFFSET,
          HALF: tl.constexpr, BLOCK: tl.constexpr, INTERLEAVED: tl.constexpr,
          CONJUGATE: tl.constexpr, HAS_POS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    token = row // H
    head = row % H
    j = tl.arange(0, BLOCK)
    mask = j < HALF
    if HAS_POS:
        position = tl.load(Pos + token).to(tl.int64) + POS_OFFSET
    else:
        position = token + POS_OFFSET
    cos = tl.load(CS + position * CS_S + j, mask, other=0.0)
    sin = tl.load(CS + position * CS_S + HALF + j, mask, other=0.0)
    base = X + token * XS_T + head * XS_H
    if INTERLEAVED:
        i0 = 2 * j
        i1 = 2 * j + 1
    else:
        i0 = j
        i1 = HALF + j
    x0 = tl.load(base + i0, mask, other=0.0).to(tl.float32)
    x1 = tl.load(base + i1, mask, other=0.0).to(tl.float32)
    if CONJUGATE:
        y0 = tl.fma(x0, cos, x1 * sin)
        y1 = tl.fma(x0, -sin, x1 * cos)
    else:
        y0 = tl.fma(x0, cos, -(x1 * sin))
        y1 = tl.fma(x0, sin, x1 * cos)
    tl.store(base + i0, y0.to(X.dtype.element_ty), mask)
    tl.store(base + i1, y1.to(X.dtype.element_ty), mask)


def rope_inplace(x, cos_sin_cache, positions, *, interleaved, conjugate, position_offset=0):
    """Rotate x[tokens, heads, rot_dim] (stride(-1) == 1) in place."""
    tokens, heads, rot_dim = x.shape
    if rot_dim % 2 or x.stride(-1) != 1 or cos_sin_cache.dtype != torch.float32:
        raise ValueError("rope expects even rot_dim, dense last dim and FP32 cos/sin cache")
    if cos_sin_cache.shape[1] < rot_dim or cos_sin_cache.stride(1) != 1:
        raise ValueError("cos/sin cache must hold rot_dim dense columns")
    if not tokens * heads:
        return x
    half = rot_dim // 2
    with torch.cuda.device(x.device):
        _rope[(tokens * heads,)](
            x, positions if positions is not None else x, cos_sin_cache, tokens, heads,
            x.stride(0), x.stride(1), cos_sin_cache.stride(0), position_offset,
            half, triton.next_power_of_2(half), interleaved, conjugate, positions is not None,
            num_warps=1, enable_fp_fusion=False)
    return x


def apply_rotary(query, cos_sin_cache, key=None, positions=None, interleaved=False,
                 conjugate=False, seqlen_offset=0):
    """Pinned tile_kernels.transform.apply_rotary contract (in place, 3D or 4D)."""
    for x in (query,) if key is None else (query, key):
        if x.shape[-1] != cos_sin_cache.shape[1]:
            raise ValueError("apply_rotary requires the last dim to equal the cache's rot_dim")
        if x.ndim == 4:
            batch, seqlen, heads, rot_dim = x.shape
            if x.stride(0) != seqlen * x.stride(1):
                raise NotImplementedError("HCU apply_rotary needs mergeable batch/seq strides for 4D input")
            pos = None
            if positions is not None:
                pos = positions.reshape(batch * seqlen)
            elif batch > 1:
                pos = torch.arange(seqlen, device=x.device).repeat(batch)
            rope_inplace(x.view(batch * seqlen, heads, rot_dim) if x.is_contiguous() else
                         x.flatten(0, 1), cos_sin_cache, pos,
                         interleaved=interleaved, conjugate=conjugate, position_offset=seqlen_offset)
        else:
            rope_inplace(x, cos_sin_cache, positions, interleaved=interleaved,
                         conjugate=conjugate, position_offset=seqlen_offset)
