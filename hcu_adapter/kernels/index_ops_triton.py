# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Triton sliding-window index construction."""

import torch
import triton
import triton.language as tl

@triton.jit
def _expand_swa_index(P, Out, W: tl.constexpr, BLOCK_W: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    offset1 = tl.load(P + row * 4)
    length1 = tl.load(P + row * 4 + 1)
    offset2 = tl.load(P + row * 4 + 2)
    length2 = tl.load(P + row * 4 + 3)
    # int32 arithmetic wraps like the reference.
    value = tl.where(col < length1, offset1 + col,
                     tl.where(col < length1 + length2, offset2 - length1 + col, -1))
    tl.store(Out + row * W + col, value, col < W)


def expand_swa_index(pivots, out):
    width = out.shape[1]
    block = min(1024, triton.next_power_of_2(width))
    with torch.cuda.device(pivots.device):
        _expand_swa_index[(pivots.shape[0], triton.cdiv(width, block))](pivots, out, width, block)
