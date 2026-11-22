# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Fuse index translation and short-row ordering after the native torch top-k.

Selection, tie breaking and value order remain torch.topk's responsibility.
Only the existing integer/mask postprocessing is combined into one launch.
"""
from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _indexer_topk_post(Values, Indices, Starts, Ends, Out,
                        WIDTH: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, K)
    start = tl.load(Starts + row)
    end = tl.load(Ends + row)
    values = tl.load(Values + row * K + col)
    indices = tl.load(Indices + row * K + col).to(tl.int32)
    invalid = values == -float('inf')
    if end - start <= K:
        ordered = tl.sort(tl.where(invalid, WIDTH, indices), descending=False)
        result = tl.where(ordered == WIDTH, -1, ordered - start)
    else:
        result = tl.where(invalid, -1, indices - start)
    tl.store(Out + row * K + col, result)


@lru_cache(maxsize=None)
def _supported_device(index):
    p = torch.cuda.get_device_properties(index)
    return getattr(p, 'gcnArchName', '').split(':')[0] == 'gfx938'


def finish_topk(values, indices, starts, ends, k, width):
    """Return exact existing postprocessing, or None outside the tested ABI."""
    if (torch.version.hip is None or not values.is_cuda or values.ndim != 2
            or k != 512 or not 512 <= width < 2**31
            or values.shape[0] <= 0 or values.shape[1] != k
            or values.dtype != torch.float32 or not values.is_contiguous()
            or indices.shape != values.shape or indices.dtype != torch.int64
            or not indices.is_contiguous()
            or starts.shape != values.shape[:1] or ends.shape != starts.shape
            or starts.dtype != torch.int32 or ends.dtype != torch.int32
            or not starts.is_contiguous() or not ends.is_contiguous()
            or any(t.device != values.device for t in (indices, starts, ends))
            or any(t.is_neg() for t in (values, indices, starts, ends))):
        return None
    if not _supported_device(values.device.index):
        return None
    out = torch.empty(values.shape, dtype=torch.int32, device=values.device)
    _indexer_topk_post[(values.shape[0],)](values, indices, starts, ends, out,
                                         width, k, num_warps=4)
    return out
