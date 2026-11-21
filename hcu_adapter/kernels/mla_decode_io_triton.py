# SPDX-License-Identifier: Apache-2.0
"""Exact native decode output masking and writeback on gfx938."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


def native_query_and_sink(q, attn_sink, native_heads):
    """Native full-head contiguous inputs already have the required layout."""
    rows, heads, _ = q.shape
    admitted = (not q.is_cuda or (rows > 0 and getattr(
        torch.cuda.get_device_properties(q.device), 'gcnArchName', '').split(':')[0] == 'gfx938'))
    if (admitted and heads == native_heads and q.is_contiguous()
            and (not q.is_cuda or q.data_ptr() % 256 == 0)):
        queries = q.unsqueeze(1)
    else:
        queries = torch.zeros((rows, 1, native_heads, 512), dtype=q.dtype, device=q.device)
        queries[:, 0, :heads].copy_(q)
    sink = None
    if attn_sink is not None:
        if (admitted and heads == native_heads and attn_sink.numel() == native_heads
                and attn_sink.is_contiguous()
                and (not attn_sink.is_cuda or attn_sink.data_ptr() % 256 == 0)):
            sink = attn_sink
        else:
            sink = torch.full((native_heads,), -torch.inf, dtype=torch.float32, device=q.device)
            sink[:heads].copy_(attn_sink[:heads])
    return queries, sink


if triton is not None:
    @triton.jit
    def _mask_copy_native_mla(Result, Output, Main, Extra,
                              WIDTH: tl.constexpr, HAS_EXTRA: tl.constexpr,
                              BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        count = tl.load(Main + row)
        if HAS_EXTRA:
            count += tl.load(Extra + row)
        value = tl.load(Result + row * WIDTH + col, (col < WIDTH) & (count > 0), other=0.)
        tl.store(Output + row * WIDTH + col, value, col < WIDTH)


def try_mask_copy(result, output, main_lens, extra_lens):
    tensors = (result, output, main_lens) + (() if extra_lens is None else (extra_lens,))
    if (triton is None or result.ndim != 3 or result.shape[0] <= 0
            or result.shape[1:] not in ((64, 512), (128, 512))
            or output.shape != result.shape or result.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
            or not all(t.is_cuda and t.device == result.device and t.is_contiguous() for t in tensors)
            or main_lens.shape != (result.shape[0],) or main_lens.dtype != torch.int32
            or (extra_lens is not None and (extra_lens.shape != main_lens.shape or extra_lens.dtype != torch.int32))):
        return False
    if getattr(torch.cuda.get_device_properties(result.device), 'gcnArchName', '').split(':')[0] != 'gfx938':
        return False
    width = result.shape[1] * 512
    _mask_copy_native_mla[(result.shape[0], triton.cdiv(width, 1024))](
        result, output, main_lens, extra_lens, width, extra_lens is not None,
        1024, num_warps=4)
    return True
