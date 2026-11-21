# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Native exact-rank Top512 plus original indexer output ordering.

The native API excludes NaNs and compares ordered IEEE FP32 keys. A monotone
bit mapping places NaNs above +inf while preserving all other score ranks.
Selection covers the whole score row, as the old torch.topk did; starts/ends
affect output translation and short-row ordering only. Tied score membership
may differ, as it does across repeated original multiblock top-k calls.
"""
import torch
import triton
import triton.language as tl

from .indexer_topk_post_triton import _supported_device


@triton.jit
def _topk_native_scores(X, Scores, Lengths, Offsets,
                        WIDTH: tl.constexpr, PITCH: tl.constexpr,
                        XROW: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0).to(tl.int64), tl.program_id(1)
    col = block * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + row * XROW + col, col < WIDTH, other=-float('inf'))
    bits = value.to(tl.uint32, bitcast=True)
    negative = (bits & 0x80000000) != 0
    zero = (bits & 0x7fffffff) == 0
    nan = (bits & 0x7fffffff) > 0x7f800000
    # Native ordered keys distinguish signed zero. Moving positive encodings
    # down one bit frees +inf for NaNs; the original two zero signs collapse
    # to -0, and +min-subnormal maps to +0 (strictly above -0 in native order).
    mapped = tl.where(nan, 0x7f800000,
              tl.where(zero, 0x80000000, tl.where(negative, bits, bits - 1)))
    tl.store(Scores + row * PITCH + col, mapped.to(tl.float32, bitcast=True), col < PITCH)
    if block == 0:
        tl.store(Lengths + row, WIDTH)
        tl.store(Offsets + row, 0)


@triton.jit
def _topk_live_scores(X, Ends, Scores, Offsets,
                      WIDTH: tl.constexpr, PITCH: tl.constexpr,
                      XROW: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    length = tl.minimum(tl.load(Ends + row), WIDTH)
    if tl.program_id(1) == 0:
        tl.store(Offsets + row, 0)
    for tile in range(tl.program_id(1), tl.cdiv(length, BLOCK), tl.num_programs(1)):
        col = tile * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(X + row * XROW + col, col < length, other=-float('inf'))
        bits = value.to(tl.uint32, bitcast=True)
        negative = (bits & 0x80000000) != 0
        zero = (bits & 0x7fffffff) == 0
        nan = (bits & 0x7fffffff) > 0x7f800000
        mapped = tl.where(nan, 0x7f800000,
                  tl.where(zero, 0x80000000, tl.where(negative, bits, bits - 1)))
        tl.store(Scores + row * PITCH + col,
                 mapped.to(tl.float32, bitcast=True), col < length)


@triton.jit
def _topk_native_output(X, Indices, Starts, Ends, Out,
                        WIDTH: tl.constexpr, XROW: tl.constexpr,
                        K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, K)
    idx = tl.load(Indices + row * K + lane)
    value = tl.load(X + row * XROW + idx, (idx >= 0) & (idx < WIDTH), other=-float('inf'))
    invalid = (idx < 0) | (idx >= WIDTH) | (value == -float('inf'))
    start, end = tl.load(Starts + row), tl.load(Ends + row)
    if end - start <= K:
        ordered = tl.sort(tl.where(invalid, 2147483647, idx), descending=False)
        result = tl.where(ordered == 2147483647, -1, ordered - start)
    else:
        bits = value.to(tl.uint32, bitcast=True)
        key = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
        key = tl.where((bits & 0x7fffffff) > 0x7f800000, 0xffffffff, key)
        key = tl.where(invalid, 0, key).to(tl.uint32)
        # This installed Triton exposes scalar sort, but no key/value sort.
        # Keep the complete FP32 rank and int32 index in a uint64 sort key.
        packed = (key.to(tl.uint64) << 32) | idx.to(tl.uint32).to(tl.uint64)
        packed_sorted = tl.sort(tl.where(invalid, 0, packed), descending=True)
        selected = (packed_sorted & 0xffffffff).to(tl.int32)
        result = tl.where((packed_sorted >> 32) == 0, -1, selected - start)
    tl.store(Out + row * K + lane, result)


def select_topk(logits, starts, ends, k, *, valid_lengths=None):
    """Select exact ranks, returning None outside the native contract.

    The default retains global Top512 semantics. ``valid_lengths`` declares
    fully materialized zero-based score prefixes; other columns may be
    uninitialized. Those prefixes use the native ragged Top512/1024/2048 ABI
    at any positive storage width and follow current device lengths on replay.
    """
    if (torch.version.hip is None or not logits.is_cuda or logits.ndim != 2
            or logits.dtype != torch.float32 or logits.stride(1) != 1
            or k not in ((512, 1024, 2048) if valid_lengths is not None else (512,))
            or logits.shape[0] <= 0 or logits.shape[1] <= 0
            or (valid_lengths is None and not 512 <= logits.shape[1] <= 32768)
            or logits.stride(0) < logits.shape[1]
            or starts.shape != logits.shape[:1] or ends.shape != starts.shape
            or starts.dtype != torch.int32 or ends.dtype != torch.int32
            or not starts.is_contiguous() or not ends.is_contiguous()
            or starts.device != logits.device or ends.device != logits.device
            or any(t.is_neg() for t in (logits, starts, ends))):
        return None
    if valid_lengths is not None and (
            valid_lengths.shape != ends.shape or valid_lengths.dtype != torch.int32
            or valid_lengths.device != logits.device or not valid_lengths.is_contiguous()):
        return None
    if not _supported_device(logits.device.index):
        return None
    try:
        from lightop.fuse_topk_transform import topk_transform_ragged_v2
    except ImportError:
        return None
    rows, width = logits.shape
    pitch = triton.cdiv(width, 4) * 4
    scores = torch.empty((rows, pitch), dtype=torch.float32, device=logits.device)
    lengths = (torch.empty((rows,), dtype=torch.int32, device=logits.device)
               if valid_lengths is None else valid_lengths)
    offsets = torch.empty_like(lengths)
    indices = torch.empty((rows, k), dtype=torch.int32, device=logits.device)
    out = torch.empty_like(indices)
    if valid_lengths is None:
        _topk_native_scores[(rows, triton.cdiv(pitch, 4096))](
            logits, scores, lengths, offsets,
            width, pitch, logits.stride(0), 4096, num_warps=4)
    else:
        sms = torch.cuda.get_device_properties(logits.device).multi_processor_count
        _topk_live_scores[(rows, min(triton.cdiv(pitch, 4096), triton.cdiv(sms, rows)))](
            logits, lengths, scores, offsets,
            width, pitch, logits.stride(0), 4096, num_warps=4)
    topk_transform_ragged_v2(scores, lengths, out_offsets=offsets, out_indices=indices)
    _topk_native_output[(rows,)](logits, indices, starts, ends, out,
        width, logits.stride(0), k, num_warps=4)
    return out
