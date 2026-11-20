# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Convert full vendor logits to Core request-local columns without index grids."""
from vllm.triton_utils import tl, triton


@triton.jit
def _compact(source, output, starts, ends, source_stride: tl.constexpr,
             output_stride: tl.constexpr,
             width: tl.constexpr, source_width: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * block + tl.arange(0, block)
    start = tl.load(starts + row)
    stop = tl.load(ends + row)
    absolute = start + col
    valid = (col < width) & (absolute >= 0) & (absolute < stop) & (absolute < source_width)
    value = tl.load(source + row * source_stride + absolute, mask=valid, other=-float('inf'))
    tl.store(output + row * output_stride + col, value, mask=col < width)


def compact(source, output, starts, ends, row_offset):
    if source.shape[0] and output.shape[1]:
        destination = output[row_offset:row_offset + source.shape[0]]
        _compact[(source.shape[0], triton.cdiv(output.shape[1], 256))](
            source, destination, starts, ends, source.stride(0), output.stride(0),
            output.shape[1], source.shape[1], 256)
