# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Private exact FP32 copy/padding for the admitted HIP paged-MQA path."""
import torch
import triton
import triton.language as tl


@triton.jit
def _owned_logits_mask(Scores, Lengths, Out, S0: tl.constexpr, S1: tl.constexpr,
                       WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(Lengths + row)
    # Do not read undefined scorer padding. No arithmetic on valid score bits.
    value = tl.load(Scores + row * S0 + columns.to(tl.int64) * S1,
                    mask=(columns < WIDTH) & (columns < length), other=0.0)
    tl.store(Out + row * WIDTH + columns, value, mask=columns < WIDTH)


def owned_logits_mask(scores, lengths):
    if (scores.ndim != 2 or scores.dtype != torch.float32
            or scores.device.type != "cuda"
            or lengths.device != scores.device
            or lengths.dtype not in (torch.int32, torch.int64)
            or not lengths.is_contiguous()
            or not (lengths.ndim == 1 or
                    (lengths.ndim == 2 and lengths.shape[1] == 1))
            or lengths.numel() != scores.shape[0]):
        raise ValueError(
            "owned logits mask requires CUDA FP32 scores and contiguous integer row lengths"
        )
    rows, width = scores.shape
    # Raw Triton launches use the ambient device/its current stream. Match
    # the tensor device just as the previous PyTorch mask did, then restore it.
    with torch.cuda.device(scores.device):
        output = torch.empty((rows, width), dtype=torch.float32, device=scores.device)
        if rows and width:
            _owned_logits_mask[(rows, triton.cdiv(width, 1024))](
                scores, lengths, output, *scores.stride(), width, 1024, num_warps=4,
            )
    return output
