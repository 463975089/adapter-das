# SPDX-License-Identifier: Apache-2.0
"""Candidate masking over live decode columns, retaining a static row pitch."""
import torch
import triton
import triton.language as tl

from .indexer_topk_post_triton import _supported_device


@triton.jit
def _live_flags(Candidates, Ends, Flags, WIDTH: tl.constexpr,
                 NBLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                 CROW: tl.constexpr, CCOL: tl.constexpr, K: tl.constexpr,
                 TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.cdiv(tl.minimum(tl.load(Ends + row), WIDTH), BLOCK_SIZE)
    lane = tl.arange(0, TILE)
    for tile in range(tl.cdiv(blocks, TILE)):
        col = tile * TILE + lane
        tl.store(Flags + row * (NBLOCKS + 1) + col, 0, col < blocks)
    tl.store(Flags + row * (NBLOCKS + 1) + NBLOCKS, 0)
    ids = tl.arange(0, triton.next_power_of_2(K))
    block = tl.load(Candidates + row * CROW + ids * CCOL, ids < K, other=-1)
    block = tl.minimum(block, NBLOCKS)
    tl.debug_barrier()
    tl.store(Flags + row * (NBLOCKS + 1) + block, 1,
             (ids < K) & (block >= 0) & ((block < blocks) | (block == NBLOCKS)))


@triton.jit
def _live_mask(X, Ends, Flags, WIDTH: tl.constexpr, XROW: tl.constexpr,
                NBLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    end = tl.minimum(tl.load(Ends + row), WIDTH)
    edge = tl.load(Flags + row * (NBLOCKS + 1) + NBLOCKS)
    for tile in range(tl.program_id(1), tl.cdiv(end, TILE), tl.num_programs(1)):
        col = tile * TILE + tl.arange(0, TILE)
        keep = tl.load(Flags + row * (NBLOCKS + 1) + col // BLOCK_SIZE,
                       col < end, other=0)
        keep = (keep != 0) | ((col == WIDTH - 1) & (edge != 0))
        tl.store(X + row * XROW + col, -float('inf'), (col < end) & ~keep)


def mask_decode_candidates(logits, ends, candidates, block_size):
    """Lengths bound both selection and this mask; tails need no writes."""
    if (not logits.is_cuda or torch.version.hip is None
            or not _supported_device(logits.device.index)
            or logits.dtype != torch.float32 or logits.stride(1) != 1):
        return False
    rows, width = logits.shape
    if not rows or not width:
        return True
    blocks = triton.cdiv(width, block_size)
    flags = torch.empty((rows, blocks + 1), dtype=torch.uint8, device=logits.device)
    _live_flags[(rows,)](candidates, ends, flags, width, blocks, block_size,
                         *candidates.stride(), candidates.shape[1], 1024, num_warps=4)
    sms = torch.cuda.get_device_properties(logits.device).multi_processor_count
    _live_mask[(rows, min(triton.cdiv(width, 1024), triton.cdiv(sms, rows)))](
        logits, ends, flags, width, logits.stride(0), blocks, block_size, 1024, num_warps=4)
    return True
