"""Triton implementations of the pinned state_and_constraints contracts."""

import torch
import triton
import triton.language as tl

@triton.jit
def _mask_logits(LOGITS, MASK, INDICES, FIRST, N_REQUESTS: tl.constexpr, LOGIT_ROWS: tl.constexpr,
                 LOGIT_STRIDE: tl.constexpr, MASK_STRIDE: tl.constexpr, WIDTH: tl.constexpr,
                 STEP: tl.constexpr, HAS_FIRST: tl.constexpr, BLOCK: tl.constexpr):
    req = tl.load(INDICES + tl.program_id(0))
    row = req
    valid = True
    if HAS_FIRST:
        row = tl.load(FIRST + req) + STEP
        end = tl.load(FIRST + req + 1, req + 1 < N_REQUESTS, other=LOGIT_ROWS)
        valid = row < end
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.load(MASK + req * MASK_STRIDE + col // 32, col < WIDTH, other=0).to(tl.uint32)
    old = tl.load(LOGITS + row * LOGIT_STRIDE + col, (col < WIDTH) & valid, other=0)
    new = tl.where((bits & (1 << (col % 32))) != 0, old, -float('inf'))
    tl.store(LOGITS + row * LOGIT_STRIDE + col, new, (col < WIDTH) & valid)


def mask_logits(logits, bitmask, indices, first, step):
    if not indices.numel():
        return
    _mask_logits[(indices.numel(), triton.cdiv(logits.shape[1], 512))](
        logits, bitmask, indices, first, 0 if first is None else first.numel(), logits.shape[0],
        logits.stride(0), bitmask.stride(0), logits.shape[1], step, first is not None, 512)
