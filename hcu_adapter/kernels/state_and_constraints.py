"""Circular state transfer and constrained-output device operations."""

import torch
import triton
import triton.language as tl


def select_state_slice(state_tensor, state_cache, last_hit_position):
    infer = state_tensor.shape[1]
    pos = torch.arange(infer, device=state_tensor.device) + last_hit_position - infer
    state_tensor.index_copy_(1, pos.remainder(infer).long(),
                             state_cache.index_select(1, pos.remainder(state_cache.shape[1]).long()))


def load_state_to_cache(state_tensor, state_cache, last_hit_position):
    infer = state_tensor.shape[1]
    pos = torch.arange(infer, device=state_tensor.device) + last_hit_position - infer
    state_cache.index_copy_(1, pos.remainder(state_cache.shape[1]).long(),
                            state_tensor.index_select(1, pos.remainder(infer).long()))


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


def detect_outlier(data, identifier, max_abs):
    # DTK Triton's printf/assert runtime buffers are unavailable on this build:
    # the negative diagnostic probe VMFaulted. Use PyTorch's supported async
    # device assertion; value/offset diagnostics remain an explicit R-15 gap.
    torch._assert_async(((data.float() > -max_abs) & (data.float() < max_abs)).all(),
                        f"outlier detected: id={identifier}, permitted abs(value) < {max_abs}")
