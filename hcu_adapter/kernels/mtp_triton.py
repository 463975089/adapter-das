"""Triton implementations of the pinned mtp contracts."""

import torch
import triton
import triton.language as tl

@triton.jit
def _store_proposals(SLOTS, TOKENS, PROBS, INDEX, CONF, TOKEN_POOL, PROB_POOL, INDEX_POOL, CONF_POOL,
                     STEPS: tl.constexpr, WIDTH: tl.constexpr, HAS_INDEX: tl.constexpr,
                     HAS_CONF: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    req, step = row // STEPS, row % STEPS
    slot = tl.load(SLOTS + req)
    if slot != 0:
        target = slot * STEPS + step
        if tl.program_id(1) == 0:
            tl.store(TOKEN_POOL + target, tl.load(TOKENS + row))
            if HAS_CONF:
                tl.store(CONF_POOL + target, tl.load(CONF + row))
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        tl.store(PROB_POOL + target * WIDTH + col,
                 tl.load(PROBS + row * WIDTH + col, col < WIDTH, other=0), col < WIDTH)
        if HAS_INDEX:
            tl.store(INDEX_POOL + target * WIDTH + col,
                     tl.load(INDEX + row * WIDTH + col, col < WIDTH, other=0), col < WIDTH)


def store_proposals(slots, tokens, probs, index, confidence, token_pool, prob_pool, index_pool, confidence_pool):
    requests, steps = tokens.shape
    if not requests:
        return
    width = probs.shape[-1]
    _store_proposals[(requests * steps, triton.cdiv(width, 512))](
        slots, tokens, probs, index, confidence, token_pool, prob_pool, index_pool, confidence_pool,
        steps, width, index is not None, confidence is not None, 512)
