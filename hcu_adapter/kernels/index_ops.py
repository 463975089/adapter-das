"""Index-construction kernels for the pinned tilelang_ops contracts (no deepinfer imports)."""

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


def expand_swa_index(pivots, out, *, backend):
    """Fill int32 [n, width] from [n, 4] (offset1, len1, offset2, len2) pivots."""
    if backend == "torch":
        return _torch_expand_swa_index(pivots, out)
    width = out.shape[1]
    block = min(1024, triton.next_power_of_2(width))
    with torch.cuda.device(pivots.device):
        _expand_swa_index[(pivots.shape[0], triton.cdiv(width, block))](pivots, out, width, block)


def _torch_expand_swa_index(pivots, out):
    col = torch.arange(out.shape[1], dtype=torch.int32, device=pivots.device)
    offset1, length1, offset2, length2 = (p.unsqueeze(1) for p in pivots.unbind(1))
    second = torch.where(col < length1 + length2, offset2 - length1 + col, -1)
    torch.where(col < length1, offset1 + col, second, out=out)


def kv_index_cat(kv_index, kv_ca_index, out_swa, out_ca, total_k_swa, total_k):
    """out_swa = kv_index; out_ca = kv_ca_index + total_k_swa + total_k (-1 kept)."""
    out_swa.copy_(kv_index)
    shifted = kv_ca_index + (total_k_swa + total_k)
    out_ca.copy_(torch.where(kv_ca_index == -1, kv_ca_index, shifted))


def convert_kv_index(topk_idxs, context_lens, block_tables, block_size):
    """In place: logical token index -> physical slot via block_tables; k >= context_len -> -1."""
    b, s, k = topk_idxs.shape
    n = block_tables.shape[1]
    logical = torch.div(topk_idxs, block_size, rounding_mode="floor").long().clamp(0, n - 1)
    physical = block_tables.gather(1, logical.reshape(b, s * k)).reshape_as(topk_idxs)
    slot = physical * block_size + topk_idxs % block_size
    valid = torch.arange(k, device=topk_idxs.device).view(1, 1, k) < context_lens.unsqueeze(-1)
    topk_idxs.copy_(torch.where(valid, slot, -1))


def candidate_block_max(logits, valid_lens, block_scores, valid_blocks, *, block_size):
    rows, max_len = logits.shape
    num_blocks = (max_len + block_size - 1) // block_size
    padded = torch.full((rows, num_blocks * block_size), float("-inf"), dtype=torch.bfloat16, device=logits.device)
    padded[:, :max_len] = logits
    cols = torch.arange(num_blocks * block_size, device=logits.device)
    padded.masked_fill_(cols[None, :] >= valid_lens[:, None], float("-inf"))
    scores = padded.view(rows, num_blocks, block_size).amax(-1).clamp_max(torch.finfo(torch.bfloat16).max)
    blocks = torch.div(valid_lens + block_size - 1, block_size, rounding_mode="floor").clamp_max(num_blocks)
    last = (blocks - 1).clamp_min(0).long().unsqueeze(1)
    last_scores = torch.where(blocks[:, None] > 0, torch.full_like(scores.gather(1, last), float("inf")),
                              scores.gather(1, last))
    scores.scatter_(1, last, last_scores)
    block_scores.copy_(scores)
    valid_blocks[:rows].copy_(blocks)


def candidate_compact_lens(candidate_blocks, valid_lens, compact_lens, *, block_size):
    starts = candidate_blocks * block_size
    tokens = (valid_lens[:, None] - starts).clamp(0, block_size)
    tokens = torch.where(candidate_blocks < 0, 0, tokens)
    compact_lens.copy_(tokens.sum(dim=1, dtype=torch.int32))


def restore_candidate_indices(compact_indices, candidate_blocks, output_idx_offset, *, block_size, has_offset):
    capacity = candidate_blocks.shape[1] * block_size
    valid = (compact_indices >= 0) & (compact_indices < capacity)
    safe = compact_indices.clamp(0, capacity - 1)
    slots = torch.div(safe, block_size, rounding_mode="floor").long()
    block_ids = candidate_blocks.gather(1, slots)
    restored = block_ids * block_size + safe.remainder(block_size)
    if has_offset:
        restored = restored + output_idx_offset[:, None]
    compact_indices.copy_(torch.where(valid & (block_ids >= 0), restored, -1))


def load_kv_index(kv_index_offset_len, kv_index, flatten_block_tables, *, block_size, has_block_tables):
    """(offset, len) rows -> int32 index rows (pinned get_kv_index.py refs; -1 beyond len).

    has_block_tables: paged slot of token offset + j through flatten_block_tables.
    block_size > 0 without tables: ring slot of offset + j inside offset's block (SWA).
    block_size == 0: contiguous offset + j (prefill CA).
    """
    n, length = kv_index.shape
    if not n:
        return
    offset = kv_index_offset_len[:, 0:1].long()
    count = kv_index_offset_len[:, 1:2].long()
    j = torch.arange(length, device=kv_index.device)[None, :]
    valid = j < count
    if has_block_tables:
        pos = offset + torch.minimum(j, count)
        blocks = flatten_block_tables.long()[(pos // block_size).clamp(0, max(flatten_block_tables.shape[0] - 1, 0))]
        value = blocks * block_size + torch.remainder(pos, block_size)
    elif block_size:
        value = offset // block_size * block_size + torch.remainder(offset + j, block_size)
    else:
        value = offset + j
    kv_index.copy_(torch.where(valid, value, -1))
