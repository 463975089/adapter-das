# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""KV compression state kernels for the pinned tilelang_ops/compress.py contracts.

Anchor: deepinfer tilelang_ops/cuda/compress.py @774ce24. For each compressed
token: gather COMPRESS_RATIO (score, latent) rows from the state ring buffer
(positions before the chunk) or the new rows, softmax over the window in FP32,
weighted latent sum, BF16 out. Then the last min(state_size, qlen) new rows of
each sequence are written to the ring. All metadata stays on device (no host
reads), so these run under the forward sync guard and graph capture.
"""

import torch


def _window_rows(state_cache, state_idx, score, latent, pos, qstart, src_row):
    """Gather [n, R, dim] score/latent for absolute positions pos[n, R]."""
    size = state_cache.shape[1]
    from_cache = pos < qstart
    cache_idx = torch.remainder(pos, size)
    slot = state_idx[:, None].expand_as(pos)
    cached_score = state_cache[slot, cache_idx, 0]
    cached_latent = state_cache[slot, cache_idx, 1]
    rows = src_row.clamp(0, max(score.shape[0] - 1, 0))
    new_score = score.index_select(0, rows.flatten()).view(*pos.shape, -1)
    new_latent = latent.index_select(0, rows.flatten()).view(*pos.shape, -1)
    mask = from_cache[..., None]
    return torch.where(mask, cached_score, new_score), torch.where(mask, cached_latent, new_latent)


def _compress(score_win, latent_win):
    weights = torch.softmax(score_win.float(), dim=1)
    return (latent_win.float() * weights).sum(dim=1).to(torch.bfloat16)


def compress_prefill(score, latent, state_cache, state_block_idx, compressed_batch_mapping, cu_seqlens_q,
                     cu_seqlens_k, cu_compressed_seqlens_q, mirror_cp_reverse_index, kv_compressed, *, ratio):
    total_c = compressed_batch_mapping.shape[0]
    seq = compressed_batch_mapping.long()
    qs, qe = cu_seqlens_q.long()[:-1], cu_seqlens_q.long()[1:]
    klen = cu_seqlens_k.long()[1:] - cu_seqlens_k.long()[:-1]
    qstart = klen - (qe - qs)
    state_idx = state_block_idx.long()
    if total_c:
        block_id = torch.arange(total_c, device=score.device) - cu_compressed_seqlens_q.long()[seq]
        new_blocks = klen[seq] // ratio - qstart[seq] // ratio
        valid = (block_id >= 0) & (block_id < new_blocks)
        start = (qstart[seq] // ratio + block_id) * ratio
        pos = start[:, None] + torch.arange(ratio, device=score.device)
        src = pos - qstart[seq][:, None] + qs[seq][:, None]
        if mirror_cp_reverse_index is not None:
            src = mirror_cp_reverse_index.long().index_select(
                0, src.clamp(0, mirror_cp_reverse_index.shape[0] - 1).flatten()).view_as(src)
        score_win, latent_win = _window_rows(state_cache, state_idx[seq], score, latent, pos, qstart[seq][:, None], src)
        result = _compress(score_win, latent_win)
        kv_compressed.copy_(torch.where(valid[:, None], result, torch.zeros((), dtype=result.dtype, device=result.device)))

    # Tail store: the last min(state_size, qlen) new rows of each sequence.
    size = state_cache.shape[1]
    num_seqs = qs.shape[0]
    if not num_seqs:
        return
    lastq = torch.arange(size, device=score.device)
    stored = lastq[None, :] < torch.minimum(qe - qs, torch.full_like(qs, size))[:, None]  # [S, size]
    row = qe[:, None] - lastq[None, :] - 1
    if mirror_cp_reverse_index is not None:
        row = mirror_cp_reverse_index.long().index_select(
            0, row.clamp(0, mirror_cp_reverse_index.shape[0] - 1).flatten()).view_as(row)
    row = row.clamp(0, max(score.shape[0] - 1, 0))
    cache_idx = torch.remainder(torch.remainder(klen - 1, size)[:, None] - lastq[None, :], size)
    slot = state_idx[:, None].expand_as(cache_idx)
    keep = state_cache[slot, cache_idx]  # [S, size, 2, dim]
    new = torch.stack((score.index_select(0, row.flatten()), latent.index_select(0, row.flatten())), 1)
    new = new.view(*row.shape, 2, -1)
    state_cache[slot, cache_idx] = torch.where(stored[..., None, None], new, keep)


def compress_decode(score, latent, state_cache, state_block_idx, context_lens, kv_compressed, *, ratio):
    total_q = score.shape[0]
    if not total_q:
        return
    size = state_cache.shape[1]
    state_idx = state_block_idx.long()
    abs_pos = context_lens.long() - 1
    cache_pos = torch.remainder(abs_pos, size)
    # Reference stores tokens in order, so for a repeated (slot, pos) the last token wins.
    # Every writer of a cell writes the winner's row, so the scatter is order-independent.
    key = state_idx * size + cache_pos
    order = torch.arange(total_q, device=key.device)
    winner = ((key[None, :] == key[:, None]) * order[None, :]).amax(dim=1)
    rows = torch.stack((score.float(), latent.float()), 1).index_select(0, winner)
    state_cache[state_idx, cache_pos] = rows
    ends_block = torch.remainder(abs_pos, ratio) == ratio - 1
    start = abs_pos // ratio * ratio
    pos = start[:, None] + torch.arange(ratio, device=score.device)
    cache_idx = torch.remainder(pos, size)
    slot = state_idx[:, None].expand_as(pos)
    result = _compress(state_cache[slot, cache_idx, 0], state_cache[slot, cache_idx, 1])
    kv_compressed.copy_(torch.where(ends_block[:, None], result, torch.zeros((), dtype=result.dtype, device=result.device)))
