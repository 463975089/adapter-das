"""torch-native DSA top-k functions for sgl_kernel stub (HCU).

sglang's DSA indexer defaults to the sgl-kernel topk backend; only
fast_topk_transform_fused / fast_topk_transform_ragged_fused /
fast_topk_v2 are missing on the HIP build. These torch equivalents keep the
same contract (values in [-inf-masked windows]; -1 padding for short rows;
paged transform gathers the page_size=1 table).
"""

import torch


def _topk_local_indices(score, lengths, topk, row_starts=None):
    """Per-row top-k of the [row_start, row_start+length) window.

    Returns int32 [B, topk] indices local to the window start, -1 padded.
    """
    batch, width = score.shape
    out = score.new_full((batch, topk), -1, dtype=torch.int32)
    if batch == 0 or topk == 0 or width == 0:
        return out
    if row_starts is None:
        row_starts = torch.zeros_like(lengths)
    row_starts = row_starts.to(dtype=torch.int32, device=score.device)
    lengths = lengths.to(dtype=torch.int32, device=score.device)

    cols = torch.arange(width, dtype=torch.int32, device=score.device).unsqueeze(0)
    valid = (cols >= row_starts.unsqueeze(1)) & (
        cols < (row_starts + lengths).unsqueeze(1)
    )
    masked = score.masked_fill(~valid, float("-inf"))
    k = min(topk, width)
    values, idx = torch.topk(masked, k, dim=-1)
    local = idx.to(torch.int32) - row_starts.unsqueeze(1)
    local = local.masked_fill(values == float("-inf"), -1)
    out[:, :k] = local
    return out


def fast_topk_transform_fused(
    score,
    lengths,
    page_table_size_1,
    cu_seqlens_q=None,
    topk=2048,
    row_starts=None,
):
    """Top-k + page-table transform (page_size=1 table gather)."""
    local = _topk_local_indices(score, lengths, topk, row_starts)
    safe = local.clamp(min=0).long()
    table = page_table_size_1.to(torch.int32)
    gathered = table.gather(1, safe)
    return torch.where(local < 0, local.new_full(local.shape, -1), gathered)


def fast_topk_transform_ragged_fused(score, lengths, topk_indices_offset, topk=2048, row_starts=None):
    """Top-k + ragged-KV base offset transform (extend path)."""
    local = _topk_local_indices(score, lengths, topk, row_starts)
    offset = topk_indices_offset.to(torch.int32).unsqueeze(1)
    return torch.where(
        local < 0, local.new_full(local.shape, -1), local + offset
    )


def fast_topk_v2(score, lengths, topk, row_starts=None):
    """Plain top-k returning local-column indices int32 [B, topk]."""
    return _topk_local_indices(score, lengths, topk, row_starts)


def fast_topk_transform_fused_ragged(*args, **kwargs):  # legacy alias safety
    return fast_topk_transform_ragged_fused(*args, **kwargs)


def install():
    import sgl_kernel

    sgl_kernel.fast_topk_transform_fused = fast_topk_transform_fused
    sgl_kernel.fast_topk_transform_ragged_fused = fast_topk_transform_ragged_fused
    sgl_kernel.fast_topk_v2 = fast_topk_v2
