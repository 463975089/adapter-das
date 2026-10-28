"""Mirror-CP and YOCO metadata, all scans and lookups on the device."""

import torch


def _prefix(values):
    return torch.cat((values.new_zeros(1), values.cumsum(0, dtype=torch.int32)))


def prepare_mirror(cu_q, cu_k, prefixes, reverse, index, ks, ke, rank, *, cp_size, ratios):
    sq, sk = cu_q[1:] - cu_q[:-1], cu_k[1:] - cu_k[:-1]
    pieces = (sq + 2 * cp_size - 1) // (2 * cp_size)
    rows = [_prefix((sk // ratio).clamp(min=1)) for ratio in ratios]
    rows.append(_prefix(2 * pieces))
    prefixes.copy_(torch.stack(rows))
    t = torch.arange(reverse.numel(), device=cu_q.device, dtype=torch.int32)
    seg = torch.searchsorted(cu_q[1:], t, right=True)
    ps = pieces.index_select(0, seg)
    local = t - cu_q.index_select(0, seg)
    block = local // ps
    half = block // cp_size
    owner = block + half * (2 * cp_size - 1 - 2 * block)
    reverse.copy_(owner * rows[-1][-1] + rows[-1].index_select(0, seg) + half * ps + local - ps * block)
    out = torch.arange(index.numel(), device=cu_q.device, dtype=torch.int32)
    seg = torch.searchsorted(rows[-1][1:], out, right=True)
    ps = pieces.index_select(0, seg)
    offset = out - rows[-1].index_select(0, seg)
    half = offset // ps
    block = rank + half * (2 * cp_size - 1 - 2 * rank)
    local = ps * block + offset - half * ps
    valid = local < sq.index_select(0, seg)
    index.copy_(torch.where(valid, cu_q.index_select(0, seg) + local, -1))
    delta = torch.where(valid, sk.index_select(0, seg) - sq.index_select(0, seg) + local + 1, 0)
    for i, ratio in enumerate(ratios):
        start = rows[i].index_select(0, seg)
        ks[i].copy_(start)
        ke[i].copy_(start + delta // ratio)


def yoco_boundary(cu_q, cu_compact, prefixes, contribution, reverse, rank, *, cp_size):
    sq = cu_q[1:] - cu_q[:-1]
    keep = cu_compact[1:] - cu_compact[:-1]
    clip = sq - keep
    pieces = (sq + 2 * cp_size - 1) // (2 * cp_size)

    def kept(block):
        return (torch.minimum(pieces * (block + 1), sq) - torch.maximum(pieces * block, clip)).clamp(min=0)

    rows = [_prefix(kept(r) + kept(2 * cp_size - 1 - r)) for r in range(cp_size)]
    rows.append(_prefix(2 * pieces))
    prefixes.copy_(torch.stack(rows))
    g = torch.arange(reverse.numel(), device=cu_q.device, dtype=torch.int32)
    seg = torch.searchsorted(cu_compact[1:], g, right=True)
    ps = pieces.index_select(0, seg)
    lower = clip.index_select(0, seg)
    local = lower + g - cu_compact.index_select(0, seg)
    block = local // ps
    half = block // cp_size
    owner = block + half * (2 * cp_size - 1 - 2 * block)
    kept0 = (torch.minimum(ps * (owner + 1), sq.index_select(0, seg)) - torch.maximum(ps * owner, lower)).clamp(min=0)
    ordinal = prefixes[owner.long(), seg] + half * kept0 + local - torch.maximum(ps * block, lower)
    reverse.copy_(owner * contribution.numel() + ordinal)
    p = torch.arange(contribution.numel(), device=cu_q.device, dtype=torch.int32)
    valid = p < rows[rank][-1]
    # Safe lookup for padded tail positions, including ranks with no contribution.
    seg = torch.searchsorted(rows[rank][1:], p, right=True).clamp(max=sq.numel() - 1)
    ps = pieces.index_select(0, seg)
    lower = clip.index_select(0, seg)
    offset = p - rows[rank].index_select(0, seg)
    kept0 = kept(rank).index_select(0, seg)
    half = (offset >= kept0).to(torch.int32)
    block = rank + half * (2 * cp_size - 1 - 2 * rank)
    local = torch.maximum(ps * block, lower) + offset - half * kept0
    value = rows[-1].index_select(0, seg) + half * ps + local - ps * block
    contribution.copy_(torch.where(valid, value, -1))
