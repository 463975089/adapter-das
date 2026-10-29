"""Torch/emu implementations of the pinned mtp contracts."""

import torch

def dflash_input_store(hidden, out):
    if hidden.shape[1] == 1:
        out.copy_(hidden[:, 0])
    else:
        out.copy_(hidden.float().mean(dim=1))


def confidence_head(hidden, markov, weight, out, temperature):
    features = torch.cat((hidden, markov), dim=-1) if markov is not None else hidden
    logits = features.float() @ weight.float()
    out.copy_(torch.sigmoid((logits / temperature).clamp(-60., 60.)))


def copy_decode_input(*args, has_driver_inputs):
    if len(args) != 27:
        raise ValueError("MTP decode copy requires three groups of nine tensors")
    src, padding, dst = args[:9], args[9:18], args[18:]
    for i in range(9 if has_driver_inputs else 1):
        n = src[i].shape[0]
        tail = dst[i].shape[0] - n
        if tail < 0 or tail > padding[i].shape[0]:
            raise ValueError("MTP source/padding cannot fill the destination")
        dst[i][:n].copy_(src[i])
        dst[i][n:].copy_(padding[i][:tail])


def schedule_by_confidence(remap, previous, cumulative, seqlens_q, total_q):
    n, steps = cumulative.shape
    if not n:
        return
    # Explicit FP32 serial products match the pinned CUDA kernel; cumprod may
    # use a different internal precision and flip a near-equal ranking.
    safe = remap.clamp(min=0).long()
    if previous.shape[0]:
        gathered = previous.index_select(0, safe)
        gathered = torch.where((remap != -1)[:, None], gathered, 0.)
    else:
        torch._assert_async((remap == -1).all(), "MTP remap references an empty prior batch")
        gathered = torch.zeros_like(cumulative)
    acc = torch.ones(n, dtype=torch.float32, device=remap.device)
    for j in range(steps):
        acc = acc * gathered[:, j]
        cumulative[:, j].copy_(acc)
    order = torch.argsort(cumulative.flatten(), descending=True, stable=True)
    ranks = torch.empty(n * steps, dtype=torch.int32, device=remap.device)
    ranks.scatter_(0, order, torch.arange(n * steps, dtype=torch.int32, device=remap.device))
    seqlens_q.copy_(1 + (ranks.view(n, steps) < total_q - n).sum(dim=1))


def dflash_get_input(slots, positions, first, last, new_slots, new_positions, input_ids,
                     valid_mask, force_random, kv_index, kv_len, window_size, *,
                     swa_block_size, has_valid_token_mask, has_gate_force_random):
    n, steps = new_slots.shape
    if not n:
        return
    if not slots.numel():
        raise ValueError("DFlash source needs a padding token even when every request is padding")
    valid = first >= 0
    idx = first.clamp(min=0).long()
    base_position = positions.index_select(0, idx)
    base_slot = slots.index_select(0, idx) - base_position % swa_block_size
    accepted = base_position + last
    draft_position = accepted[:, None] + 1 + torch.arange(steps, device=positions.device)
    new_positions.copy_(torch.where(valid[:, None], draft_position, 0))
    new_slots.copy_(torch.where(valid[:, None], draft_position % swa_block_size + base_slot[:, None], -1))
    input_ids.zero_()
    if has_valid_token_mask:
        valid_mask.copy_(valid[:, None].expand_as(valid_mask))
    if has_gate_force_random:
        force_random.zero_()
    cols = torch.arange(kv_index.shape[1], device=positions.device)
    logical = accepted[:, None] - (window_size - 1) + cols
    values = torch.where(valid[:, None] & (cols < kv_len) & (logical >= 0),
                         logical % swa_block_size + base_slot[:, None], -1)
    kv_index[:n * steps].copy_(values[:, None, :].expand(n, steps, -1).reshape(n * steps, -1))
