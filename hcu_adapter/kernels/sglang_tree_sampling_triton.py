import torch
import triton
import triton.language as tl


@triton.jit
def _traverse(Predict, Accept, Count, Candidates, Retrieve, Next, Sibling,
              Coins, Target, Draft, Rows, Slots, D: tl.constexpr, S: tl.constexpr,
              V: tl.constexpr, SINGLE: tl.constexpr, ACC: tl.constexpr):
    b = tl.program_id(0)
    node = tl.full((), 0, tl.int64)
    row = b.to(tl.int64) * D
    last = tl.load(Retrieve + b * D)
    coin = tl.load(Coins + b * D)
    tl.store(Accept + b * S, last)
    accepted = 0
    mass = 0.0
    active = True
    while (accepted < S - 1) & active:
        node = tl.load(Next + b * D + node)
        found = False
        while (node != -1) & (~found):
            token = tl.load(Candidates + b * D + node)
            value = tl.load(Target + row * V + token)
            mass += value
            if (value > 0.0) & ((coin < mass / ACC) | (value >= SINGLE)):
                index = tl.load(Retrieve + b * D + node)
                tl.store(Predict + last, token)
                accepted += 1
                tl.store(Accept + b * S + accepted, index)
                last = index
                row = b.to(tl.int64) * D + node
                coin = tl.load(Coins + b * D + node)
                mass = 0.0
                found = True
            else:
                tl.store(Draft + row * V + token, value)
                node = tl.load(Sibling + b * D + node)
        active = node != -1
    tl.store(Count + b, accepted)
    tl.store(Rows + b, row)
    tl.store(Slots + b, last)


@triton.jit
def _partial(Target, Draft, Rows, Count, Partial, V: tl.constexpr,
             S: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    b, tile = tl.program_id(0), tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(Rows + b)
    count = tl.load(Count + b)
    q = tl.load(Target + row * V + offsets, offsets < V, other=0.0)
    p = tl.load(Draft + row * V + offsets, (offsets < V) & (count != S - 1), other=0.0)
    mass = tl.maximum(q - p, 0.0).to(tl.float64)
    tl.store(Partial + b * TILES + tile, tl.sum(mass, 0))


@triton.jit
def _sample(Target, Draft, Rows, Slots, Count, Partial, Coins, Predict,
            V: tl.constexpr, S: tl.constexpr, TILES: tl.constexpr,
            TILE_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    tiles = tl.arange(0, TILE_BLOCK)
    masses = tl.load(Partial + b * TILES + tiles, tiles < TILES, other=0.0)
    prefix = tl.cumsum(masses, 0)
    total = tl.sum(masses, 0)
    u = tl.load(Coins + b).to(tl.float64) * total
    tile = tl.min(tl.where((tiles < TILES) & (prefix > u) & (masses > 0), tiles, TILES), 0)
    last_tile = tl.max(tl.where((tiles < TILES) & (masses > 0), tiles, -1), 0)
    tile = tl.where(tile == TILES, tl.maximum(last_tile, 0), tile)
    prior = tl.sum(tl.where(tiles == tile - 1, prefix, 0.0), 0)
    local_u = tl.maximum(u - prior, 0.0)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(Rows + b)
    count = tl.load(Count + b)
    q = tl.load(Target + row * V + offsets, offsets < V, other=0.0)
    p = tl.load(Draft + row * V + offsets, (offsets < V) & (count != S - 1), other=0.0)
    mass = tl.maximum(q - p, 0.0).to(tl.float64)
    cdf = tl.cumsum(mass, 0)
    selected = tl.min(tl.where((offsets < V) & (cdf > local_u) & (mass > 0), offsets, V), 0)
    last_valid = tl.max(tl.where((offsets < V) & (mass > 0), offsets, -1), 0)
    selected = tl.where(selected == V, tl.where(last_valid >= 0, last_valid, V - 1), selected)
    slot = tl.load(Slots + b)
    tl.store(Predict + slot, selected)


def tree_speculative_sampling_target_only(
    predicts, accept_index, accept_token_num, candidates, retrive_index,
    retrive_next_token, retrive_next_sibling, uniform_samples,
    uniform_samples_for_final_sampling, target_probs, draft_probs,
    threshold_single=1.0, threshold_acc=1.0, deterministic=True,
):
    from hcu_adapter.ops.selector import select

    select("sglang_tree_sampling", "triton")
    if candidates.ndim != 2 or target_probs.ndim != 3 or accept_index.ndim != 2:
        raise ValueError("Invalid tree sampling tensor ranks")
    batch, depth = candidates.shape
    steps = accept_index.shape[1]
    vocab = target_probs.shape[2]
    if depth < 1 or steps < 1 or vocab < 1:
        raise ValueError("Tree sampling requires nonempty tree and vocabulary")
    if not 0 <= threshold_single <= 1 or not 0 <= threshold_acc <= 1:
        raise ValueError("Tree sampling thresholds must be in [0, 1]")
    specs = (
        (predicts, (batch * depth,), torch.int32),
        (accept_index, (batch, steps), torch.int32),
        (accept_token_num, (batch,), torch.int32),
        (candidates, (batch, depth), torch.int64),
        (retrive_index, (batch, depth), torch.int64),
        (retrive_next_token, (batch, depth), torch.int64),
        (retrive_next_sibling, (batch, depth), torch.int64),
        (uniform_samples, (batch, depth), torch.float32),
        (uniform_samples_for_final_sampling, (batch,), torch.float32),
        (target_probs, (batch, depth, vocab), torch.float32),
        (draft_probs, (batch, depth, vocab), torch.float32),
    )
    for tensor, shape, dtype in specs:
        if (tensor.shape != shape or tensor.dtype != dtype
                or tensor.device != candidates.device or not tensor.is_cuda
                or not tensor.is_contiguous()):
            raise ValueError("Tree sampling requires contiguous same-device tensors matching the upstream ABI")
    if not batch:
        return
    rows = torch.empty(batch, device=candidates.device, dtype=torch.int64)
    slots = torch.empty_like(rows)
    block = 1024
    tiles = triton.cdiv(vocab, block)
    partial = torch.empty((batch, tiles), device=candidates.device, dtype=torch.float64)
    _traverse[(batch,)](
        predicts, accept_index, accept_token_num, candidates, retrive_index,
        retrive_next_token, retrive_next_sibling, uniform_samples, target_probs,
        draft_probs, rows, slots, depth, steps, vocab,
        float(threshold_single), max(float(threshold_acc), 1e-9), num_warps=4,
    )
    _partial[(batch, tiles)](
        target_probs, draft_probs, rows, accept_token_num, partial,
        vocab, steps, tiles, block, num_warps=4,
    )
    _sample[(batch,)](
        target_probs, draft_probs, rows, slots, accept_token_num, partial,
        uniform_samples_for_final_sampling, predicts, vocab, steps, tiles,
        triton.next_power_of_2(tiles), block, num_warps=4,
    )
