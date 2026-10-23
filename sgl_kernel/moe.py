"""MoE routing and dispatch ops mirroring sgl_kernel's moe API.

v1 strategy: topk_softmax / topk_sigmoid / moe_fused_gate / moe_align_block_size /
moe_sum(_reduce) are implemented in batched fp32 torch ops with exactly the
semantics of the sgl-kernel CUDA kernels (same ranking, slot order, tie and
padding contracts). Device-portable (HCU/CUDA/CPU) so the triton fused_moe
pipeline runs end-to-end; hand-tuned triton ports land later.

Ranking/weight conventions (verified against sglang-das
`kernels/ops/moe/moe_fused_gate.py::_router_triton_kernel` and the AOT topk
kernels):
    sigmoid      : act = sigmoid(x);      rank = act + bias
    sqrtsoftplus : act = sqrt(softplus(x)); rank = act + bias
    softmax      : act = softmax(softcap(x)); rank = softcap(x) + bias
    weight of a picked expert = act[expert]  (always bias-free)
    shared slots : weight = routed_sum / routed_scaling_factor
    renormalize  : after shared fill, ALL slots /= routed routed_sum (0 -> 1)
    scale        : after renormalize, ALL slots *= routed_scaling_factor
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .compat import HAS_TRITON, check_cuda_tensor, next_power_of_2

if HAS_TRITON:
    import triton
    import triton.language as tl

_SCORING_FUNC_MAP = {"sigmoid": 0, "sqrtsoftplus": 1, "softmax": 2}

_RENORMALIZE_SUM_EPSILON = 1e-20


def _finish_gate_slots(
    activated: torch.Tensor,
    biased: torch.Tensor,
    topk: int,
    num_fused_shared_experts: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared tail of every gate op: routed top-k -> shared slots -> renorm ->
    scale (order mirrors _router_triton_kernel exactly)."""
    num_experts = biased.shape[1]
    k_routed = topk - max(0, num_fused_shared_experts)
    _, ids = torch.topk(biased, k=k_routed, dim=-1, sorted=False)
    weights = activated.gather(1, ids).to(torch.float32)

    routed_sum = weights.sum(dim=-1, keepdim=True, dtype=torch.float32)
    rsf = routed_scaling_factor if routed_scaling_factor else 1.0

    if num_fused_shared_experts > 0:
        shared_w = (routed_sum / rsf).expand(-1, num_fused_shared_experts).contiguous()
        weights = torch.cat([weights, shared_w], dim=1)
        shared_ids = (
            torch.arange(num_fused_shared_experts, device=ids.device, dtype=ids.dtype)
            .unsqueeze(0)
            .expand(ids.shape[0], -1)
            + num_experts
        )
        ids = torch.cat([ids, shared_ids], dim=1)

    if renormalize:
        norm = torch.where(routed_sum > 0.0, routed_sum, torch.ones_like(routed_sum))
        weights = weights / norm
    if routed_scaling_factor:
        weights = weights * routed_scaling_factor
    return weights, ids


def _write_topk_out(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> None:
    topk_weights.copy_(weights.to(topk_weights.dtype))
    topk_ids.copy_(ids.to(topk_ids.dtype))


# ---------------------------------------------------------------------------
# destination-passing topk gate (topk_softmax / topk_sigmoid)
# ---------------------------------------------------------------------------


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    """Fused softmax top-k MoE gate (destination-passing, in-place).

    topk_weights [num_tokens, topk] fp32 and topk_ids [num_tokens, topk] int32
    are written in place; gating_output is [num_tokens, num_experts].
    """
    logits = gating_output.float()
    if moe_softcapping:
        logits = moe_softcapping * torch.tanh(logits / moe_softcapping)
    activated = torch.softmax(logits, dim=-1, dtype=torch.float32)
    biased = logits if correction_bias is None else logits + correction_bias.float()
    weights, ids = _finish_gate_slots(
        activated,
        biased,
        topk_ids.shape[1],
        num_fused_shared_experts=0,
        renormalize=renormalize,
        routed_scaling_factor=0.0,
    )
    _write_topk_out(topk_weights, topk_ids, weights, ids)


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
    routed_scaling_factor: float = 1.0,
    num_fused_shared_experts: int = 0,
) -> None:
    """Fused sigmoid top-k MoE gate (destination-passing, in-place).

    Accepts both the 5-arg upstream form (no scale / shared experts) and the
    extended 7-arg form some call sites use (scale, num_fused_shared_experts).
    """
    if num_fused_shared_experts > 1:
        raise ValueError("sigmoid topk supports at most one fused shared expert")
    gating = gating_output.float()
    activated = torch.sigmoid(gating)
    biased = (
        activated if correction_bias is None else activated + correction_bias.float()
    )
    weights, ids = _finish_gate_slots(
        activated,
        biased,
        topk_ids.shape[1],
        num_fused_shared_experts=num_fused_shared_experts,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )
    _write_topk_out(topk_weights, topk_ids, weights, ids)


# ---------------------------------------------------------------------------
# moe_fused_gate (legacy positional and new keyword signatures)
# ---------------------------------------------------------------------------


def moe_fused_gate(scores: torch.Tensor, bias: Optional[torch.Tensor], *args, **kwargs):
    """Fused MoE router: scoring + optional bias + (grouped) top-k.

    Supports both upstream calling conventions:
      legacy : moe_fused_gate(input, bias, num_expert_group, topk_group, topk,
                               renormalize=, scoring_func=,
                               num_fused_shared_experts=,
                               routed_scaling_factor=,
                               apply_routed_scaling_factor_on_output=)
      new    : moe_fused_gate(scores, bias, topk, scoring_func=, ...,
                               num_expert_group=, topk_group=, moe_softcapping=)

    Returns (topk_weights fp32 [M, topk], topk_ids int32 [M, topk]).
    """
    legacy = len(args) >= 3 and all(isinstance(a, int) for a in args[:3])
    if legacy:
        num_expert_group, topk_group, topk = args[0], args[1], args[2]
        if len(args) > 3:
            raise TypeError(f"unexpected positional args: {args[3:]}")
        scoring_func = kwargs.pop("scoring_func", "sigmoid")
        num_shared = kwargs.pop("num_fused_shared_experts", 0)
        renormalize = kwargs.pop("renormalize", True)
        rsf = kwargs.pop("routed_scaling_factor", 1.0)
        apply_scale = kwargs.pop("apply_routed_scaling_factor_on_output", False)
        softcap = kwargs.pop("moe_softcapping", 0.0)
    else:
        if not args and "topk" not in kwargs:
            raise TypeError("moe_fused_gate requires topk")
        topk = args[0] if args else kwargs.pop("topk")
        scoring_func = kwargs.pop("scoring_func", "sigmoid")
        num_shared = kwargs.pop("num_fused_shared_experts", 0)
        renormalize = kwargs.pop("renormalize", True)
        rsf = kwargs.pop("routed_scaling_factor", 1.0)
        apply_scale = kwargs.pop("apply_routed_scaling_factor_on_output", False)
        softcap = kwargs.pop("moe_softcapping", 0.0)
        num_expert_group = kwargs.pop("num_expert_group", 1)
        topk_group = kwargs.pop("topk_group", 1)
    if kwargs:
        raise TypeError(f"unexpected kwargs: {list(kwargs)}")
    if rsf is None:
        rsf = 1.0

    key = str(scoring_func).lower()
    if key not in _SCORING_FUNC_MAP:
        raise ValueError(f"unknown scoring_func {scoring_func!r}")
    logits = scores.float()
    if key == "softmax":
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        biased = logits if bias is None else logits + bias.float()
        activated = torch.softmax(logits, dim=-1, dtype=torch.float32)
    else:
        if key == "sigmoid":
            activated = torch.sigmoid(logits)
        else:  # sqrtsoftplus: sqrt(log1p(exp(x))), overflow-guarded
            activated = torch.sqrt(
                torch.where(logits > 20.0, logits, torch.log1p(torch.exp(logits)))
            )
        biased = activated if bias is None else activated + bias.float()

    num_expert_group = num_expert_group or 1
    if num_expert_group > 1:
        m, n = biased.shape
        assert n % num_expert_group == 0, "num_experts % num_expert_group != 0"
        per_group = n // num_expert_group
        g = biased.view(m, num_expert_group, per_group)
        top2 = torch.topk(g, k=min(2, per_group), dim=-1).values
        group_score = top2.sum(dim=-1)
        kgrp = min(topk_group, num_expert_group)
        keep = torch.topk(group_score, k=kgrp, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_score, dtype=torch.bool)
        group_mask.scatter_(1, keep, True)
        expert_mask = (
            group_mask.unsqueeze(-1).expand(m, num_expert_group, per_group).reshape(m, n)
        )
        biased = biased.masked_fill(~expert_mask, float("-inf"))

    if num_shared > topk:
        raise ValueError("topk must be > num_fused_shared_experts")

    weights, ids = _finish_gate_slots(
        activated,
        biased,
        topk,
        num_fused_shared_experts=num_shared,
        renormalize=renormalize,
        routed_scaling_factor=rsf if apply_scale else 0.0,
    )
    return weights.to(torch.float32), ids.to(torch.int32)


# ---------------------------------------------------------------------------
# moe_align_block_size (torch port of the sgl-kernel CUDA contract)
# ---------------------------------------------------------------------------


def _moe_align_block_size_torch(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    cumsum_buffer,
    ignore_invalid_expert,
):
    numel = topk_ids.numel()
    flat = topk_ids.flatten().to(torch.long)
    # Upstream AOT semantics (the contract the fused kernel + filter_expert
    # rely on): sentinel/invalid lanes (-1 on EP ranks) are counted into a
    # dedicated LAST bucket, occupy a block-aligned segment in the total,
    # and the blocks owned by that bucket are flagged expert id -1 so the
    # fused kernel's filter_expert branch ZEROES those output rows.
    # Dropping the lanes (prior behavior) left cache slots unwritten.
    nb = num_experts + 1  # bucket count: real experts 0..E-1 + sentinel bucket E
    # mirror the triton hist kernel exactly: valid experts keep their id,
    # the sentinel (-1) and any stray out-of-range value fall into bucket E
    # (the +1-offset convention's sentinel slot). clamp(0, E) is WRONG here:
    # it folds -1 into expert 0's bucket (caught by rerun3, 2026-10-18).
    bucket = torch.where(
        (flat >= 0) & (flat < num_experts),
        flat,
        torch.full_like(flat, num_experts),
    )
    sorted_token_ids.fill_(numel)

    counts = torch.bincount(bucket, minlength=nb)
    block_counts = (counts + block_size - 1) // block_size * block_size
    cumsum = torch.zeros(nb + 1, dtype=torch.int32, device=topk_ids.device)
    cumsum[1:] = block_counts.cumsum(0).to(torch.int32)

    if cumsum_buffer.shape[0] >= nb:
        cumsum_buffer[:nb].copy_(cumsum[:nb])
    else:
        cumsum_buffer.copy_(cumsum[: cumsum_buffer.shape[0]])
    total = int(cumsum[nb].item())
    num_tokens_post_pad.fill_(total)

    max_blocks = experts_ids.shape[0]
    if numel == 0 or max_blocks == 0:
        experts_ids.fill_(nb)  # tail marker = num_buckets, same as triton
        return

    # Stable sort by expert id keeps the CUDA kernel's per-expert FIFO fill
    # order (tokens of one expert keep their original flat order).
    order = torch.argsort(bucket, stable=True)
    e_sorted = bucket[order]
    starts = torch.cumsum(counts, 0).to(torch.long) - counts.to(torch.long)
    rank = torch.arange(order.numel(), device=order.device) - starts[e_sorted]
    dest = cumsum[e_sorted].to(torch.long) + rank
    sorted_token_ids[dest] = order.to(torch.int32)  # order IS the flat pos now

    n_blocks = min(total // block_size, max_blocks)
    block_starts = torch.arange(n_blocks, device=topk_ids.device) * block_size
    expert_of_block = torch.searchsorted(
        cumsum[1:].to(torch.long), block_starts, right=True
    ).to(torch.int32)
    # blocks owned by the sentinel bucket (bucket index num_experts) become -1
    expert_of_block[expert_of_block == num_experts] = -1
    experts_ids[:n_blocks].copy_(expert_of_block)
    if max_blocks > n_blocks:
        experts_ids[n_blocks:].fill_(nb)


def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    cumsum_buffer,
    pad_sorted_token_ids=False,
    ignore_invalid_expert=False,
):
    """Group flat token ids by expert, padding every expert's slot count up to
    a multiple of ``block_size`` (in place on all output buffers).

    Contracts (matching the CUDA kernel consumed by the triton fused_moe):
      - padding value in sorted_token_ids = topk_ids.numel() (out-of-range token id)
      - experts_ids[b] = expert of block b for b < ceil(total/block); tail = num_experts
      - num_tokens_post_pad (1,) int32 = total padded length
      - cumsum_buffer[i] = exclusive prefix sum of block-aligned per-expert counts
    Upstream semantics for invalid ids (the EP sentinel -1, matching the AOT
    kernel's +1-offset convention that the runner relies on): they are
    counted into a dedicated LAST bucket, occupy a block-aligned segment of
    the total, and the blocks owned by that bucket carry expert id -1 — the
    fused kernel's filter_expert branch zeroes those output rows. Dropping
    the lanes instead leaves cache slots unwritten (garbage combine).
    ignore_invalid_expert is accepted for call-compatibility; both values
    take this AOT-compatible path.

    Runs on GPU via fully device-side triton kernels: the CUDA-graph capture
    path of the fused_moe pipeline calls this every step, so host syncs
    (`.item()`), dynamic-shape ops (`nonzero`, boolean-mask indexing) and
    data-dependent raises are forbidden there. The triton path keeps every
    scalar contract bit-exact; within one expert's segment the token order is
    arrival-scheduled (atomic cursors) rather than strictly FIFO, which does
    not change the fused-moe result (rows are independent per token). The
    eager/validating range check only runs off capture, like the AOT kernel
    which trusts its inputs. CPU falls back to the exact torch port above.
    """
    if not (HAS_TRITON and topk_ids.is_cuda and num_experts <= 16384):
        return _moe_align_block_size_torch(
            topk_ids,
            num_experts,
            block_size,
            sorted_token_ids,
            experts_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            ignore_invalid_expert,
        )
    check_cuda_tensor(sorted_token_ids, experts_ids, num_tokens_post_pad, cumsum_buffer)

    flat = topk_ids.reshape(-1)
    if flat.dtype not in (torch.int32, torch.int64):
        flat = flat.to(torch.int64)
    numel = flat.numel()
    # Upstream AOT semantics (matches the runner's +1-offset call and the
    # fused kernel's filter_expert zero-output branch): sentinel lanes (-1
    # on EP ranks) are counted into a dedicated LAST bucket, occupy a
    # block-aligned segment of the total, and blocks owned by that bucket
    # are flagged expert id -1 so filter_expert ZEROES those rows.
    # Dropping the lanes left intermediate_cache slots unwritten (garbage
    # combine). NB_ = number of buckets = real experts + sentinel bucket.
    nb = num_experts + 1
    block_e = max(16, next_power_of_2(nb))
    write_extra = 1 if cumsum_buffer.shape[0] > nb else 0

    # Small shapes (every decode step: ~96-576 lanes) go through ONE
    # single-CTA kernel instead of the 6-launch quartet (2 aten setup
    # fills + hist + scan + fill + block_expert) — same outputs bit-exact
    # except intra-segment token order, which the runner ignores.
    fused_max = 2048
    try:
        from ._adapters.hcu_env import resolve

        _mode, _src = resolve("HCU_MOE_ALIGN_FUSED")
        if _mode in ("0", "off", "false"):
            fused_max = 0
        elif isinstance(_mode, str) and _mode.isdigit():
            fused_max = int(_mode)
    except Exception:
        pass
    if fused_max and numel <= fused_max:
        _moe_align_fused_small_kernel[(1,)](
            flat,
            numel,
            sorted_token_ids,
            sorted_token_ids.shape[0],
            experts_ids,
            experts_ids.shape[0],
            num_tokens_post_pad,
            cumsum_buffer,
            nb,
            block_size,
            NBLK_E=block_e,
            WRITE_EXTRA=write_extra,
            TILE_N=128,
            TILE_E=64,
        )
        return

    sorted_token_ids.fill_(numel)
    counts = torch.zeros(nb, dtype=torch.int32, device=topk_ids.device)
    cursor = torch.empty(nb, dtype=torch.int32, device=topk_ids.device)

    block_hist = 1024
    if numel:
        grid = (triton.cdiv(numel, block_hist),)
        _moe_align_hist_kernel[grid](
            flat, numel, counts, nb, BLOCK=block_hist
        )
    _moe_align_scan_kernel[(1,)](
        counts,
        cumsum_buffer,
        cursor,
        num_tokens_post_pad,
        nb,
        block_size,
        write_extra,
        BLOCK_E=block_e,
    )
    if numel:
        _moe_align_fill_kernel[grid](
            flat,
            numel,
            cursor,
            sorted_token_ids,
            nb,
            BLOCK=block_hist,
        )
    if experts_ids.shape[0]:
        _moe_align_block_expert_kernel[(experts_ids.shape[0],)](
            cumsum_buffer,
            experts_ids,
            num_tokens_post_pad,
            nb,
            block_size,
            BLOCK_E=block_e,
        )


if HAS_TRITON:

    @triton.jit
    def _moe_align_hist_kernel(
        ids_ptr,
        numel,
        counts_ptr,
        num_buckets,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        inb = offs < numel
        e = tl.load(ids_ptr + offs, mask=inb, other=-1)
        # AOT-compatible crash-free mapping: valid experts keep their id,
        # sentinel (-1) and any stray out-of-range value fall into the LAST
        # bucket (the +1-offset convention's sentinel slot).
        b = tl.where((e >= 0) & (e < num_buckets - 1), e, num_buckets - 1)
        tl.atomic_add(counts_ptr + b, 1, mask=inb)

    @triton.jit
    def _moe_align_scan_kernel(
        counts_ptr,
        cumsum_ptr,
        cursor_ptr,
        post_pad_ptr,
        num_experts,
        block_size,
        WRITE_EXTRA: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK_E)
        m = offs < num_experts
        counts = tl.load(counts_ptr + offs, mask=m, other=0)
        aligned = ((counts + block_size - 1) // block_size) * block_size
        incl = tl.cumsum(aligned, 0)
        excl = incl - aligned
        total = tl.sum(aligned, 0)
        tl.store(cumsum_ptr + offs, excl, mask=m)
        tl.store(cumsum_ptr + num_experts, total)
        if WRITE_EXTRA:
            # callers size this (E+2,) buffer; define the trailing slot too
            tl.store(cumsum_ptr + num_experts + 1, total)
        tl.store(post_pad_ptr, total)
        tl.store(cursor_ptr + offs, excl, mask=m)

    @triton.jit
    def _moe_align_fill_kernel(
        ids_ptr,
        numel,
        cursor_ptr,
        sorted_ptr,
        num_buckets,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        inb = offs < numel
        e = tl.load(ids_ptr + offs, mask=inb, other=-1)
        # same bucket mapping as hist: sentinels write their token id into
        # the sentinel segment (the block gets expert -1 -> filter zeroes
        # the output; the row itself stays inert)
        b = tl.where((e >= 0) & (e < num_buckets - 1), e, num_buckets - 1)
        pos = tl.atomic_add(cursor_ptr + b, 1, mask=inb)
        tl.store(sorted_ptr + pos, offs.to(tl.int32), mask=inb)

    @triton.jit
    def _moe_align_block_expert_kernel(
        cumsum_ptr,
        experts_ptr,
        post_pad_ptr,
        num_buckets,
        block_size,
        BLOCK_E: tl.constexpr,
    ):
        b = tl.program_id(0)
        total = tl.load(post_pad_ptr)
        padded = (total + block_size - 1) // block_size
        if b < padded:
            s = b * block_size
            offs = tl.arange(0, BLOCK_E)
            # bucket num_buckets-1 is the sentinel bucket: blocks from it
            # carry expert id -1 (the fused kernel's filter_expert zero
            # branch); tail blocks beyond total carry num_buckets as before.
            em = offs < num_buckets
            start = tl.load(cumsum_ptr + offs, mask=em, other=1 << 30)
            end = tl.load(cumsum_ptr + offs + 1, mask=em, other=-1)
            # exactly one non-empty expert's [start, end) contains s; pick it
            hit = ((start <= s) & (s < end)).to(tl.int32) * (offs + 1)
            eid = tl.max(hit, 0).to(tl.int32) - 1
            sentinel = eid == num_buckets - 1
            tl.store(experts_ptr + b, -1 if sentinel else eid)
        else:
            tl.store(experts_ptr + b, num_buckets)

    @triton.jit
    def _moe_align_fused_small_kernel(
        ids_ptr,
        numel,
        sorted_ptr,
        sorted_cap,
        experts_ptr,
        exp_cap,
        post_pad_ptr,
        cumsum_ptr,
        nb,
        block_size,
        NBLK_E: tl.constexpr,
        WRITE_EXTRA: tl.constexpr,
        TILE_N: tl.constexpr,
        TILE_E: tl.constexpr,
    ):
        # Single-CTA replacement for the hist/scan/fill/block_expert
        # quartet (+ the two aten setup fills) at decode shapes: numel is
        # at most a couple thousand lanes, so one program can hold the
        # whole histogram in registers. Bucket mapping, scan math and the
        # sentinel/tail conventions are bit-identical to the quartet; the
        # only divergence is intra-segment token order (FIFO here,
        # atomic-arrival there), which the fused_moe runner ignores.
        e_offs = tl.arange(0, NBLK_E)
        em = e_offs < nb

        counts = tl.zeros((NBLK_E,), tl.int32)
        for t0 in range(0, numel, TILE_N):
            offs = t0 + tl.arange(0, TILE_N)
            inb = offs < numel
            e = tl.load(ids_ptr + offs, mask=inb, other=-1)
            b = tl.where(inb & (e >= 0) & (e < nb - 1), e, nb - 1)
            counts += tl.sum(
                ((b[:, None] == e_offs[None, :]) & inb[:, None]).to(tl.int32),
                axis=0,
            )

        aligned = tl.where(em, ((counts + block_size - 1) // block_size) * block_size, 0)
        incl = tl.cumsum(aligned, 0)
        excl = incl - aligned
        total = tl.sum(aligned, 0)
        tl.store(cumsum_ptr + e_offs, excl, mask=em)
        tl.store(cumsum_ptr + nb, total)
        if WRITE_EXTRA:
            tl.store(cumsum_ptr + nb + 1, total)
        tl.store(post_pad_ptr, total)

        # fill the whole sorted buffer with the pad value (numel), then
        # scatter the lanes into the first counts[b] slots of each
        # block-aligned segment — disjoint, so ordering is free.
        pad = numel + tl.zeros((TILE_N,), tl.int32)
        for s0 in range(0, sorted_cap, TILE_N):
            so = s0 + tl.arange(0, TILE_N)
            tl.store(sorted_ptr + so, pad, mask=so < sorted_cap)

        carry = tl.zeros((NBLK_E,), tl.int32)
        for t0 in range(0, numel, TILE_N):
            offs = t0 + tl.arange(0, TILE_N)
            inb = offs < numel
            e = tl.load(ids_ptr + offs, mask=inb, other=-1)
            b = tl.where(inb & (e >= 0) & (e < nb - 1), e, nb - 1)
            # write position = segment start + filled-so-far (carry) +
            # in-tile rank among same-bucket lanes with a smaller index
            earlier = (offs[None, :] < offs[:, None]) & (
                b[None, :] == b[:, None]
            ) & inb[None, :]
            rank_in = tl.sum(earlier.to(tl.int32), axis=1)
            seg = tl.sum(
                tl.where(b[:, None] == e_offs[None, :], excl[None, :], 0), axis=1
            )
            car = tl.sum(
                tl.where(b[:, None] == e_offs[None, :], carry[None, :], 0), axis=1
            )
            pos = seg + car + rank_in
            tl.store(sorted_ptr + pos, offs.to(tl.int32), mask=inb)
            carry += tl.sum(
                ((b[:, None] == e_offs[None, :]) & inb[:, None]).to(tl.int32),
                axis=0,
            )

        # block -> expert map: one owner per block start in [0, total);
        # sentinel-bucket blocks carry -1, tail blocks carry nb
        padded = (total + block_size - 1) // block_size
        for b0 in range(0, exp_cap, TILE_E):
            bi = b0 + tl.arange(0, TILE_E)
            s = bi * block_size
            owns = (excl[None, :] <= s[:, None]) & (
                s[:, None] < (excl + aligned)[None, :]
            ) & em[None, :]
            hit = tl.max(tl.where(owns, (e_offs + 1)[None, :], 0), axis=1)
            eid = hit - 1
            val = tl.where(bi < padded, tl.where(eid == nb - 1, -1, eid), nb)
            tl.store(experts_ptr + bi, val, mask=bi < exp_cap)


def moe_sum(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    """output = sum of input over dim=1 (input [M, topk, D] -> output [M, D])."""
    if not _moe_sum_scale_triton(input_tensor, output_tensor, 0.0):
        output_tensor.copy_(input_tensor.sum(dim=1))


def moe_sum_reduce(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    routed_scaling_factor: float = 0,
):
    """output = sum(input, dim=1), scaled by routed_scaling_factor when != 0."""
    if not _moe_sum_scale_triton(input_tensor, output_tensor, routed_scaling_factor):
        out = input_tensor.sum(dim=1)
        if routed_scaling_factor:
            out = out * routed_scaling_factor
        output_tensor.copy_(out.to(output_tensor.dtype))


if HAS_TRITON:

    @triton.jit
    def _moe_sum_scale_kernel(
        x_ptr,
        y_ptr,
        rsf,
        K,
        D,
        stride_m,
        BLOCK_D: tl.constexpr,
    ):
        # out[m, d] = rsf * sum_k x[m, k, d]  (fp32 accumulate, one store)
        m = tl.program_id(0)
        d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        base = x_ptr + m.to(tl.int64) * stride_m
        for _ in range(K):
            acc += tl.load(base + d, mask=d < D, other=0.0).to(tl.float32)
            base += D
        if rsf != 0.0:
            acc = acc * rsf
        tl.store(
            y_ptr + m.to(tl.int64) * D + d,
            acc.to(y_ptr.dtype.element_ty),
            mask=d < D,
        )


def _moe_sum_scale_triton(x: torch.Tensor, y: torch.Tensor, rsf: float) -> bool:
    """Launch the fused combine; False means caller must run the torch fallback.

    Replaces the aten reduce+mul+copy triple (the in-graph combine was
    reduce_kernel<128,4> at 135-165 GB/s plus two extra elementwise passes).
    """
    if not (
        HAS_TRITON
        and x.is_cuda
        and x.dim() == 3
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.is_contiguous()
        and y.is_contiguous()
        and y.shape[0] == x.shape[0]
        and y.shape[1] == x.shape[2]
        and x.shape[1] <= 64
    ):
        return False
    M, K, D = x.shape
    BLOCK_D = 512
    grid = (M, triton.cdiv(D, BLOCK_D))
    _moe_sum_scale_kernel[grid](
        x,
        y,
        float(rsf) if rsf else 0.0,
        K,
        D,
        x.stride(0),
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return True


__all__ = [
    "topk_softmax",
    "topk_sigmoid",
    "moe_fused_gate",
    "moe_align_block_size",
    "moe_sum",
    "moe_sum_reduce",
]
