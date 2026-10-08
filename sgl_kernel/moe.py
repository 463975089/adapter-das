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
    if ignore_invalid_expert:
        valid = (flat >= 0) & (flat < num_experts)
    elif numel:
        flat_min = int(flat.min().item())
        flat_max = int(flat.max().item())
        if flat_min < 0 or flat_max >= num_experts:
            raise ValueError(
                "topk_ids contains expert ids outside [0, num_experts); "
                "pass ignore_invalid_expert=True for EP sentinel lanes"
            )
        valid = torch.ones_like(flat, dtype=torch.bool)
    else:
        valid = torch.zeros_like(flat, dtype=torch.bool)

    sorted_token_ids.fill_(numel)

    flat_v = flat[valid]
    counts = torch.bincount(flat_v, minlength=num_experts)
    block_counts = (counts + block_size - 1) // block_size * block_size
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    cumsum[1:] = block_counts.cumsum(0).to(torch.int32)

    if cumsum_buffer.shape[0] >= num_experts + 1:
        cumsum_buffer[: num_experts + 1].copy_(cumsum)
    else:
        cumsum_buffer.copy_(cumsum[: cumsum_buffer.shape[0]])
    total = int(cumsum[num_experts].item())
    num_tokens_post_pad.fill_(total)

    max_blocks = experts_ids.shape[0]
    if numel == 0 or max_blocks == 0:
        experts_ids.fill_(num_experts)
        return

    # Stable sort by expert id keeps the CUDA kernel's per-expert FIFO fill
    # order (tokens of one expert keep their original flat order). `order`
    # indexes the compacted valid array; map back to original flat positions.
    order = torch.argsort(flat_v, stable=True)
    orig_pos = valid.nonzero(as_tuple=False).squeeze(1)  # (num_valid,)
    e_sorted = flat_v[order]
    starts = torch.cumsum(counts, 0).to(torch.long) - counts.to(torch.long)
    rank = torch.arange(order.numel(), device=order.device) - starts[e_sorted]
    dest = cumsum[e_sorted].to(torch.long) + rank
    sorted_token_ids[dest] = orig_pos[order].to(torch.int32)

    n_blocks = min(total // block_size, max_blocks)
    block_starts = torch.arange(n_blocks, device=topk_ids.device) * block_size
    expert_of_block = torch.searchsorted(
        cumsum[1:].to(torch.long), block_starts, right=True
    ).to(torch.int32)
    experts_ids[:n_blocks].copy_(expert_of_block)
    if max_blocks > n_blocks:
        experts_ids[n_blocks:].fill_(num_experts)


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
    With ignore_invalid_expert=True, ids outside [0, num_experts) (e.g. the EP
    sentinel -1) are skipped entirely.

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
    capturing = torch.cuda.is_current_stream_capturing()
    if not ignore_invalid_expert and numel and not capturing:
        flat_min = int(flat.min().item())
        flat_max = int(flat.max().item())
        if flat_min < 0 or flat_max >= num_experts:
            raise ValueError(
                "topk_ids contains expert ids outside [0, num_experts); "
                "pass ignore_invalid_expert=True for EP sentinel lanes"
            )

    sorted_token_ids.fill_(numel)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    cursor = torch.empty(num_experts, dtype=torch.int32, device=topk_ids.device)

    block_hist = 1024
    if numel:
        grid = (triton.cdiv(numel, block_hist),)
        _moe_align_hist_kernel[grid](
            flat, numel, counts, num_experts, ignore_invalid_expert, BLOCK=block_hist
        )
    block_e = max(16, next_power_of_2(num_experts))
    write_extra = 1 if cumsum_buffer.shape[0] > num_experts + 1 else 0
    _moe_align_scan_kernel[(1,)](
        counts,
        cumsum_buffer,
        cursor,
        num_tokens_post_pad,
        num_experts,
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
            num_experts,
            ignore_invalid_expert,
            BLOCK=block_hist,
        )
    if experts_ids.shape[0]:
        _moe_align_block_expert_kernel[(experts_ids.shape[0],)](
            cumsum_buffer,
            experts_ids,
            num_tokens_post_pad,
            num_experts,
            block_size,
            BLOCK_E=block_e,
        )


if HAS_TRITON:

    @triton.jit
    def _moe_align_hist_kernel(
        ids_ptr,
        numel,
        counts_ptr,
        num_experts,
        IGNORE_INVALID: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        inb = offs < numel
        e = tl.load(ids_ptr + offs, mask=inb, other=-1)
        if IGNORE_INVALID:
            ok = inb & (e >= 0) & (e < num_experts)
        else:
            ok = inb
        tl.atomic_add(counts_ptr + e, 1, mask=ok)

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
        num_experts,
        IGNORE_INVALID: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        inb = offs < numel
        e = tl.load(ids_ptr + offs, mask=inb, other=-1)
        if IGNORE_INVALID:
            ok = inb & (e >= 0) & (e < num_experts)
        else:
            ok = inb
        pos = tl.atomic_add(cursor_ptr + e, 1, mask=ok)
        tl.store(sorted_ptr + pos, offs.to(tl.int32), mask=ok)

    @triton.jit
    def _moe_align_block_expert_kernel(
        cumsum_ptr,
        experts_ptr,
        post_pad_ptr,
        num_experts,
        block_size,
        BLOCK_E: tl.constexpr,
    ):
        b = tl.program_id(0)
        total = tl.load(post_pad_ptr)
        padded = (total + block_size - 1) // block_size
        if b < padded:
            s = b * block_size
            offs = tl.arange(0, BLOCK_E)
            em = offs < num_experts
            start = tl.load(cumsum_ptr + offs, mask=em, other=1 << 30)
            end = tl.load(cumsum_ptr + offs + 1, mask=em, other=-1)
            # exactly one non-empty expert's [start, end) contains s; pick it
            hit = ((start <= s) & (s < end)).to(tl.int32) * (offs + 1)
            tl.store(experts_ptr + b, tl.max(hit, 0).to(tl.int32) - 1)
        else:
            tl.store(experts_ptr + b, num_experts)


def moe_sum(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    """output = sum of input over dim=1 (input [M, topk, D] -> output [M, D])."""
    output_tensor.copy_(input_tensor.sum(dim=1))


def moe_sum_reduce(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    routed_scaling_factor: float = 0,
):
    """output = sum(input, dim=1), scaled by routed_scaling_factor when != 0."""
    out = input_tensor.sum(dim=1)
    if routed_scaling_factor:
        out = out * routed_scaling_factor
    output_tensor.copy_(out.to(output_tensor.dtype))


__all__ = [
    "topk_softmax",
    "topk_sigmoid",
    "moe_fused_gate",
    "moe_align_block_size",
    "moe_sum",
    "moe_sum_reduce",
]
