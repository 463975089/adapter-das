"""HCU draft router, matching SGLang's V4.1 Triton score/selection path.

Reference: sglang 7b91060ae3ef5e112048988da6f4f673cf4b6a89,
python/sglang/kernels/ops/moe/moe_fused_gate.py::_router_triton_kernel.
LightOp v41 only supports 256/384 experts. Keep the 128/top3 draft on
SGLang's log1p, NaN-first, lowest-id-tie path; DeepInfer metadata is applied
by moe_gate_lightop after this score-only kernel. No framework dependency.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _draft_gate(logits, bias, weights, indices, M,
                STRIDE_M: tl.constexpr, STRIDE_N: tl.constexpr,
                SCALE: tl.constexpr):
    rows = tl.program_id(0) * 2 + tl.arange(0, 2)
    experts = tl.arange(0, 128)
    values = tl.load(logits + rows[:, None] * STRIDE_M +
                     experts[None, :] * STRIDE_N, rows[:, None] < M, other=0).to(tl.float32)
    correction = tl.load(bias + experts).to(tl.float32)
    scores = libdevice.sqrt(tl.where(values > 20., values,
                                    libdevice.log1p(libdevice.exp(values))))
    ranked = scores + correction[None, :]
    ranked = tl.where(ranked == ranked, ranked, float('inf'))
    remaining = tl.full((2, 128), True, tl.int1)
    slots = tl.arange(0, 4)
    selected_values = tl.zeros((2, 4), tl.float32)
    selected_ids = tl.zeros((2, 4), tl.int32)
    for k in tl.static_range(3):
        maximum = tl.max(ranked, 1)[:, None]
        winner = tl.min(tl.where(remaining & (ranked == maximum),
                                experts[None, :], 129), 1)[:, None]
        value = tl.sum(tl.where(experts[None, :] == winner, scores, 0.), 1)[:, None]
        selected_values = tl.where(slots[None, :] == k, value, selected_values)
        selected_ids = tl.where(slots[None, :] == k, winner, selected_ids)
        remaining = remaining & (experts[None, :] != winner)
        ranked = tl.where(remaining, ranked, -float('inf'))
    total = tl.sum(selected_values, 1)[:, None]
    # Same zero-sum guard as SGLang's non-grouped V4.1 route.
    selected_values = selected_values / tl.where(total > 0., total, 1.) * SCALE
    mask = (rows[:, None] < M) & (slots[None, :] < 3)
    offsets = rows[:, None] * 3 + slots[None, :]
    tl.store(weights + offsets, selected_values, mask)
    tl.store(indices + offsets, selected_ids, mask)


def sqrtsoftplus_topk(logits, bias, topk, scale):
    if logits.ndim != 2 or logits.shape[1] != 128 or topk != 3:
        raise NotImplementedError('HCU draft router currently requires 128 experts/top3')
    if bias.shape != (128,) or not bias.is_contiguous():
        raise ValueError('Draft router requires contiguous bias[128]')
    m = logits.shape[0]
    weights = torch.empty((m, 3), dtype=torch.float32, device=logits.device)
    indices = torch.empty((m, 3), dtype=torch.int64, device=logits.device)
    if m:
        _draft_gate[(triton.cdiv(m, 2),)](
            logits, bias, weights, indices, m, logits.stride(0), logits.stride(1),
            float(scale), num_warps=1)
    return weights, indices
