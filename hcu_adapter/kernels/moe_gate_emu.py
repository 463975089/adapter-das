# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MoE routing gate (sqrtsoftplus top-k) per the pinned TileKernels / DeepGEMM contracts.

Anchors: tile_kernels 2.1.0 torch/topk.py::moe_topk_gate_forward (reference) and
DeepGEMM csrc/apis/mega_gate.hpp (bf16_mega_gate = BF16 gate GEMM fused with the
same routing). Graph-safe rewrite of the reference: masks instead of nonzero().
force_random rows (DEEPINFER_FORCE_LOAD_BALANCE benchmarking) are routed normally;
the bridge refuses to install when that upstream switch is on.
"""

import torch
import torch.nn.functional as F


def moe_topk_gate(logits, num_topk, use_shared_as_routed, num_shared_experts, routed_scaling_factor, ep_rank,
                  scoring_func="sqrtsoftplus", mask=None, bias=None, image_bias=None, image_token_mask=None,
                  fix_routing_mask=None, to_physical_map=None, logical_count=None, unmapped_topk_idx=None,
                  force_random=None, out=None, scores=None):
    if scoring_func != "sqrtsoftplus":
        raise ValueError("moe_topk_gate supports sqrtsoftplus scoring only")
    if scores is not None:
        raise NotImplementedError("HCU moe_topk_gate does not emit the training scores output")
    tokens, routed = logits.shape
    shared = num_shared_experts if use_shared_as_routed else 0
    physical_topk = num_topk + shared
    device = logits.device
    if out is None:
        topk_idx = torch.empty((tokens, physical_topk), dtype=torch.int64, device=device)
        topk_weights = torch.empty((tokens, physical_topk), dtype=torch.float32, device=device)
    else:
        topk_idx, topk_weights = out
    if tokens == 0:
        return topk_idx, topk_weights
    active = mask if mask is not None else torch.ones(tokens, dtype=torch.bool, device=device)
    score = F.softplus(logits.float()).sqrt()
    rank_bias = torch.zeros(routed, dtype=torch.float32, device=device) if bias is None else bias.float()
    ranking = score + rank_bias[None, :]
    if image_token_mask is not None:
        ranking = torch.where(image_token_mask[:, None], score + image_bias.float()[None, :], ranking)
    _, order = torch.sort(ranking, dim=1, descending=True, stable=True)
    selected = order[:, :num_topk]
    if fix_routing_mask is not None and unmapped_topk_idx is not None:
        selected = torch.where(fix_routing_mask[:, None], unmapped_topk_idx.long(), selected)
    picked = score.gather(1, selected.clamp(min=0))
    weights = picked / (picked.sum(dim=-1, keepdim=True) + 1e-20) * routed_scaling_factor
    if unmapped_topk_idx is not None:
        unmapped_topk_idx.copy_(torch.where(active[:, None], selected, -1))
    if shared:
        shared_idx = torch.arange(routed, routed + shared, dtype=torch.int64, device=device)
        selected = torch.cat((selected, shared_idx.expand(tokens, -1)), dim=1)
        weights = torch.cat((weights, torch.ones((tokens, shared), dtype=torch.float32, device=device)), dim=1)
    if to_physical_map is not None:
        global_idx = torch.arange(tokens, dtype=torch.int64, device=device)[:, None]
        logical = selected.clamp(min=0)
        dup = torch.remainder(ep_rank + global_idx * 23333, logical_count.long()[logical])
        selected = to_physical_map.long()[logical, dup]
    topk_idx.copy_(torch.where(active[:, None], selected, -1))
    topk_weights.copy_(torch.where(active[:, None], weights, 0.0))
    return topk_idx, topk_weights


def bf16_mega_gate(x, weight, num_topk, use_shared_as_routed, num_shared_experts, routed_scaling_factor, ep_rank,
                   scoring_func="sqrtsoftplus", mask=None, bias=None, image_bias=None, image_token_mask=None,
                   fix_routing_mask=None, to_physical_map=None, logical_count=None, unmapped_topk_idx=None,
                   force_random=None, out=None):
    """BF16 gate GEMM (FP32 accumulate) fused with moe_topk_gate."""
    logits = x.float() @ weight.float().T
    return moe_topk_gate(logits, num_topk, use_shared_as_routed, num_shared_experts, routed_scaling_factor, ep_rank,
                         scoring_func, mask, bias, image_bias, image_token_mask, fix_routing_mask, to_physical_map,
                         logical_count, unmapped_topk_idx, force_random, out)
