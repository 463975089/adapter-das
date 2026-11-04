# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Bounded-memory FP32 varlen attention for deepinfer's ViT call contract."""

import math

import torch


def flash_attn_fwd(q, k, v, *, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None,
                   max_seqlen_k=None, softmax_scale=None, causal=False,
                   window_size_left=-1, window_size_right=-1, num_splits=1,
                   pack_gqa=None, return_lse=False, **kwargs):
    if kwargs or causal or window_size_left != -1 or window_size_right != -1:
        raise NotImplementedError("HCU ViT attention implements the pinned noncausal, full-window call")
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or q.shape[1:] != k.shape[1:]:
        raise ValueError("ViT attention requires matching [tokens, heads, head_dim] Q/K/V")
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("ViT Q/K/V must share a GPU")
    nq, heads, dim = q.shape
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.tensor([0, nq], dtype=torch.int32, device=q.device)
    if cu_seqlens_k is None:
        cu_seqlens_k = torch.tensor([0, k.shape[0]], dtype=torch.int32, device=q.device)
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("Q/K sequence counts differ")
    max_q = nq if max_seqlen_q is None else max_seqlen_q
    max_k = k.shape[0] if max_seqlen_k is None else max_seqlen_k
    if not nq or not k.shape[0]:
        out = torch.zeros_like(q)
        lse = torch.full((heads, nq), -float('inf'), dtype=torch.float32, device=q.device)
        return out, lse if return_lse else None
    if max_q <= 0 or max_k <= 0:
        raise ValueError("Positive ViT maximum lengths are required")
    scale = 1. / math.sqrt(dim) if softmax_scale is None else softmax_scale
    # Padding destinations live in separate storage, so no padded row can
    # overwrite a real output. Sequence boundaries stay as device tensors.
    storage = torch.empty((nq + max_q, heads, dim), device=q.device, dtype=q.dtype)
    lse_storage = torch.empty((nq + max_q, heads), device=q.device, dtype=torch.float32)
    keys = torch.arange(max_k, device=q.device)
    for sequence in range(cu_seqlens_q.numel() - 1):
        q_start, k_start = cu_seqlens_q[sequence], cu_seqlens_k[sequence]
        q_len = cu_seqlens_q[sequence + 1] - q_start
        k_len = cu_seqlens_k[sequence + 1] - k_start
        key_rows = (k_start + keys).clamp(0, k.shape[0] - 1).long()
        key = k.index_select(0, key_rows).permute(1, 2, 0).float()
        value = v.index_select(0, key_rows).permute(1, 0, 2).float()
        for begin in range(0, max_q, 64):
            positions = torch.arange(begin, min(begin + 64, max_q), device=q.device)
            rows = q_start + positions
            query = q.index_select(0, rows.clamp(0, nq - 1).long()).permute(1, 0, 2).float()
            logits = (query @ key) * scale
            logits.masked_fill_((keys >= k_len)[None, None, :], -float('inf'))
            probs = logits.softmax(dim=-1)
            probs = torch.where(k_len > 0, probs, 0.)
            output = (probs @ value).permute(1, 0, 2).to(q.dtype)
            destination = torch.where(positions < q_len, rows, nq + positions).long()
            storage.index_copy_(0, destination, output)
            if return_lse:
                lse_storage.index_copy_(0, destination, logits.logsumexp(dim=-1).T)
    return storage[:nq], lse_storage[:nq].T if return_lse else None
