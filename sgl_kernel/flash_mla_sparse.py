# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Sparse FlashMLA forward (torch emulation) for the DSA backends.

`dsa_backend._forward_flashmla_sparse` (decode/extend sparse attention over
the indexer's top-k selectors) and the DSV4 backend both import
``flash_mla_sparse_fwd`` from this package. On HCU there is no sparse
FlashMLA wheel — same story as the dense emu above — so this supplies the
API surface:

Contract (flashmla v3 sparse):
- q:        (s_q, H, d_qk) bf16 — one query row per token
- kv:       (pages * page_size, h_kv=1, d_qk) bf16 paged (compressed) pool
- indices:  (s_q, h_kv=1, topk) int32 — token ids INTO the flat kv axis;
            tail padding is -1 and masked out
- sm_scale, d_v (v width < d_qk: the rope channels score but no value),
  attn_sink (per-head zero-valued virtual-key logit), topk_length
            (per-row valid width; rows shorter than topk mask the tail)
- returns   (o [s_q, H, d_v], lse [s_q, H], ?) — third slot ignored upstream

Math is the emu contract everywhere else in this repo: fp32
gather + scores + softmax + p@v, cast back to q dtype.
"""

import torch

__all__ = ["flash_mla_sparse_fwd"]


def flash_mla_sparse_fwd(
    q,
    kv,
    indices,
    sm_scale: float,
    d_v: int,
    attn_sink=None,
    topk_length=None,
    **_ignored,
):
    if q.dim() != 3:
        raise ValueError(f"flash_mla_sparse (HCU emu): q must be (s_q, H, d), got {q.shape}")
    s_q, heads, d_qk = q.shape
    kv_flat = kv.reshape(-1, kv.shape[-2], kv.shape[-1])  # [T_kv, h_kv, d_qk]
    if kv_flat.shape[1] != 1:
        raise ValueError(
            f"flash_mla_sparse (HCU emu): h_kv={kv_flat.shape[1]} unsupported (1 only)"
        )
    idx = indices.reshape(s_q, -1).to(torch.int64)  # [s_q, topk]
    topk = idx.shape[1]
    valid = idx >= 0
    if topk_length is not None:
        # rows with fewer valid selectors mask the tail as well
        ar = torch.arange(topk, device=q.device)[None, :]
        valid &= ar < topk_length.reshape(-1, 1).to(torch.int64)
    idx_safe = idx.clamp_min(0)

    # gather: [s_q, topk, d_qk] — per-query selected keys/values
    k_sel = kv_flat[:, 0, :][idx_safe]  # token dim is index 0
    k = k_sel.to(torch.float32)
    scores = torch.einsum("thd,tkd->thk", q.to(torch.float32), k) * sm_scale
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    real_lse = torch.logsumexp(scores, dim=-1)  # [s_q, heads]
    probs = torch.softmax(scores, dim=-1)  # [s_q, heads, topk]
    if attn_sink is not None:
        sink = attn_sink.reshape(1, -1).to(scores.dtype)
        lse = torch.logaddexp(real_lse, sink)
        probs = probs * torch.exp(real_lse - lse).unsqueeze(-1)
    else:
        lse = real_lse
    v = k[..., :d_v]
    o = torch.einsum("thk,tkd->thd", probs, v).to(q.dtype)
    return o, lse, None
