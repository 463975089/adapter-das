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

_CHUNK_MB = None


def _chunk_mb():
    """Per-chunk fp32 working-set budget (MiB) for the row-split emu."""
    global _CHUNK_MB
    if _CHUNK_MB is None:
        try:
            from hcu_adapter import hcu_env

            _CHUNK_MB = max(1, int(hcu_env.resolve("HCU_FLASHMLA_SPARSE_CHUNK_MB")[0]))
        except Exception:
            _CHUNK_MB = 512
    return _CHUNK_MB


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
    if s_q == 0:
        return (
            q.new_empty(0, heads, d_v),
            torch.empty(0, heads, dtype=torch.float32, device=q.device),
            None,
        )
    idx_all = indices.reshape(s_q, -1)
    topk = idx_all.shape[1]
    tl_all = topk_length.reshape(-1) if topk_length is not None else None
    sink = attn_sink.reshape(1, -1).to(torch.float32) if attn_sink is not None else None

    # The fp32 working set is gather [rows, topk, d_qk] + scores/probs
    # [rows, H, topk]; doing all s_q rows at once OOMs on long chunks
    # (34 GiB @ 8k-token prefill) and during prefill-graph capture. Rows are
    # independent, so split along s_q under a fixed budget.
    per_row = 4 * topk * (d_qk + 3 * heads)
    rows = max(1, min(s_q, (_chunk_mb() << 20) // max(per_row, 1)))
    o = q.new_empty(s_q, heads, d_v)
    lse = torch.empty(s_q, heads, dtype=torch.float32, device=q.device)
    kv0 = kv_flat[:, 0, :]  # token dim is index 0
    ar = torch.arange(topk, device=q.device)[None, :] if tl_all is not None else None
    for s in range(0, s_q, rows):
        e = min(s + rows, s_q)
        idx = idx_all[s:e].to(torch.int64)  # [r, topk]
        valid = idx >= 0
        if tl_all is not None:
            # rows with fewer valid selectors mask the tail as well
            valid &= ar < tl_all[s:e].reshape(-1, 1).to(torch.int64)
        # gather: [r, topk, d_qk] - per-query selected keys/values
        k = kv0[idx.clamp_min(0)].to(torch.float32)
        scores = torch.einsum("thd,tkd->thk", q[s:e].to(torch.float32), k) * sm_scale
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        real_lse = torch.logsumexp(scores, dim=-1)  # [r, heads]
        probs = torch.softmax(scores, dim=-1)  # [r, heads, topk]
        if sink is not None:
            lse_c = torch.logaddexp(real_lse, sink)
            probs = probs * torch.exp(real_lse - lse_c).unsqueeze(-1)
        else:
            lse_c = real_lse
        o[s:e] = torch.einsum("thk,tkd->thd", probs, k[..., :d_v]).to(q.dtype)
        lse[s:e] = lse_c
    return o, lse, None
