# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Paged MHA/GQA flash-attention emulation for the fa3 backend.

sglang v0.5.21's ``--attention-backend fa3`` resolves to
``sglang.kernels.ops.attention.flash_attention``, whose fa3 path dispatches
(via ``SGLANG_USE_SGL_FA3_KERNEL=1``, forced by the HCU adapter env rules)
to ``from sgl_kernel.flash_attn import flash_attn_varlen_func,
flash_attn_with_kvcache``; the backend additionally imports
``merge_state_v2`` and ``get_scheduler_metadata``. On HCU there is no
FlashAttention-3 wheel, so this module supplies the same API surface as a
torch emulation.

Semantics pinned to the FlashAttention-3 contract as used by sglang:

- ``flash_attn_with_kvcache``: paged KV (``(pages, page, H_kv, D)`` HND
  layout), flattened q ``(M, H, D)`` with ``cu_seqlens_q`` giving the query
  window per request. Causal masking is aligned to the bottom-right corner:
  query row ``j`` of request ``b`` sits at logical position
  ``cache_seqlens[b] - len_q[b] + j``.
- ``flash_attn_varlen_func``: ragged dense K/V with the same bottom-right
  causal alignment per sequence.
- fp8 KV cache: the pool stores fp8 (uint8 storage, ``.view(fp8)`` on read)
  and the backend passes ``k_descale``/``v_descale`` (layer scales); the
  emulation upcasts each gathered chunk to fp32 and multiplies the descale.
- ``merge_state_v2``: numerically exact log-sum-exp merge of two partial
  attention states (MTP verify / cascade paths).

Memory model: attention runs as chunked online softmax (flash-decoding
style) — KV is gathered in fixed-size token chunks (whole pages), each
chunk produces a partial (o, lse), and an exact logsumexp merge folds them
into the accumulator. Query rows are additionally sub-tiled (QTILE_TOKENS),
since the (bs, rows, H, C) score slabs grow with the padded window too —
without the row tile an 8k-token extend face peaked at ~22 GiB live fp32
(R-13 #2). Peak memory is
O(bs * QTILE_TOKENS * H * CHUNK_TOKENS + bs * maxL * H * D) instead of
O(bs * maxL * H * T), which is what lets CUDA-graph capture at bs=256 with
full context width survive on a card whose KV pool already took ~90% of
HBM. The chunk count depends only on the block-table width, so
capture/replay shapes stay identical.

gfx938 constraints honored throughout (see flash_mla.py): no einsum on
virtual-batch shapes (VMFault), p@v folded to plain 3D bmm with equal
batch dims, page-granular ``index_select`` gathers with ``clamp_min(0)``
for the -1-padded tail of the block table.
"""

import os

import torch

__all__ = [
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "get_scheduler_metadata",
    "merge_state_v2",
]

# KV tokens processed per chunk; must be a multiple of every page size we
# see (all powers of two up to 128 -> 2048 works). Bigger = fewer python
# iterations, smaller = lower peak memory during graph capture.
CHUNK_TOKENS = 2048
# Sub-tile over the query rows: the (bs, maxL, H, C) scores/probs slabs grow
# with BOTH the kv chunk and the padded window length, and an 8k-token
# extend face (chunked prefill) made only the kv loop's 2048-wide chunking
# peak at ~22 GiB live fp32 -- fatal on top of a loaded serving pool (the
# R-13 #2 "10k wall": OOM 28.88 GiB single-alloc rank-kill on glm53f tp4).
# Query rows are independent, so tiling them needs no softmax trickery:
# each row block streams the same kv chunks with the identical LSE merge
# order -> bit-identical results. Tile sizes come from static shapes only
# (maxL/total_tokens), never runtime values -> capture-safe.
QTILE_TOKENS = 1024


def get_scheduler_metadata(**_kwargs):
    """FA3 tile-scheduler prelude stub.

    The backend precomputes this once per decode batch and threads it
    through ``scheduler_metadata=``; the emulation ignores it entirely, and
    None is the FA API's "compute internally" value, so nothing downstream
    cares about the shape.
    """
    return None


def _descale(mul, tensor, heads):
    """Apply a FA descale factor to a (.., H_kv, D) fp32 chunk.

    FA descale semantics: ``k_descale``/``v_descale`` carry one factor per
    KV head (``layer.k_scale.expand``); None means 1.0.
    """
    if mul is None:
        return tensor
    if not torch.is_tensor(mul):
        mul = tensor.new_tensor([float(mul)])
    mul = mul.to(tensor.dtype).reshape(1, 1, heads, 1)
    return tensor * mul


class _PagedKV:
    """Chunked (bs, C, H_kv, D) view over a paged HND KV cache."""

    def __init__(self, cache, page_table):
        pages, page, self.Hk, self.D = cache.shape
        self.page = page
        self.flat = cache.reshape(pages, page * self.Hk * self.D)
        self.pt = page_table.to(torch.int64).clamp_min(0)

    def __call__(self, c0, c1):
        bs = self.pt.shape[0]
        p0, p1 = c0 // self.page, c1 // self.page
        rows = torch.index_select(
            self.flat, 0, self.pt[:, p0:p1].reshape(-1)
        )
        chunk = rows.reshape(
            bs, (p1 - p0) * self.page, self.Hk, self.D
        ).to(torch.float32)
        return chunk


class _DenseKV:
    """Chunked (1, C, H_kv, D) view over one sequence's ragged K/V slice."""

    def __init__(self, k_seq, v_seq):
        self.k = k_seq
        self.v = v_seq

    def __call__(self, c0, c1):
        return (
            self.k[c0:c1].to(torch.float32).unsqueeze(0),
            self.v[c0:c1].to(torch.float32).unsqueeze(0),
        )


def _grouped_scores(qp, k_c, scale):
    """q·k^T with GQA grouping, all-batched matmul form.

    qp: (bs, maxL, H, D) fp32; k_c: (bs, C, H_kv, D) fp32.
    Returns (bs, maxL, H, C) fp32.
    """
    bs, maxL, H, D = qp.shape
    C, Hk = k_c.shape[1], k_c.shape[2]
    G = H // Hk
    q2 = qp.view(bs, maxL, Hk, G, D).permute(0, 2, 3, 1, 4).reshape(
        bs * Hk, G * maxL, D
    )
    k2 = k_c.permute(0, 2, 3, 1).reshape(bs * Hk, D, C)
    s = torch.bmm(q2, k2)  # (bs*Hk, G*maxL, C)
    return (
        s.view(bs, Hk, G, maxL, C)
        .permute(0, 3, 1, 2, 4)
        .reshape(bs, maxL, H, C)
        * scale
    )


def _grouped_pv(probs, v_c):
    """p@v with GQA grouping via plain 3D bmm (no giant expand).

    probs: (bs, maxL, H, C); v_c: (bs, C, H_kv, D_v).
    Returns (bs, maxL, H, D_v).
    """
    bs, maxL, H, C = probs.shape
    Hk, Dv = v_c.shape[2], v_c.shape[3]
    G = H // Hk
    p2 = (
        probs.view(bs, maxL, Hk, G, C)
        .permute(0, 2, 3, 1, 4)
        .reshape(bs * Hk, G * maxL, C)
    )
    v2 = v_c.permute(0, 2, 1, 3).reshape(bs * Hk, C, Dv)
    o = torch.bmm(p2, v2)  # (bs*Hk, G*maxL, Dv)
    return (
        o.view(bs, Hk, G, maxL, Dv).permute(0, 3, 1, 2, 4).reshape(
            bs, maxL, H, Dv
        )
    )


def _merge_acc(o_acc, lse_acc, o_c, lse_c):
    """Exact streaming merge of partial attention states (both fp32)."""
    m = torch.maximum(lse_acc, lse_c)
    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    w_acc = torch.exp(lse_acc - m_safe)
    w_c = torch.exp(lse_c - m_safe)
    denom = w_acc + w_c
    o = (o_acc * w_acc[..., None] + o_c * w_c[..., None]) / denom[..., None]
    # fully-masked rows (denominator 0, both lse -inf) -> zero output, -inf
    # lse, matching the FA contract for all-masked rows
    o = torch.nan_to_num(o, nan=0.0)
    lse = m + torch.log(denom)
    return o, lse


def _attend_chunked(qp, kv_src, seqlens, q_start, scale, causal, window_size,
                    softcap, k_descale, v_descale, out_dtype, total_tokens,
                    qvp=None):
    """Online-softmax attention over logical-order KV in token chunks.

    qp: (bs, maxL, H, D) fp32 zero-padded queries
    qvp: optional (bs, maxL, H, D_v) fp32 "value query" — the FA3 MLA split
    form (sglang's ROCm absorbed path): logits = q.k + qv.v, output = P@v.
    kv_src(c0, c1) -> (k, v) (bs, C, H_kv, D) fp32 chunks in logical order
    seqlens/q_start: (bs,) int64; maxL folded into qp.
    total_tokens: STATIC full width (page-aligned block-table span). The
    chunk count derives from it only — never from runtime tensor values —
    so CUDA-graph capture and replay run the identical python loop.
    """
    bs, maxL, H, _ = qp.shape
    device = qp.device
    T = max(int(total_tokens), 1)

    left = right = None
    if window_size is not None and tuple(window_size) != (-1, -1):
        left = int(window_size[0])
        right = int(window_size[1])

    chunk = CHUNK_TOKENS
    qtile = QTILE_TOKENS
    o_blocks, lse_blocks = [], []
    saw_kv = False
    for q0 in range(0, maxL, qtile):
        q1 = min(q0 + qtile, maxL)
        qp_b = qp[:, q0:q1]
        qvp_b = qvp[:, q0:q1] if qvp is not None else None
        qpos = q_start[:, None] + torch.arange(q0, q1, device=device)[None, :]
        o_acc, lse_acc = None, None
        for c0 in range(0, T, chunk):
            c1 = min(c0 + chunk, T)
            saw_kv = True
            k_c, v_c = kv_src(c0, c1)
            Hk = v_c.shape[2] if k_c is None else k_c.shape[2]
            k_c = _descale(k_descale, k_c, Hk)
            v_c = _descale(v_descale, v_c, Hk)
            if k_c is None:
                # only_qv: logits = qv·v alone (zero query slab contributes
                # nothing)
                scores = _grouped_scores(qvp_b, v_c, scale)  # (bs, qL, H, C)
            else:
                scores = _grouped_scores(qp_b, k_c, scale)
                if qvp_b is not None:
                    scores = scores + _grouped_scores(qvp_b, v_c, scale)

            col = torch.arange(c0, c1, device=device)
            valid = col[None, :] < seqlens[:, None]  # (bs, C)
            keep = valid[:, None, None, :].expand(bs, q1 - q0, 1, c1 - c0)
            if causal:
                keep = keep & (
                    col[None, None, None, :] <= qpos[:, :, None, None]
                )
            if left is not None:
                rel = col[None, None, None, :] - qpos[:, :, None, None]
                # FA window: query at pos p attends kv in [p-left, p+right]
                keep = keep & (rel >= (-left if left >= 0 else -(2**62)))
                keep = keep & (rel <= (right if right >= 0 else 2**62))
            scores = scores.masked_fill(~keep, float("-inf"))
            if softcap and softcap > 0.0:
                scores = softcap * torch.tanh(scores / softcap)
            probs = torch.softmax(scores, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            o_c = _grouped_pv(probs, v_c)
            lse_c = torch.logsumexp(scores, dim=-1)  # (bs, qL, H)

            if o_acc is None:
                o_acc, lse_acc = o_c, lse_c
            else:
                o_acc, lse_acc = _merge_acc(o_acc, lse_acc, o_c, lse_c)
        if o_acc is None:  # T == 0: everything masked
            o_acc = qp_b.new_zeros(
                bs, q1 - q0, qp_b.shape[2], qvp_b.shape[3]
                if qvp_b is not None else qp_b.shape[3])
            lse_acc = qp_b.new_full((bs, q1 - q0, qp_b.shape[2]),
                                    float("-inf"))
        o_blocks.append(o_acc)
        lse_blocks.append(lse_acc)

    if not saw_kv:  # T == 0: everything masked
        raise ValueError("flash_attn emu: empty KV range")
    return (torch.cat(o_blocks, dim=1).to(out_dtype),
            torch.cat(lse_blocks, dim=1))


def _pad_queries(q3, cu_q, bs, maxL, device):
    """Zero-pad flattened (M, H, D) queries into a static (bs, maxL, H, D)
    grid — CUDA-graph-safe: no op here has a content-dependent output shape
    (repeat_interleave on a tensor syncs D2H and aborts capture)."""
    M = q3.shape[0]
    lens = (cu_q[1:] - cu_q[:-1]).to(torch.int64)  # (bs,)
    cu_q64 = cu_q.to(torch.int64)
    grid_j = torch.arange(maxL, device=device).unsqueeze(0)  # (1, maxL)
    valid_cell = grid_j < lens.unsqueeze(1)  # (bs, maxL)
    src = (cu_q64[:-1].unsqueeze(1) + grid_j).clamp(max=M - 1)
    # upcast before the mask mul: fp8-quantized q (the backend's fp8-q path)
    # has no mul_cuda; fp32 here also fixes the compute precision for free
    qg = q3[src.reshape(-1)].to(torch.float32) * valid_cell.reshape(-1, 1, 1)
    qp = qg.view(bs, maxL, *q3.shape[1:])
    # reverse map for the output gather: token m -> (bs*maxL) grid cell
    tok = torch.arange(M, device=device)
    b_of_tok = torch.searchsorted(cu_q64[1:], tok, right=True)
    cell_of_tok = b_of_tok * maxL + (tok - cu_q64[b_of_tok])
    return qp, cell_of_tok, lens


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    page_table=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    max_seqlen_q=None,
    rotary_seqlens=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=None,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    only_qv=False,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    out=None,
    **_ignored,
):
    if (
        q is None
        and not only_qv
        and qv is not None
        and k_cache is None
    ):
        # GLM-5.x DSA sparse decode (dsa_backend flashmla_sparse route over
        # the fa3 router) calls flash_attn_with_kvcache(q=None, qv=...,
        # k_cache=None, v_cache=c_kv) WITHOUT the only_qv kwarg: with no k
        # side at all there is nothing q could attend, so qv is necessarily
        # the only query — infer the flag instead of rejecting (§0.5: the
        # upstream call-shape is the contract).
        only_qv = True
    if q is None and not only_qv:
        raise ValueError("flash_attn emu: q is None but only_qv is False")
    if only_qv:
        if qv is None:
            raise ValueError("flash_attn emu: only_qv requires qv")
        if q_descale is not None or sinks is not None:
            raise NotImplementedError(
                "flash_attn (HCU emu): q_descale/sinks paths not supported"
            )
        # FA3 only_qv=true (MLA absorbed decode with rope dim 0): qv is the
        # sole query; logits = qv.v (the k side is a zero slab so the shared
        # chunked scorer degenerates to it), softmax scale applies to the
        # qv·v dot itself — folded into qvp below.
        q_shape = qv.shape
        q = torch.zeros(q_shape[:-1] + (1,), dtype=qv.dtype, device=qv.device)
    if q is None or (not only_qv and (q_descale is not None or sinks is not None)):
        raise NotImplementedError(
            "flash_attn (HCU emu): q_descale/sinks paths not supported"
        )
    if q.dim() not in (3, 4):
        raise ValueError(f"flash_attn emu: q must be 3D/4D, got {q.dim()}")
    if rotary_cos is not None or k is not None:
        raise NotImplementedError(
            "flash_attn (HCU emu): in-kernel cache update / rotary not "
            "supported — sglang always writes the KV pool separately"
        )
    if page_table is None:
        raise NotImplementedError(
            "flash_attn (HCU emu): non-paged batch KV path not used by sglang"
        )
    if cache_seqlens is None:
        raise ValueError("flash_attn emu: cache_seqlens required")

    if q.numel() == 0:
        # degenerate warmup/spec batch: 0 tokens (or the D=0 zero-query
        # slab, also 0 elements) — reshape(-1, H, D) is ambiguous on an
        # empty tensor, so emit empty outputs directly.
        dtype = torch.bfloat16 if q.dtype.itemsize == 1 else q.dtype
        h0 = q.shape[-2]
        dv0 = v_cache.shape[-1] if v_cache is not None else 0
        o = q.new_empty((0, h0, dv0), dtype=dtype)
        if out is not None:
            o = out
        if return_softmax_lse:
            return o, q.new_empty((0, h0), dtype=torch.float32)
        return o

    device = q.device
    # fp8-quantized q means the caller runs the fp8-q path; outputs must
    # come back in a compute dtype either way
    dtype = torch.bfloat16 if q.dtype.itemsize == 1 else q.dtype
    q3 = q.reshape(-1, q.shape[-2], q.shape[-1])
    M, H, D = q3.shape
    bs = page_table.shape[0]
    # only_qv (kq_rope_dim==0 DSA decode upstream passes k_cache=None): the
    # zero query slab makes q·k identically zero, so any K geometry works
    Hk = k_cache.shape[2] if k_cache is not None else H
    if H % Hk:
        raise ValueError(f"flash_attn emu: GQA {H} not divisible by {Hk}")

    if cu_seqlens_q is None:
        cu_q = torch.arange(0, M + 1, M // bs, dtype=torch.int32, device=device)
        maxL = max(M // bs, 1)
    else:
        cu_q = cu_seqlens_q.to(torch.int32)
        # STATIC window length only: int(lens_q.max()) is a D2H sync and
        # aborts CUDA-graph capture (hipErrorStreamCaptureUnsupported). The
        # backend passes max_seqlen_q on the graph-captured paths; the
        # tensor read stays only as an eager fallback.
        if max_seqlen_q is not None:
            maxL = max(int(max_seqlen_q), 1)
        elif q.dim() == 4:
            maxL = q.shape[1]
        else:
            maxL = max(int((cu_q[1:] - cu_q[:-1]).max()), 1)
    lens_q = (cu_q[1:] - cu_q[:-1]).to(torch.int64)

    if os.environ.get("HCU_FA_EMU_DEBUG", "") and not torch.cuda.is_current_stream_capturing():
        import logging

        logging.getLogger("hcu.adapter").warning(
            "FA_EMU kvcache: q %s cu_q=%s cache_seqlens=%s causal=%s "
            "qsum=%.4f",
            tuple(q.shape),
            [int(v) for v in cu_q[:8].tolist()],
            [int(v) for v in cache_seqlens[:8].tolist()],
            causal,
            float(q.float().sum()),
        )

    scale = softmax_scale if softmax_scale is not None else (
        qv.shape[-1] ** -0.5 if only_qv else D**-0.5
    )
    qp, cell, _ = _pad_queries(q3, cu_q, bs, maxL, device)
    qvp = None
    if only_qv:
        # zero out the rope-dot side of the logit fusion below by clearing q;
        # upstream's softmax_scale applies to the fused (q.k + qv.v) and with
        # q.k absent it belongs entirely to the qv·v term — fold into qvp and
        # feed _attend_chunked scale=1.
        qv3 = qv.reshape(-1, qv.shape[-2], qv.shape[-1]).to(torch.float32)
        if qv3.shape[0] != M or qv3.shape[1] != H:
            raise ValueError(
                f"flash_attn emu: qv {tuple(qv.shape)} mismatches q "
                f"{tuple(q.shape)}"
            )
        qvp = _pad_queries(qv3 * scale, cu_q, bs, maxL, device)[0]
        scale = 1.0
    elif qv is not None:
        # FA3 MLA split form: qv is the w_kc-absorbed latent query whose dot
        # with the compressed KV adds to the logits (rope dot covers the
        # rest). Same padding grid as q.
        qv3 = qv.reshape(-1, qv.shape[-2], qv.shape[-1]).to(torch.float32)
        if qv3.shape[0] != M or qv3.shape[1] != H:
            raise ValueError(
                f"flash_attn emu: qv {tuple(qv.shape)} mismatches q "
                f"{tuple(q.shape)}"
            )
        qvp, _, _ = _pad_queries(qv3, cu_q, bs, maxL, device)
    src_k = None if k_cache is None else _PagedKV(k_cache, page_table)
    src_v = _PagedKV(v_cache, page_table)

    seqlens = cache_seqlens.to(torch.int64)
    q_start = seqlens - lens_q  # bottom-right aligned window start

    def kv_pair(c0, c1):
        if src_k is None:
            return None, src_v(c0, c1)
        return src_k(c0, c1), src_v(c0, c1)

    o_pad, lse_pad = _attend_chunked(
        qp, kv_pair, seqlens, q_start, scale, causal, window_size, softcap,
        k_descale, v_descale, dtype,
        total_tokens=page_table.shape[1] * v_cache.shape[1],
        qvp=qvp,
    )
    flat_o = o_pad.reshape(-1, *o_pad.shape[2:])  # (bs*maxL, H, Dv)
    o = flat_o[cell]  # (M, H, Dv) back in caller token order
    if out is not None:
        out.copy_(o)
        o = out
    if return_softmax_lse:
        flat_lse = lse_pad.reshape(-1, lse_pad.shape[-1])  # (bs*maxL, H)
        return o, flat_lse[cell].to(torch.float32)  # (M, H)
    return o


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    only_qv=False,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    out=None,
    **_ignored,
):
    if page_table is not None:
        raise NotImplementedError(
            "flash_attn emu: paged varlen path — sglang routes paged "
            "prefill through flash_attn_with_kvcache"
        )
    if qv is not None or sinks is not None or q_descale is not None:
        raise NotImplementedError(
            "flash_attn emu: qv/sinks/q_descale unsupported"
        )

    device = q.device
    dtype = q.dtype
    q3 = q.reshape(-1, q.shape[-2], q.shape[-1])
    k3 = k.reshape(-1, k.shape[-2], k.shape[-1])
    v3 = v.reshape(-1, v.shape[-2], v.shape[-1])
    M, H, D = q3.shape
    Hk = k3.shape[1]
    if H % Hk:
        raise ValueError(f"flash_attn emu: GQA {H} not divisible by {Hk}")

    cu_q = cu_seqlens_q.to(torch.int32)
    cu_k = cu_seqlens_k.to(torch.int32)
    bs = cu_q.shape[0] - 1

    scale = softmax_scale if softmax_scale is not None else D**-0.5
    outs, lses = [], []
    for b in range(bs):
        Lq = int(cu_q[b + 1] - cu_q[b])
        s = int(cu_k[b + 1] - cu_k[b])
        if Lq == 0:
            continue
        qb = q3[int(cu_q[b]) : int(cu_q[b + 1])].to(torch.float32).unsqueeze(0)
        src = _DenseKV(
            k3[int(cu_k[b]) : int(cu_k[b + 1])],
            v3[int(cu_k[b]) : int(cu_k[b + 1])],
        )

        def kv_pair(c0, c1, _src=src):
            return _src(c0, c1)

        seqlens = torch.tensor([s], dtype=torch.int64, device=device)
        q_start = torch.tensor([s - Lq], dtype=torch.int64, device=device)
        o_b, lse_b = _attend_chunked(
            qb, kv_pair, seqlens, q_start, scale, causal, window_size,
            softcap, k_descale, v_descale, dtype, total_tokens=s,
        )
        outs.append(o_b[0])
        lses.append(lse_b[0])
    o = torch.cat(outs, dim=0)
    if out is not None:
        out.copy_(o)
        o = out
    if return_softmax_lse:
        return o, torch.cat(lses, dim=0).to(torch.float32)
    return o


def merge_state_v2(o1, lse1, o2, lse2, out=None, out_lse=None):
    """Exact log-sum-exp merge of two partial attention states.

    sglang feeds (o, lse) pairs straight from flash_attn_with_kvcache, so
    lse layout here is (M, H) matching this module's emu; o is (M, H, Dv).
    The CUDA signature additionally takes ``out``/``out_lse`` write-through
    buffers and always returns the merged pair; when given they are filled
    in place and their dtype (not the input's) carries the result, since
    sglang's chunked prefix path accumulates in the buffers it allocates.
    """
    if o1.dim() == lse1.dim() + 1:  # (M, H, Dv) with (M, H) lse
        o, lse = _merge_acc(
            o1.float(), lse1.float(), o2.float(), lse2.float()
        )
        if out is not None:
            out.copy_(o)
            o = out
        else:
            o = o.to(o1.dtype)
        if out_lse is not None:
            out_lse.copy_(lse)
            lse = out_lse
        return o, lse
    raise ValueError(
        f"merge_state_v2 emu: unexpected shapes o {tuple(o1.shape)} "
        f"lse {tuple(lse1.shape)}"
    )
