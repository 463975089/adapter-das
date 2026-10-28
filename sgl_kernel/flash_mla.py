"""Generic paged MLA attention (torch emulation) for the flashmla backend.

`--attention-backend flashmla` tops out at
`from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata`
(flashmla_backend.py:13) — on HCU there is no CUDA FlashMLA wheel, so this
module supplies the same API surface:

- ``get_mla_metadata``: the tile-scheduler prelude. The emulation ignores the
  scheduler entirely, so it returns a constant-shaped row (shape[0] must stay
  identical between capture and replay — the backend keeps a fixed
  ``(sm_parts, 8)`` buffer and asserts against its capacity).
- ``flash_mla_with_kvcache``: absorbed-MLA decode/extend over paged KV.
  q is ``(bs, s_q, H, 576)`` (or ``(M, H, 576)``), k_cache is
  ``(pages, PAGE_SIZE, 1, kv_dim=576)`` where the first ``head_dim_v`` (512)
  channels double as V in the absorbed form; rope channels (512:576) take
  part in the qk dot-product only. The per-request query window is
  right-aligned (last s_q tokens), matching sglang's extend contract.

bf16 kvcache only for the dense paged form; fp8 kvcache (the descale_*
kwargs) raises there.

DSV4 sparse decode (block_table=None, indices=/topk_length=/extra_*) is a
separate branch below: the DeepSeek-V4-Flash core-attention call passes
token-level indices into the SWA pool plus an optional compressed "extra"
pool, both in the V4 byte layout (448 fp8 nope + 64 bf16 rope + 7 ue8m0
scales per token — see sglang/kernels/ops/attention/dsv4/kv_layout.py).
The emulation gathers the selected pages through the tree's own triton
dequantizer and runs the same fp32 math. Everything is tensor-only (no
host reads), so it survives CUDA-graph decode capture.
"""

import os

import torch

# DSV4 sparse-emu debug facility: SGLANG_HCU_FLASHMLA_DEBUG=1 logs the call's
# index/mask/pool stats plus a fingerprint of the gathered rows so page
# content drift across decode steps is visible (syncs — eager only; skipped
# under stream capture).
_FLASHMLA_DEBUG = bool(int(os.environ.get("SGLANG_HCU_FLASHMLA_DEBUG", "0")))
_FLASHMLA_DBG_N = [0]

# DSA sparse attention emu lives in its own module (imported as
# `sgl_kernel.flash_mla_sparse`); re-export here so the historical import
# path `from sgl_kernel.flash_mla import flash_mla_sparse_fwd` (dsa_backend
# _forward_flashmla_sparse) resolves to the emu without upstream edits.
from sgl_kernel.flash_mla_sparse import flash_mla_sparse_fwd  # noqa: F401

__all__ = ["flash_mla_with_kvcache", "get_mla_metadata", "flash_mla_sparse_fwd"]

_DSV4_V4_BYTES_PER_TOKEN = 584  # KVLayout.V4: 576 data + 8 scale bytes


def _dsv4_gather_pages(pool_view, page_ids, lengths):
    """Gather selected rows of a V4-layout paged cache, dequantized to bf16.

    pool_view: ``(pages, P, 1, 584)`` — the view the DSV4 backend builds for
    the flashmla call. page_ids: ``(bs, W)`` int, -1-padded tail — these are
    TOKEN locs into the pool's flat token axis (verified live: writes land at
    loc 256..303 and the reader receives ids 265..272 of that same range; the
    dequantizer's own contract is "token IDs into the cache" and upstream's
    prefill arm feeds it swa_token_ids the same way). lengths: ``(bs,)``
    valid row counts. Returns ``(k [bs, W, 512] bf16, valid [bs, W] bool)``.
    All tensor ops — graph-capture safe.
    """
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
        dequantize_k_cache_paged,
    )

    if pool_view.shape[-1] != _DSV4_V4_BYTES_PER_TOKEN:
        raise NotImplementedError(
            f"flash_mla (HCU emu): DSV4 sparse branch handles the V4 layout "
            f"(584 B/token) only, got {pool_view.shape[-1]} B/token"
        )
    bs, width = page_ids.shape
    page_size = pool_view.shape[1]  # layout param for the dequantizer only
    if pool_view.element_size() != 1:
        raise NotImplementedError(
            f"flash_mla (HCU emu): DSV4 sparse branch expects a byte-typed "
            f"pool view, got {pool_view.dtype}"
        )
    # pool_view is usually NON-contiguous: paged pools pad each row to the
    # layout's TMA stride (V4 rounds the page stride up to a 576B multiple,
    # e.g. 256*584 -> 149760), and the backend builds this view by
    # row-slicing + dim-splitting, so stride(0) > P*584. Re-materialize the
    # 2D (pages, row_stride) form via as_strided: contiguous by construction,
    # zero-copy, and it hands the dequant kernel the padded row stride its
    # page_base addressing already expects.
    row_stride = pool_view.stride(0)
    pool2d = torch.as_strided(
        pool_view, (pool_view.shape[0], row_stride), (row_stride, 1)
    )
    tok = page_ids.to(torch.int64)  # [bs, W] token locs, -1 tail
    k = dequantize_k_cache_paged(
        pool2d, tok.clamp_min(0).reshape(-1).to(torch.int32), page_size
    ).reshape(bs, width, -1)  # [bs, W, 512] bf16
    valid = (
        (tok >= 0)
        & (
            torch.arange(width, device=page_ids.device)[None, :]
            < lengths.to(torch.int64).clamp_min(1)[:, None]
        )
    )  # clamp_min(1): a zero-length row would softmax all--inf into NaN;
    # negative locs clamp to row 0 for the gather, the mask kills them
    return k, valid


def _flash_mla_with_kvcache_sparse(
    q,
    k_cache,
    head_dim_v,
    softmax_scale,
    indices,
    topk_length,
    attn_sink,
    extra_k_cache,
    extra_indices_in_kvcache,
    extra_topk_length,
):
    """DSV4 core-attention form: selected SWA token rows (+ optional compressed
    extra pool), fp8 rope-absorbed cache. Mirrors the upstream call in
    deepseek_v4_backend._forward_attention (q 4D, token-level 3D indices).
    """
    if q.dim() == 3:
        q = q.unsqueeze(1)
    bs, win, heads, d_qk = q.shape
    if indices.ndim == 3:  # (bs, 1, W) page ids
        indices = indices.reshape(indices.shape[0], -1)
    k, valid = _dsv4_gather_pages(k_cache, indices, topk_length)
    if extra_k_cache is not None and extra_indices_in_kvcache is not None:
        if extra_indices_in_kvcache.ndim == 3:
            extra_indices_in_kvcache = extra_indices_in_kvcache.reshape(
                extra_indices_in_kvcache.shape[0], -1
            )
        k_extra, valid_extra = _dsv4_gather_pages(
            extra_k_cache, extra_indices_in_kvcache, extra_topk_length
        )
        k = torch.cat([k, k_extra], dim=1)
        valid = torch.cat([valid, valid_extra], dim=1)

    scale = softmax_scale if softmax_scale is not None else d_qk**-0.5
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    if _FLASHMLA_DEBUG:
        try:
            capturing = torch.cuda.is_current_stream_capturing()
        except Exception:
            capturing = False
        _FLASHMLA_DBG_N[0] += 1
        n = _FLASHMLA_DBG_N[0]
        if not capturing and (n < 40 or n % 100 == 0):
            idx2 = indices.reshape(indices.shape[0], -1)
            W = idx2.shape[1]
            kf_dbg = k.to(torch.float32)  # [bs, W, D]
            va = kf_dbg[valid].abs().mean().item()
            first6 = idx2[0, :6].tolist()
            pstats = "/".join(
                f"#{j}(loc{first6[j] if j < len(first6) else -1})"
                f":{kf_dbg[0, j].abs().mean().item():.4f}"
                for j in range(min(6, W))
            )
            # raw-byte nonzero-ness of suspect pages: 0 (never written?) and
            # the page holding the newest selected token loc
            rs = k_cache.stride(0)
            raw = torch.as_strided(
                k_cache, (k_cache.shape[0], rs), (rs, 1)
            ).view(torch.int16)
            sel_last = int(idx2[0].max().item())
            nzp = (raw != 0).sum(dim=1)  # nonzero int16 words per page
            top5 = nzp.topk(min(5, nzp.numel()))
            rawstat = (
                f"raw pg0={raw[0].abs().float().mean().item():.2f}"
                f" pg1={raw[1].abs().float().mean().item():.2f}"
                f" pg{sel_last // max(1, k_cache.shape[1])}={raw[sel_last // max(1, k_cache.shape[1])].abs().float().mean().item():.2f}"
                f" | hotpages={(nzp > 64).sum().item()}"
                f" top={list(zip(top5.indices.tolist()[:5], top5.values.tolist()[:5]))}"
            )
            print(
                f"[flashmla-dbg #{n}] q={tuple(q.shape)} pool={tuple(k_cache.shape)}"
                f" stride={k_cache.stride(0)} idx[min={idx2.min().item()}"
                f" max={idx2.max().item()} uniq={idx2.unique().numel()}]"
                f" topk={topk_length.reshape(-1)[:8].tolist()}"
                f" validrows_absmean={va:.4f} page_absmean[{pstats}] {rawstat}"
                + (
                    f" EXTRA pool={tuple(extra_k_cache.shape)}"
                    f" topk_e={extra_topk_length.reshape(-1)[:4].tolist()}"
                    if extra_k_cache is not None
                    else ""
                ),
                flush=True,
            )
    v = kf[..., :head_dim_v]
    # same operand shapes as the dense path's proven q@k 4D matmul
    scores = torch.matmul(qf, kf.transpose(1, 2).unsqueeze(1))  # [bs,win,H,T]
    scores = scores * scale
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    if attn_sink is not None:
        scores = scores + attn_sink.to(torch.float32).reshape(1, 1, -1, 1)
    probs = torch.softmax(scores, dim=-1)
    # p@v as 3D bmm (gfx938 rejects the p@v 4D combo): fold (win, heads)
    # into the batch — v needs no per-head expand this way
    T = k.shape[1]
    o = torch.matmul(probs.reshape(bs, win * heads, T), v).reshape(
        bs, win, heads, head_dim_v
    )
    lse = torch.logsumexp(scores, dim=-1)  # [bs, win, H]
    return o.to(q.dtype), lse


class FlashMLASchedMeta:
    """Stand-in for upstream's persistent scheduler-state object.

    Upstream AOT ``get_mla_metadata()`` with ``cache_seqlens=None`` returns
    ``(FlashMLASchedMeta(), None)``; callers keep the object and overwrite
    ``tile_scheduler_metadata`` / ``num_splits`` on later shape-bearing
    calls (deepseek_v4_backend*.py do ``get_mla_metadata()[0]``).
    """

    def __init__(self):
        self.have_initialized = False
        self.config = None
        self.tile_scheduler_metadata = None
        self.num_splits = None


def get_mla_metadata(
    cache_seqlens=None,
    num_q_tokens_per_head_k: int = None,
    num_heads_k: int = None,
    num_heads_q: int = None,
    is_fp8_kvcache: bool = False,
    topk: int = None,
):
    """Constant-shape stand-in for the FlashMLA tile scheduler.

    Signature matches the upstream AOT API (v0.5.21 dsa_backend
    ``_compute_flashmla_metadata`` calls it fully keyworded with
    ``cache_seqlens=``/``num_q_tokens_per_head_k=``/``topk=``). Positional
    order keeps the legacy flash-mla call form
    ``get_mla_metadata(seqlens, q_heads_per_kv, kv_heads)`` working
    (tests/kernels/test_flash_mla.py). Returns a single scheduler row so
    capture/replay shapes can never diverge; ``num_splits`` follows the
    upstream ``(bs + 1,)`` int32 shape. Both are ignored by the emulation
    below.
    """
    if cache_seqlens is None:
        return FlashMLASchedMeta(), None
    dev = cache_seqlens.device
    mla_metadata = torch.zeros((1, 8), dtype=torch.int32, device=dev)
    num_splits = torch.zeros(
        (cache_seqlens.shape[0] + 1,), dtype=torch.int32, device=dev
    )
    return mla_metadata, num_splits


def flash_mla_with_kvcache(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    head_dim_v: int,
    tile_scheduler_metadata=None,
    num_splits=None,
    softmax_scale=None,
    causal: bool = False,
    descale_q=None,
    descale_k=None,
    indices=None,
    topk_length=None,
    attn_sink=None,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    extra_topk_length=None,
    **_ignored,
):
    if block_table is None:
        # DSV4 sparse core-attention: token-level indices, fp8 V4-layout
        # pools (deepseek_v4_backend._forward_attention). tile scheduler /
        # num_splits / is_fp8_kvcache are the CUDA kernel's business only.
        if indices is None:
            raise ValueError(
                "flash_mla (HCU emu): block_table=None requires the DSV4 "
                "sparse call form (indices=/topk_length=)"
            )
        return _flash_mla_with_kvcache_sparse(
            q,
            k_cache,
            head_dim_v,
            softmax_scale,
            indices,
            topk_length,
            attn_sink,
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"flash_mla (HCU emu): q dtype {q.dtype} unsupported; bf16/fp16 "
            "kvcache only — fp8 kvcache belongs to the NSA/DSA backends"
        )
    if descale_q is not None or descale_k is not None:
        raise NotImplementedError(
            "flash_mla (HCU emu): fp8 kvcache descale path not supported"
        )
    has_batch = q.dim() == 4
    if has_batch:
        bs, win, heads, d_qk = q.shape
        q3 = q.reshape(bs * win, heads, d_qk)
    else:
        m, heads, d_qk = q.shape
        bs = cache_seqlens.shape[0]
        if m % bs:
            raise ValueError(
                f"flash_mla (HCU emu): q tokens {m} not divisible by batch {bs}"
            )
        win = m // bs
        q3 = q
    if head_dim_v > d_qk:
        raise ValueError(
            f"head_dim_v {head_dim_v} exceeds d_qk {d_qk}"
        )

    scale = softmax_scale if softmax_scale is not None else d_qk**-0.5
    dtype = q.dtype
    device = q.device
    page_size = k_cache.shape[1]
    max_pages = block_table.shape[1]
    seqlens = cache_seqlens.to(torch.int64)

    flat_k = k_cache.reshape(k_cache.shape[0], page_size, k_cache.shape[-1])

    # gather whole pages once per (request, page) pair: index row = page id,
    # 2D-gather [bs, max_pages] page rows, then expand slots to token grid.
    # Rows past each request's length stay -1-padded by the backend's fill
    # kernel (it only rewrites up to the current K len): clamp them to page 0
    # so index_select stays in bounds — the seqlens mask below makes those
    # columns' values irrelevant.
    page_ids = block_table.to(torch.int64).clamp_min(0)  # [bs, max_pages]
    page_rows = torch.index_select(flat_k, 0, page_ids.reshape(-1)).reshape(
        bs, max_pages, page_size, d_qk
    )
    k_full = page_rows.reshape(bs, max_pages * page_size, d_qk)
    k = k_full.to(torch.float32)
    v = k[..., :head_dim_v]
    qf = q3.to(torch.float32)

    # plain bmm form — einsum on this virtual-batch shape VMFaults gfx938
    q4 = qf.view(bs, win, heads, d_qk)
    scores = torch.matmul(q4, k.transpose(1, 2).unsqueeze(1)).permute(
        0, 2, 1, 3
    )  # [bs, heads, win, T_kv]
    scores = scores * scale
    tok_ids = torch.arange(max_pages * page_size, device=device)
    # logical token position = index of the page IN THE REQUEST'S block list
    # (page_of_tok), NOT the stored page id — causal masking compares
    # per-request token orderings
    kv_pos = (
        (tok_ids // page_size)[None, :] * page_size + (tok_ids % page_size)
    ).expand(bs, -1)  # [bs, T_kv] — sequential layout, same for all requests
    valid_kv = kv_pos < seqlens[:, None]  # same ordering gates causality AND validity
    if causal and win > 1:
        # query row j (0..win-1) sits at logical position seqlen - win + j
        q_pos = (seqlens - win)[:, None, None, None] + torch.arange(
            win, device=device
        )[None, None, :, None]  # [bs, 1, win, 1]
        scores = scores.masked_fill(kv_pos[:, None, None, :] > q_pos, float("-inf"))
    scores = scores.masked_fill(
        ~valid_kv[:, None, None, :], float("-inf")
    )
    probs = torch.softmax(scores, dim=-1)  # [bs, heads, win, T]
    # p@v via plain 3D bmm: HIP's 4D-matmul path rejects this batch combo —
    # fold (heads, win) into the bmm batch and tile the shared V per query
    T_kv = max_pages * page_size
    pm = probs.permute(0, 2, 1, 3).reshape(bs * win * heads, 1, T_kv).contiguous()
    v3 = v[:, None, None].expand(
        bs, win, heads, T_kv, head_dim_v
    ).reshape(bs * win * heads, T_kv, head_dim_v).contiguous()
    o = torch.bmm(pm, v3)  # [bs*win*heads, 1, d_v]
    # rank follows q: flashmla_backend's caller does
    # o.view(-1, tp_q_head_num * v_head_dim) on the 3D [M, H, d_v] form,
    # while dsa_backend _forward_flashmla_kv passes 4D (bs, s_q, H, D) and
    # head-slices o[:, :, :num_q_heads, :] — give each the rank it sent.
    if has_batch:
        o = o.view(bs, win, heads, head_dim_v).to(dtype)
    else:
        o = o.view(bs * win, heads, head_dim_v).to(dtype)
    lse = torch.logsumexp(scores, dim=-1)  # [bs, heads, win]
    return o, lse
