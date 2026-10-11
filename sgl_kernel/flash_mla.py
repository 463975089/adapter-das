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

bf16 kvcache only; fp8 kvcache (the descale_* kwargs) raises — fp8 sparse
paths belong to the DSA/NSA backends, not this one.
"""

import torch

__all__ = ["flash_mla_with_kvcache", "get_mla_metadata"]


def get_mla_metadata(
    seq_lens,
    num_q_heads_per_kv_head: int,
    num_kv_heads: int,
    is_fp8_kvcache: bool = False,
):
    """Constant-shape stand-in for the FlashMLA tile scheduler.

    Returns a single scheduler row so capture/replay shapes can never
    diverge; ``num_splits`` follows the upstream ``(bs + 1,)`` int32 shape.
    Both are ignored by the emulation below.
    """
    dev = seq_lens.device
    mla_metadata = torch.zeros((1, 8), dtype=torch.int32, device=dev)
    num_splits = torch.zeros(
        (seq_lens.shape[0] + 1,), dtype=torch.int32, device=dev
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
    **_ignored,
):
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
    # upstream contract (flashmla_backend.py): caller does
    # o.view(-1, tp_q_head_num * v_head_dim) — 3D [M, H, d_v] always works
    o = o.view(bs * win, heads, head_dim_v).to(dtype)
    lse = torch.logsumexp(scores, dim=-1)  # [bs, heads, win]
    return o, lse
