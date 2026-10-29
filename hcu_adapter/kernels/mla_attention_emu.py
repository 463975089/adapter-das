"""FlashMLA fused norm + RoPE + sparse attention + O-RoPE + FP8 cast (prefill) on HCU.

Contract: vendor FlashMLA flash_mla/fused_norm_rope_attn_rope_cast.py::prefill and
csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp; numerics follow the vendor
reference tests/ref.py::ref_sparse_attn_fwd plus the fused test's Q/O RoPE.
Layouts: q arrives permuted by permute_q_b_proj (16-column chunks interleaved
across heads); the FP8 output is permuted to match permute_wv_proj (32-column
chunks interleaved across the heads of each wv group). No deepinfer imports.
"""

import torch

from .quant_emu import per_token_cast

_CHUNK_ELEMENTS = 1 << 28  # bound the gathered FP32 KV per token chunk (~1 GiB)


def _rope_interleaved(x, cos_sin_cache, positions, rope_dim, conjugate):
    """In-place FP32 interleaved RoPE on x[..., -rope_dim:] (x: [s, h, d], float32)."""
    cos = cos_sin_cache.index_select(0, positions.long())[:, None, :rope_dim // 2]
    sin = cos_sin_cache.index_select(0, positions.long())[:, None, rope_dim // 2:]
    if conjugate:
        sin = -sin
    part = x[..., -rope_dim:]
    x0, x1 = part[..., 0::2].clone(), part[..., 1::2].clone()
    part[..., 0::2] = torch.addcmul(-(x1 * sin), x0, cos)
    part[..., 1::2] = torch.addcmul(x1 * cos, x0, sin)


def _unpermute_q(q, h_q, d_qk):
    s_q = q.shape[0]
    return q.reshape(s_q, d_qk // 16, h_q, 16).permute(0, 2, 1, 3).reshape(s_q, h_q, d_qk)


def _sparse_attention(q, kv, indices, sm_scale, d_v, attn_sink, topk_length, norm_scale):
    """Vendor ref.py::ref_sparse_attn_fwd on one token chunk. Returns (out fp32, max_logits, lse)."""
    s_q, h_q, _ = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[-1]
    idx = indices.reshape(s_q, topk)
    invalid = (idx < 0) | (idx >= s_kv)
    if topk_length is not None:
        invalid |= torch.arange(topk, device=idx.device)[None, :] >= topk_length[:, None]
    safe = torch.where(invalid, 0, idx).long()
    gathered = kv.reshape(s_kv, -1).index_select(0, safe.flatten()).reshape(s_q, topk, -1).float()
    logits = torch.bmm(q.float(), gathered.transpose(1, 2))
    scale = sm_scale * norm_scale.unsqueeze(-1) if norm_scale is not None else sm_scale
    logits = logits * scale
    logits = logits.masked_fill(invalid[:, None, :], float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    max_logits = logits.amax(dim=-1)
    if attn_sink is not None:
        lse_for_o = torch.logsumexp(torch.stack((lse, attn_sink.float().expand_as(lse))), dim=0)
    else:
        lse_for_o = lse
    lse_for_o = torch.where(lse_for_o == float("-inf"), float("inf"), lse_for_o)
    out = torch.bmm(torch.exp(logits - lse_for_o.unsqueeze(-1)), gathered[..., :d_v])
    lse = torch.where(lse == float("-inf"), float("inf"), lse)
    return out, max_logits, lse


def fused_prefill(enable_q_norm, rms_norm_eps, token_positions, is_rope_neox_style, rope_dim,
                  cos_sin_cache, n_wv_group, num_per_channels, use_tma_aligned_col_major_sf,
                  round_sf, use_packed_ue8m0, q, kv, indices, sm_scale, d_v=512,
                  attn_sink=None, topk_length=None):
    if (is_rope_neox_style or rope_dim != 64 or num_per_channels != 32 or d_v != 512
            or not (use_tma_aligned_col_major_sf and round_sf and use_packed_ue8m0)):
        raise NotImplementedError("FlashMLA fused prefill supports the pinned V4.1 configuration only")
    s_q, h_q, d_qk = q.shape
    if kv.shape[1] != 1 or indices.shape[:2] != (s_q, 1) or h_q % n_wv_group or q.stride(1) != d_qk:
        raise ValueError("FlashMLA fused prefill shape/layout contract violated")
    wv_group_size = h_q // n_wv_group
    q_heads = _unpermute_q(q, h_q, d_qk)
    out = torch.empty(s_q, h_q, d_v, device=q.device, dtype=torch.float32)
    max_logits = torch.empty(s_q, h_q, device=q.device, dtype=torch.float32)
    lse = torch.empty(s_q, h_q, device=q.device, dtype=torch.float32)
    chunk = max(1, _CHUNK_ELEMENTS // (indices.shape[-1] * d_qk))
    for start in range(0, s_q, chunk):
        rows = slice(start, min(s_q, start + chunk))
        q_chunk = q_heads[rows].float()
        norm_scale = None
        if enable_q_norm:
            norm_scale = torch.rsqrt(q_chunk.square().sum(-1) / d_qk + rms_norm_eps)
        q_chunk = q_chunk.clone()
        _rope_interleaved(q_chunk, cos_sin_cache, token_positions[rows], rope_dim, conjugate=False)
        q_chunk = q_chunk.to(torch.bfloat16)
        out[rows], max_logits[rows], lse[rows] = _sparse_attention(
            q_chunk, kv, indices[rows], sm_scale, d_v, attn_sink,
            None if topk_length is None else topk_length[rows], norm_scale)
    _rope_interleaved(out, cos_sin_cache, token_positions, rope_dim, conjugate=True)
    # [s_q, n, wv, d_v/32, 32] -> [s_q, n, d_v/32, wv, 32]: the permute_wv_proj order.
    permuted = out.view(s_q, n_wv_group, wv_group_size, d_v // 32, 32).transpose(2, 3)
    flat = permuted.reshape(s_q, n_wv_group * wv_group_size * d_v)
    out_fp8, sf = per_token_cast(flat, "e4m3", 32, use_tma_aligned_col_major_sf=True,
                                 round_sf=True, use_packed_ue8m0=True)
    packs = wv_group_size * d_v // 128
    aligned = sf.stride(1)
    # Same storage as the vendor allocate_scale_factor(...).transpose(0, 1) layout.
    out_sf = sf.as_strided((s_q, n_wv_group, packs), (1, packs * aligned, aligned))
    return out_fp8.view(s_q, n_wv_group, wv_group_size * d_v), out_sf, max_logits, lse


_LOGITS_CHUNK_ELEMENTS = 1 << 27  # bound the per-chunk [q, heads, kv] FP32 score tensor


def _dequant_mqa_operand(x, sf):
    """FP8 E4M3 or packed E2M1 [..., d(/2)] with per-row packed UE8M0 sf -> BF16 [..., d]."""
    from .gemm_emu import dequantize_mxfp4, dequantize_fp8

    if x.dtype == torch.int8:
        d = x.shape[-1] * 2
        # One int32 pack per row: 4 UE8M0 exponents -> gran_k = d / 4.
        return dequantize_mxfp4(x, sf.unsqueeze(-1), d // 4, torch.bfloat16)
    d = x.shape[-1]
    return dequantize_fp8(x, sf.unsqueeze(-1), 1, d // 4, torch.bfloat16)


def fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k, schedule_meta=None, *, output_dtype=torch.bfloat16):
    """Pinned DeepGEMM fp8_fp4_mqa_logits (SM100 contract) on HCU.

    logits[i, j] = sum_h weights[i, h] * relu(q[i, h] . kv[ks[i] + j]) for
    j < ke[i] - ks[i] (compressed layout, BF16); entries past a row's span are
    unspecified (vendor leaves them uncleaned; callers bound reads by length).
    """
    q_x, q_sf = q
    kv_x, kv_sf = kv
    if q_sf is None:
        raise NotImplementedError("HCU MQA logits implements the SM100 MX-scale contract only")
    seq_len, num_heads = q_x.shape[:2]
    q_deq = _dequant_mqa_operand(q_x, q_sf).float()          # [S, H, D]
    k_deq = _dequant_mqa_operand(kv_x, kv_sf).float()        # [N, D]
    w = weights.float()
    ks = cu_seq_len_k_start.long()
    span = cu_seq_len_k_end.long() - ks
    logits = torch.zeros(seq_len, max_seqlen_k, device=q_x.device, dtype=output_dtype)
    n_kv = k_deq.shape[0]
    cols = max(1, min(max_seqlen_k, _LOGITS_CHUNK_ELEMENTS // max(1, seq_len * num_heads)))
    for c0 in range(0, max_seqlen_k, cols):
        c1 = min(max_seqlen_k, c0 + cols)
        offs = torch.arange(c0, c1, device=q_x.device)
        pos = (ks[:, None] + offs[None, :]).clamp(0, n_kv - 1)            # [S, C]
        k_rows = k_deq.index_select(0, pos.flatten()).view(seq_len, c1 - c0, -1)
        score = torch.bmm(q_deq, k_rows.transpose(1, 2)).relu_()         # [S, H, C]
        chunk = torch.bmm(w.unsqueeze(1), score).squeeze(1)              # [S, C]
        valid = offs[None, :] < span[:, None]
        logits[:, c0:c1] = torch.where(valid, chunk, 0).to(output_dtype)
    return logits


def get_sparse_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, sparse_kv_block_indices,
                                   qk_dtype, sparse_block_kv, use_unaligned_ks=False):
    """Opaque schedule for fp8_fp4_sparse_mqa_logits. The HCU provider needs no scheduling,
    so it returns the inputs it consumes. Like the vendor metadata these are a snapshot taken
    now (device clones; nothing is read on host): callers may refill the buffers afterwards."""
    return {"ks": cu_seq_len_k_start.clone(), "ke": cu_seq_len_k_end.clone(), "num_kv_tokens": num_kv_tokens,
            "blocks": sparse_kv_block_indices.clone(), "block_kv": sparse_block_kv, "unaligned": use_unaligned_ks}


def fp8_fp4_sparse_mqa_logits(q, kv, weights, metadata, num_max_sparse_blocks, sparse_block_kv,
                              use_unaligned_ks=False, *, output_dtype=torch.bfloat16):
    """Pinned DeepGEMM fp8_fp4_sparse_mqa_logits on HCU (vendor tests/test_attention.py contract).

    logits[i, b*bs + o] = dense logit of token blocks[i, b]*bs + (ks[i] % bs if unaligned) + o.
    Entries outside [ks, ke) are unspecified upstream; they are zeroed here.
    """
    q_x, q_sf = q
    kv_x, kv_sf = kv
    if q_sf is None or not isinstance(metadata, dict):
        raise NotImplementedError("HCU sparse MQA logits needs MX scales and HCU metadata")
    if metadata["block_kv"] != sparse_block_kv or metadata["unaligned"] != use_unaligned_ks:
        raise ValueError("sparse MQA metadata does not match the call")
    seq_len, num_heads = q_x.shape[:2]
    blocks = metadata["blocks"].long()[:, :num_max_sparse_blocks]
    ks, ke = metadata["ks"].long(), metadata["ke"].long()
    base = torch.remainder(ks, sparse_block_kv) if use_unaligned_ks else torch.zeros_like(ks)
    offs = torch.arange(sparse_block_kv, device=q_x.device)
    tokens = (blocks[..., None] * sparse_block_kv + base[:, None, None] + offs).flatten(1)  # [S, T]
    valid = (tokens >= ks[:, None]) & (tokens < ke[:, None]) & (blocks >= 0).repeat_interleave(sparse_block_kv, 1)
    q_deq = _dequant_mqa_operand(q_x, q_sf).float()
    k_deq = _dequant_mqa_operand(kv_x, kv_sf).float()
    w = weights.float()
    n_out = num_max_sparse_blocks * sparse_block_kv
    logits = torch.zeros(seq_len, n_out, device=q_x.device, dtype=output_dtype)
    cols = max(1, min(n_out, _LOGITS_CHUNK_ELEMENTS // max(1, seq_len * num_heads)))
    safe = tokens.clamp(0, k_deq.shape[0] - 1)
    for c0 in range(0, n_out, cols):
        c1 = min(n_out, c0 + cols)
        k_rows = k_deq.index_select(0, safe[:, c0:c1].flatten()).view(seq_len, c1 - c0, -1)
        score = torch.bmm(q_deq, k_rows.transpose(1, 2)).relu_()
        chunk = torch.bmm(w.unsqueeze(1), score).squeeze(1)
        logits[:, c0:c1] = torch.where(valid[:, c0:c1], chunk, 0).to(output_dtype)
    return logits


def _split_index_cache(fused_kv_cache, is_fp4):
    """Index KV page = [page_kv x value bytes | page_kv x int32 packed sf] (vendor layout)."""
    num_pages, page_kv, _, width = fused_kv_cache.shape
    value_bytes = width - 4
    # Span only the page's own bytes: a layer's cache sits at an offset inside the shared
    # per-block buffer, so a full-stride row for the last page would run past the storage.
    flat = fused_kv_cache.view(torch.uint8).as_strided(
        (num_pages, page_kv * width), (fused_kv_cache.stride(0), 1))
    values = flat[:, :page_kv * value_bytes].reshape(num_pages, page_kv, value_bytes)
    sf = flat[:, page_kv * value_bytes:page_kv * width].reshape(num_pages, page_kv, 4).contiguous().view(torch.int32)
    values = values.view(torch.int8) if is_fp4 else values.view(torch.float8_e4m3fn)
    return values, sf.squeeze(-1)


def _paged_tokens(fused_kv_cache, block_table, token_pos, valid, is_fp4):
    """Dequantize index-cache tokens at logical positions token_pos[S, C] of query S.

    Block-table row i belongs to query i (vendor: block_table.size(0) == num_q_tokens; the
    varlen `indices` only group queries for scheduling and may hold the placeholder
    sentinel). Like the vendor kernel, pages are only looked up where `valid`
    (pos < context_len); other lanes read page 0 and must be masked by the caller.
    """
    values, sf = _split_index_cache(fused_kv_cache, is_fp4)
    num_pages, page_kv = values.shape[:2]
    pos = torch.where(valid, token_pos, 0)
    page_col = (pos // page_kv).clamp(max=block_table.shape[1] - 1)
    page = torch.where(valid, block_table.long().gather(1, page_col), 0).clamp(0, num_pages - 1)
    slot = torch.remainder(pos, page_kv)
    return _dequant_mqa_operand(values[page, slot], sf[page, slot]).float()   # [S, C, D]


def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices=None):
    """Opaque schedule; the HCU provider schedules nothing (shape-compatible placeholder)."""
    return torch.zeros((num_sms + 1, 2), dtype=torch.int32, device=context_lens.device)


def fp8_fp4_paged_mqa_logits(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
                             max_context_len, indices=None, *, output_dtype=torch.bfloat16):
    """logits[i, t] = sum_h w[i,h] relu(q[i,h] . k(request(i), t)) for t < context_lens[i] (SM100 varlen)."""
    q_x, q_sf = q
    if q_sf is None or indices is None or q_x.shape[1] != 1:
        raise NotImplementedError("HCU paged MQA logits implements the SM100 varlen (next_n=1, MX) contract")
    num_q, _, num_heads, _ = q_x.shape
    is_fp4 = q_x.dtype == torch.int8
    q_deq = _dequant_mqa_operand(q_x[:, 0], q_sf[:, 0]).float()
    w = weights.float()
    lens = context_lens.view(-1).long()
    logits = torch.zeros(num_q, max_context_len, device=q_x.device, dtype=output_dtype)
    cols = max(1, min(max_context_len, _LOGITS_CHUNK_ELEMENTS // max(1, num_q * num_heads)))
    for c0 in range(0, max_context_len, cols):
        c1 = min(max_context_len, c0 + cols)
        pos = torch.arange(c0, c1, device=q_x.device)[None, :].expand(num_q, -1)
        valid = pos < lens[:, None]
        k = _paged_tokens(fused_kv_cache, block_table, pos, valid, is_fp4)
        score = torch.bmm(q_deq, k.transpose(1, 2)).relu_()
        chunk = torch.bmm(w.unsqueeze(1), score).squeeze(1)
        logits[:, c0:c1] = torch.where(valid, chunk, 0).to(output_dtype)
    return logits


def get_paged_sparse_mqa_logits_metadata(context_lens, block_table, indices, page_kv, sparse_kv_block_indices,
                                         qk_dtype, sparse_block_kv):
    """Snapshot of the inputs (vendor metadata is computed at this call; upstream caches it across layers)."""
    return {"lens": context_lens.clone(), "block_table": block_table.clone(),
            "blocks": sparse_kv_block_indices.clone(), "block_kv": sparse_block_kv,
            "indices": indices.clone(), "page_kv": page_kv}


def fp8_fp4_paged_sparse_mqa_logits(q, fused_kv_cache, weights, metadata, num_max_sparse_blocks, sparse_block_kv, *, output_dtype=torch.bfloat16):
    """Candidate-block logits on the paged index cache (context starts at 0 per request)."""
    q_x, q_sf = q
    if q_sf is None or not isinstance(metadata, dict) or metadata["block_kv"] != sparse_block_kv:
        raise NotImplementedError("HCU paged sparse MQA logits needs MX scales and HCU metadata")
    num_q, _, num_heads, _ = q_x.shape
    is_fp4 = q_x.dtype == torch.int8
    q_deq = _dequant_mqa_operand(q_x[:, 0], q_sf[:, 0]).float()
    w = weights.float()
    lens = metadata["lens"].view(-1).long()
    blocks = metadata["blocks"].long()[:, :num_max_sparse_blocks]
    offs = torch.arange(sparse_block_kv, device=q_x.device)
    tokens = (blocks[..., None] * sparse_block_kv + offs).flatten(1)
    valid = (tokens >= 0) & (tokens < lens[:, None]) & (blocks >= 0).repeat_interleave(sparse_block_kv, 1)
    n_out = num_max_sparse_blocks * sparse_block_kv
    logits = torch.zeros(num_q, n_out, device=q_x.device, dtype=output_dtype)
    cols = max(1, min(n_out, _LOGITS_CHUNK_ELEMENTS // max(1, num_q * num_heads)))
    for c0 in range(0, n_out, cols):
        c1 = min(n_out, c0 + cols)
        k = _paged_tokens(fused_kv_cache, metadata["block_table"], tokens[:, c0:c1], valid[:, c0:c1], is_fp4)
        score = torch.bmm(q_deq, k.transpose(1, 2)).relu_()
        chunk = torch.bmm(w.unsqueeze(1), score).squeeze(1)
        logits[:, c0:c1] = torch.where(valid[:, c0:c1], chunk, 0).to(output_dtype)
    return logits


_V41_FP8_TOKEN_BYTES = 528   # 512 E4M3 + 16 UE8M0 (tile 32)
_V41_FP4_TOKEN_BYTES = 288   # 256 packed E2M1 + 32 E4M3 (tile 16)


def _dequant_main_cache_rows(k_cache, rows):
    """Gather and dequantize token rows (flat slot ids) of a V4.1 paged MLA cache -> FP32 [..., 512].

    Layouts follow vendor FlashMLA tests/quant.py: FP8 tokens are [512 E4M3 | 16 UE8M0 per 32];
    FP4 tokens are [256 packed E2M1 (low nibble even) | 32 E4M3 per 16].
    """
    from .quant_emu import unpack_e2m1

    num_blocks, block_size = k_cache.shape[:2]
    width = k_cache.shape[-1]
    # A view even when pages are strided inside the shared block buffer (no full-cache copy).
    raw = k_cache.view(torch.uint8).reshape(num_blocks, block_size, width)
    data = raw[torch.div(rows, block_size, rounding_mode="floor"), torch.remainder(rows, block_size)]
    if width == _V41_FP8_TOKEN_BYTES:
        values = data[..., :512].view(torch.float8_e4m3fn).float().unflatten(-1, (16, 32))
        scales = torch.pow(2.0, data[..., 512:].float() - 127)
    elif width == _V41_FP4_TOKEN_BYTES:
        values = unpack_e2m1(data[..., :256].contiguous()).unflatten(-1, (32, 16))
        scales = data[..., 256:].contiguous().view(torch.float8_e4m3fn).float()
    else:
        raise NotImplementedError(f"HCU MLA decode does not support {width}-byte KV tokens")
    return (values * scales.unsqueeze(-1)).flatten(-2)


def fused_decode(enable_q_norm, rms_norm_eps, token_positions, is_rope_neox_style, rope_dim, cos_sin_cache,
                 n_wv_group, num_per_channels, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0,
                 q, k_cache, indices_in_kvcache, sm_scale, d_v=512, attn_sink=None, topk_length=None,
                 extra_k_cache=None, extra_indices_in_kvcache=None, extra_topk_length=None):
    """FlashMLA decode with the pinned CUDA fused epilogue's FP32 O-RoPE.

    The vendor Python decode reference returns BF16 before rotating O, whereas
    core_attn/kernel.cuh::finish_tile keeps its accumulator in float through
    RoPE and FP8 conversion. Preserve that fused-kernel precision here.
    """
    if (is_rope_neox_style or rope_dim != 64 or num_per_channels != 32 or d_v != 512
            or not (use_tma_aligned_col_major_sf and round_sf and use_packed_ue8m0)):
        raise NotImplementedError("FlashMLA fused decode supports the pinned V4.1 configuration only")
    s_q, h_q, d_qk = q.shape
    wv_group_size = h_q // n_wv_group
    q_heads = _unpermute_q(q, h_q, d_qk).float()
    norm_scale = torch.rsqrt(q_heads.square().sum(-1) / d_qk + rms_norm_eps) if enable_q_norm else None
    q_heads = q_heads.clone()
    _rope_interleaved(q_heads, cos_sin_cache, token_positions, rope_dim, conjugate=False)
    q_heads = q_heads.to(torch.bfloat16).float()

    def scope(cache, idx, length):
        topk = idx.shape[-1]
        idx = idx.reshape(s_q, topk).long()
        # Slots past the cache are treated like -1 so a stale index can never fault.
        invalid = (idx < 0) | (idx >= cache.shape[0] * cache.shape[1])
        if length is not None:
            invalid |= torch.arange(topk, device=idx.device)[None, :] >= length.long()[:, None]
        kv = _dequant_main_cache_rows(cache, torch.where(invalid, 0, idx))
        return torch.nan_to_num(kv, nan=0.0), invalid

    kv, invalid = scope(k_cache, indices_in_kvcache, topk_length)
    if extra_k_cache is not None:
        kv1, invalid1 = scope(extra_k_cache, extra_indices_in_kvcache, extra_topk_length)
        kv, invalid = torch.cat((kv, kv1), 1), torch.cat((invalid, invalid1), 1)
    logits = torch.bmm(q_heads, kv.transpose(1, 2)) * sm_scale
    if norm_scale is not None:
        logits = logits * norm_scale.unsqueeze(-1)
    logits = logits.masked_fill(invalid[:, None, :], float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    out = torch.bmm(torch.exp(logits - lse.unsqueeze(-1)), kv[..., :d_v])
    if attn_sink is not None:
        out = out * (1.0 / (1.0 + torch.exp(attn_sink.float()[None, :] - lse))).unsqueeze(-1)
    lonely = lse == float("-inf")
    out = out.masked_fill(lonely.unsqueeze(-1), 0.0)
    lse = lse.masked_fill(lonely, float("inf"))
    _rope_interleaved(out, cos_sin_cache, token_positions, rope_dim, conjugate=True)
    permuted = out.view(s_q, n_wv_group, wv_group_size, d_v // 32, 32).transpose(2, 3)
    out_fp8, sf = per_token_cast(permuted.reshape(s_q, -1), "e4m3", 32, use_tma_aligned_col_major_sf=True,
                                 round_sf=True, use_packed_ue8m0=True)
    packs = wv_group_size * d_v // 128
    out_sf = sf.as_strided((s_q, n_wv_group, packs), (1, packs * sf.stride(1), sf.stride(1)))
    return out_fp8.view(s_q, n_wv_group, wv_group_size * d_v), out_sf, lse
