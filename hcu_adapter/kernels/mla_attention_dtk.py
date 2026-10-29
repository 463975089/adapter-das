"""Native FlashMLA attention with the V4.1 fused FP32 Q/O epilogue contract.

Requires the pinned optional FP32 HCU API. Decode currently gathers the existing
V4.1 FP8/FP4 cache into BF16 before native attention; it does not reinterpret that
cache as FlashMLA's different MODEL1 packed format.
"""

import torch

from .mla_attention_emu import _dequant_main_cache_rows, _unpermute_q
from .rope_lightop import apply_rotary
from .quant_lightop import per_token_cast

_COUNTS = {'prefill': 0, 'decode': 0}


def usage():
    return dict(_COUNTS)


def _check(is_neox, rope_dim, num_per_channels, aligned, rounded, packed, q, d_v):
    if (is_neox or rope_dim != 64 or num_per_channels != 32 or d_v != 512
            or not (aligned and rounded and packed) or q.shape[1] not in (64, 128)
            or q.shape[2] != 512 or q.dtype != torch.bfloat16):
        raise NotImplementedError('Native fused MLA requires the pinned HCU V4.1 H64/H128 D512 configuration')


def _query(q, positions, cache, enable_norm, eps):
    heads = _unpermute_q(q, q.shape[1], q.shape[2]).float()
    scale = (torch.rsqrt(heads.square().sum(-1) / q.shape[2] + eps).contiguous()
             if enable_norm else None)
    # Preserve Q's BF16 RoPE stage followed by FP32 per-head logit scaling.
    # Applying RMS scaling to Q before its BF16 cast would change the contract.
    heads = heads.clone()
    if q.shape[0]:
        apply_rotary(heads[..., -64:], cache, positions=positions, interleaved=True)
    return heads.to(torch.bfloat16), scale


def _attention(q, kv, indices, scale, sink, length, q_scale):
    import flash_mla.cuda as provider

    if not hasattr(provider, 'sparse_prefill_fwd_fp32'):
        raise RuntimeError('Native fused MLA requires the pinned code-object-v4 FlashMLA FP32 extension')
    if not q.shape[0]:
        return (torch.empty_like(q, dtype=torch.float32),
                torch.empty(q.shape[:2], device=q.device, dtype=torch.float32),
                torch.empty(q.shape[:2], device=q.device, dtype=torch.float32))
    indices = torch.where((indices >= 0) & (indices < kv.shape[0]), indices, -1).contiguous()
    return provider.sparse_prefill_fwd_fp32(
        q, kv, indices, scale, 512, sink, length, q_scale)


def _finish(out, positions, cache, n_wv_group):
    m, heads, d_v = out.shape
    if heads % n_wv_group:
        raise ValueError('WV groups must divide the attention heads')
    if m:
        apply_rotary(out[..., -64:], cache, positions=positions, interleaved=True, conjugate=True)
    group_size = heads // n_wv_group
    permuted = out.view(m, n_wv_group, group_size, d_v // 32, 32).transpose(2, 3)
    quant, sf = per_token_cast(permuted.reshape(m, heads * d_v), 'e4m3', 32,
                              use_tma_aligned_col_major_sf=True, round_sf=True,
                              use_packed_ue8m0=True)
    packs = group_size * d_v // 128
    shaped_sf = sf.as_strided((m, n_wv_group, packs), (1, packs * sf.stride(1), sf.stride(1)))
    return quant.view(m, n_wv_group, group_size * d_v), shaped_sf


def fused_prefill(enable_q_norm, rms_norm_eps, token_positions, is_rope_neox_style, rope_dim,
                  cos_sin_cache, n_wv_group, num_per_channels, use_tma_aligned_col_major_sf,
                  round_sf, use_packed_ue8m0, q, kv, indices, sm_scale, d_v=512,
                  attn_sink=None, topk_length=None):
    _check(is_rope_neox_style, rope_dim, num_per_channels, use_tma_aligned_col_major_sf,
           round_sf, use_packed_ue8m0, q, d_v)
    q_heads, q_scale = _query(q, token_positions, cos_sin_cache, enable_q_norm, rms_norm_eps)
    out, maximum, lse = _attention(q_heads, kv, indices, sm_scale, attn_sink, topk_length, q_scale)
    quant, sf = _finish(out, token_positions, cos_sin_cache, n_wv_group)
    _COUNTS['prefill'] += 1
    return quant, sf, maximum, lse


def _gather(cache, indices, length):
    if indices.ndim not in (2, 3) or (indices.ndim == 3 and indices.shape[1] != 1):
        raise ValueError('Native decode indices require [tokens,topk] or [tokens,1,topk]')
    m, topk = indices.shape[0], indices.shape[-1]
    index = indices.reshape(m, topk).long()
    invalid = (index < 0) | (index >= cache.shape[0] * cache.shape[1])
    if length is not None:
        invalid |= torch.arange(topk, device=index.device)[None, :] >= length.long()[:, None]
    values = _dequant_main_cache_rows(cache, torch.where(invalid, 0, index))
    return torch.nan_to_num(values, nan=0.).to(torch.bfloat16), invalid


def fused_decode(enable_q_norm, rms_norm_eps, token_positions, is_rope_neox_style, rope_dim,
                 cos_sin_cache, n_wv_group, num_per_channels, use_tma_aligned_col_major_sf,
                 round_sf, use_packed_ue8m0, q, k_cache, indices_in_kvcache, sm_scale, d_v=512,
                 attn_sink=None, topk_length=None, extra_k_cache=None,
                 extra_indices_in_kvcache=None, extra_topk_length=None):
    _check(is_rope_neox_style, rope_dim, num_per_channels, use_tma_aligned_col_major_sf,
           round_sf, use_packed_ue8m0, q, d_v)
    q_heads, q_scale = _query(q, token_positions, cos_sin_cache, enable_q_norm, rms_norm_eps)
    values, invalid = _gather(k_cache, indices_in_kvcache, topk_length)
    if extra_k_cache is not None:
        extra, extra_invalid = _gather(extra_k_cache, extra_indices_in_kvcache, extra_topk_length)
        values, invalid = torch.cat((values, extra), 1), torch.cat((invalid, extra_invalid), 1)
    m, width, dim = values.shape
    bank_indices = torch.arange(m * width, device=q.device, dtype=torch.int32).view(m, width)
    bank_indices = torch.where(invalid, -1, bank_indices).unsqueeze(1)
    out, _, lse = _attention(q_heads, values.reshape(m * width, 1, dim), bank_indices,
                             sm_scale, attn_sink, None, q_scale)
    quant, sf = _finish(out, token_positions, cos_sin_cache, n_wv_group)
    _COUNTS['decode'] += 1
    return quant, sf, lse
