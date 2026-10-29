"""Compatibility entry points only; numerical work lives in the named operator families."""

import torch

from .lightop_usage import _COUNTS, _count, usage


def hc_projection(x, weight, eps):
    from .mhc_lightop import hc_projection as implementation

    return implementation(x, weight, eps)


def mega_mhc(*args, **kwargs):
    from .mhc_lightop import mega_mhc as implementation

    return implementation(*args, **kwargs)


def bf16_mega_gate(x, weight, *args, **kwargs):
    # Preserve the pinned FP32 gate projection; accelerate its routing stage.
    from .moe_gate_lightop import bf16_mega_gate as implementation

    return implementation(x, weight, *args, **kwargs)


def topk(input, topk, sorted=False, begin=None, end=None, indices_type=torch.int64,
         sorted_index=False, hint=None, output_idx=None, output_idx_offset=None,
         idx_oob_fill_value=2147483647, value_oob_fill_value=float('-inf'),
         return_value=True, abort_when_nan_found=True):
    from .deepselect_lightop import topk as implementation

    return implementation(
        input, topk, sorted, begin, end, indices_type, sorted_index, hint, output_idx,
        output_idx_offset, idx_oob_fill_value, value_oob_fill_value, return_value,
        abort_when_nan_found)


def fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end,
                       max_seqlen_k, schedule_meta=None, *, output_dtype=torch.bfloat16):
    from .mqa_logits_lightop import packed_dense as implementation

    return implementation(
        q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k, schedule_meta,
        output_dtype=output_dtype)


def per_token_cast(x, fmt, num_per_channels, x_block_size=None,
                   use_tma_aligned_col_major_sf=False, round_sf=False,
                   use_packed_ue8m0=False, use_e4m3_sf=False,
                   stochastic_cast=False, out=None):
    from .quant_lightop import per_token_cast as implementation

    return implementation(
        x, fmt, num_per_channels, x_block_size, use_tma_aligned_col_major_sf, round_sf,
        use_packed_ue8m0, use_e4m3_sf, stochastic_cast, out)


def norm(x, out, weight, residual, residual_out, sf, extra_sf, origin, *,
         has_residual, eps, out_scale, do_fp8_cast, num_per_channels,
         col_packed, round_sf, use_packed_ue8m0, return_origin_out, shared_sf_block_m):
    from .norm_lightop import norm as implementation

    return implementation(
        x, out, weight, residual, residual_out, sf, extra_sf, origin, has_residual=has_residual,
        eps=eps, out_scale=out_scale, do_fp8_cast=do_fp8_cast, num_per_channels=num_per_channels,
        col_packed=col_packed, round_sf=round_sf, use_packed_ue8m0=use_packed_ue8m0,
        return_origin_out=return_origin_out, shared_sf_block_m=shared_sf_block_m)


def swiglu_forward_and_per_token_cast(x, fmt, num_per_channels, psum_num_tokens_per_expert=None,
                                      topk_weights=None, routed_scaling_factor=None,
                                      use_tma_aligned_col_major_sf=False, round_sf=False,
                                      use_packed_ue8m0=False, clamp_value=None, clamped_count=None):
    from .swiglu_lightop import swiglu_forward_and_per_token_cast as implementation

    return implementation(
        x, fmt, num_per_channels, psum_num_tokens_per_expert, topk_weights, routed_scaling_factor,
        use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0, clamp_value, clamped_count)


def swiglu_forward(x, fmt, psum_num_tokens_per_expert=None, topk_weights=None,
                   routed_scaling_factor=None, clamp_value=None, clamped_count=None):
    from .swiglu_lightop import swiglu_forward as implementation

    return implementation(
        x, fmt, psum_num_tokens_per_expert, topk_weights, routed_scaling_factor, clamp_value,
        clamped_count)


def moe_topk_gate(logits, num_topk, use_shared_as_routed, num_shared_experts, routed_scaling_factor, ep_rank,
                  scoring_func='sqrtsoftplus', mask=None, bias=None, image_bias=None, image_token_mask=None,
                  fix_routing_mask=None, to_physical_map=None, logical_count=None, unmapped_topk_idx=None,
                  force_random=None, out=None, scores=None):
    from .moe_gate_lightop import moe_topk_gate as implementation

    return implementation(
        logits, num_topk, use_shared_as_routed, num_shared_experts, routed_scaling_factor, ep_rank,
        scoring_func, mask, bias, image_bias, image_token_mask, fix_routing_mask, to_physical_map,
        logical_count, unmapped_topk_idx, force_random, out, scores)


def apply_rotary(query, cos_sin_cache, key=None, positions=None, interleaved=False,
                 conjugate=False, seqlen_offset=0):
    from .rope_lightop import apply_rotary as implementation

    return implementation(query, cos_sin_cache, key, positions, interleaved, conjugate, seqlen_offset)
