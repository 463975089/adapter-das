"""LightOp providers with the pinned DeepInfer rounding and layout contracts.

The unsupported quantization formats retain the named compatibility provider;
usage reports distinguish native calls from those compatibility calls.
"""

import torch

_COUNTS = {}


def _count(name):
    _COUNTS[name] = _COUNTS.get(name, 0) + 1


def usage():
    return dict(_COUNTS)


def hc_projection(x, weight, eps):
    import lightop

    result = lightop.hc_mix_stats(x.contiguous(), weight.contiguous(), eps)
    if result is None:
        _count('mhc.compatibility')
        value = x.float()
        return (value @ weight.float().T) * torch.rsqrt(value.square().mean(-1) + eps)[:, None]
    _count('mhc.native')
    return result


def mega_mhc(*args, **kwargs):
    from .mhc import mega_mhc as implementation

    return implementation(*args, **kwargs, mix_backend='lightop')


def bf16_mega_gate(x, weight, *args, **kwargs):
    # Preserve the pinned FP32 gate projection; accelerate its routing stage.
    return moe_topk_gate(x.float() @ weight.float().T, *args, **kwargs)


def topk(input, topk, sorted=False, begin=None, end=None, indices_type=torch.int64,
         sorted_index=False, hint=None, output_idx=None, output_idx_offset=None,
         idx_oob_fill_value=2147483647, value_oob_fill_value=float('-inf'),
         return_value=True, abort_when_nan_found=True):
    from .deepselect_emu import _aligned, topk as compatibility_topk
    from lightop.fuse_topk_transform import topk_transform_ragged_v2

    if (topk not in (512, 1024, 2048) or input.numel() == 0 or begin is not None or hint is not None
            or not abort_when_nan_found):
        _count('topk.compatibility')
        return compatibility_topk(input, topk, sorted, begin, end, indices_type,
                                  sorted_index, hint, output_idx, output_idx_offset,
                                  idx_oob_fill_value, value_oob_fill_value,
                                  return_value, abort_when_nan_found)
    if (input.ndim != 2 or not input.is_cuda or input.stride(-1) != 1
            or input.dtype not in (torch.float32, torch.bfloat16)
            or indices_type not in (torch.int32, torch.int64)):
        raise ValueError('Unsupported native DeepSelect input or indices dtype')
    if sorted and input.dtype != torch.float32:
        raise NotImplementedError('Pinned sorted values require FP32')
    rows, width = input.shape
    lengths = (torch.full((rows,), width, dtype=torch.int32, device=input.device)
               if end is None else end)
    if lengths.shape != (rows,) or lengths.dtype != torch.int32 or lengths.device != input.device:
        raise ValueError('DeepSelect end must be an int32 device vector')
    column = torch.arange(width, device=input.device)
    torch._assert_async(((lengths >= 0) & (lengths <= width)).all(), 'DeepSelect end out of range')
    nan = (torch.isnan(input) & (column[None] < lengths[:, None])).any(1) & (lengths > topk)
    torch._assert_async((~nan).all(), 'DeepSelect: NaN in a long input row')
    indices = torch.empty((rows, topk), device=input.device, dtype=torch.int32)
    zeros = torch.zeros(rows, device=input.device, dtype=torch.int32)
    # The native API excludes NaNs; the pinned short-row path returns every ID
    # without comparing values, so sanitize only those rows for the selection.
    scores = torch.where((lengths <= topk)[:, None], 0., input.float()).contiguous()
    topk_transform_ragged_v2(scores, lengths.contiguous(), out_offsets=zeros, out_indices=indices)
    valid = indices >= 0
    safe = indices.clamp(min=0, max=max(0, width - 1)).long()
    values = input.gather(1, safe)
    # Native short rows can be unordered; DeepSelect requires original order.
    order = torch.argsort(torch.where(valid, safe, width), dim=1, stable=True)
    safe, valid, values = (tensor.gather(1, order) for tensor in (safe, valid, values))
    if sorted:
        ranks = torch.argsort(torch.where(valid, values, -float('inf')), dim=1,
                              descending=True, stable=True)
        ranks = torch.where((lengths > topk)[:, None], ranks,
                            torch.arange(topk, device=input.device).expand(rows, -1))
        safe, valid, values = (tensor.gather(1, ranks) for tensor in (safe, valid, values))
    if output_idx_offset is not None:
        if (output_idx_offset.shape != (rows,) or output_idx_offset.dtype != torch.int32
                or output_idx_offset.device != input.device):
            raise ValueError('DeepSelect output offsets require an int32 device vector')
        safe = safe + output_idx_offset[:, None]
    mapped = torch.where(valid, safe, idx_oob_fill_value)
    values = torch.where(valid, values, value_oob_fill_value)
    if sorted_index:
        # DeepSelect interface.py / common_parts.cuh keep short-row padding
        # at the end, including a negative sentinel. Sorting the sentinel
        # itself moves -1 into the valid prefix consumed by sparse planners.
        order = torch.argsort(torch.where(valid, mapped, torch.iinfo(mapped.dtype).max),
                              dim=1, stable=True)
        mapped = mapped.gather(1, order)
        values = values.gather(1, order)
    if output_idx is None:
        output_idx = _aligned(rows, topk, indices_type, input.device)
    elif (output_idx.shape != (rows, topk) or output_idx.dtype != indices_type
          or output_idx.device != input.device or output_idx.stride(-1) != 1):
        raise ValueError('DeepSelect output indices mismatch')
    output_idx.copy_(mapped)
    output_val = None
    if return_value:
        output_val = _aligned(rows, topk, input.dtype, input.device)
        output_val.copy_(values)
    _count('topk.native')
    return output_val, output_idx


def fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end,
                       max_seqlen_k, schedule_meta=None, *, output_dtype=torch.bfloat16):
    from .mla_attention_emu import _dequant_mqa_operand
    from .mla_attention_emu import fp8_fp4_mqa_logits as compatibility_logits
    from lightop.gemmopt import fp8_fp4_mqa_logits as native_logits

    q_data, q_sf = q
    k_data, k_sf = kv
    q_width = q_data.shape[-1] * (2 if q_data.dtype == torch.int8 else 1)
    if (q_sf is None or q_width != 128 or k_data.dtype != torch.int8 or k_data.shape[-1] != 64
            or k_sf.dtype != torch.int32 or k_sf.numel() != k_data.shape[0]
            or max_seqlen_k <= 0 or max_seqlen_k % 4):
        _count('mqa.compatibility')
        return compatibility_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end,
                                    max_seqlen_k, schedule_meta, output_dtype=output_dtype)
    # Keep the caller's block-scaled Q dequantization; the native entry consumes
    # BF16 Q and packed FP4 K directly. No activation scale is folded twice.
    query = _dequant_mqa_operand(q_data, q_sf).contiguous()
    output = native_logits((query, None), (k_data.contiguous(), k_sf.reshape(-1).contiguous()),
                           weights.float().contiguous(), cu_seq_len_k_start.contiguous(),
                           cu_seq_len_k_end.contiguous(), False, max_seqlen_k)
    width = torch.arange(max_seqlen_k, device=output.device)
    valid = width[None] < (cu_seq_len_k_end - cu_seq_len_k_start)[:, None]
    _count('mqa.native')
    return torch.where(valid, output, 0.).to(output_dtype)


def per_token_cast(x, fmt, num_per_channels, x_block_size=None,
                   use_tma_aligned_col_major_sf=False, round_sf=False,
                   use_packed_ue8m0=False, use_e4m3_sf=False,
                   stochastic_cast=False, out=None):
    from .quant_emu import per_token_cast as compatibility_cast

    if (fmt != 'e4m3' or num_per_channels not in (32, 128) or not round_sf
            or not use_packed_ue8m0 or use_e4m3_sf or stochastic_cast
            or x_block_size is not None or isinstance(x, tuple)):
        _count('quant.compatibility')
        return compatibility_cast(x, fmt, num_per_channels, x_block_size,
                                  use_tma_aligned_col_major_sf, round_sf,
                                  use_packed_ue8m0, use_e4m3_sf, stochastic_cast, out)
    from lightop.quant import per_token_group_quant_fp8

    if (x.ndim != 2 or not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float32)
            or x.shape[1] % 128 or x.shape[1] <= 0 or x.stride(-1) != 1):
        raise ValueError('LightOp FP8 requires dense BF16/FP32 rows, K divisible by 128')
    m, k = x.shape
    groups = k // num_per_channels
    width = (groups + 3) // 4
    if out is None:
        q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
        sf = (torch.empty_strided((m, width), (1, (m + 3) // 4 * 4), dtype=torch.int32, device=x.device)
              if use_tma_aligned_col_major_sf else torch.empty((m, width), dtype=torch.int32, device=x.device))
    else:
        if use_tma_aligned_col_major_sf:
            raise NotImplementedError('Pinned cast does not accept column-major output scale buffers')
        q, sf = out
        if (q.shape != (m, k) or q.dtype != torch.float8_e4m3fn or not q.is_contiguous()
                or sf.shape != (m, width) or sf.dtype != torch.int32
                or sf.stride(-1) != 1 or q.device != x.device or sf.device != x.device):
            raise ValueError('LightOp cast output buffers do not match the pinned ABI')
    if m:
        scales = torch.empty((m, groups), device=x.device, dtype=torch.float32)
        per_token_group_quant_fp8(x.contiguous(), num_per_channels, eps=1e-4,
                                 out_q=q, out_scale=scales, use_ue8m0=True)
        exponent = (scales.view(torch.int32) >> 23).to(torch.uint8)
        if use_tma_aligned_col_major_sf:
            raw = sf.T.view(torch.uint8)
            for lane in range(4):
                values = exponent[:, lane::4]
                raw[:values.shape[1], lane::4].copy_(values.T)
        else:
            sf.view(torch.uint8)[:, :groups].copy_(exponent)
        _count('quant.native')
    return q, sf


def norm(x, out, weight, residual, residual_out, sf, extra_sf, origin, *,
         has_residual, eps, out_scale, do_fp8_cast, num_per_channels,
         col_packed, round_sf, use_packed_ue8m0, return_origin_out, shared_sf_block_m):
    import lightop

    if shared_sf_block_m:
        raise NotImplementedError('LightOp norm candidate does not cover mega-MoE shared SF')
    value = x + residual if has_residual else x
    if has_residual:
        residual_out.copy_(value)
    # The upstream residual sum rounds to input dtype before RMSNorm. Native
    # fused_add_rms_norm has a different FP32-sum policy, so use plain RMSNorm.
    matrix = value.float().reshape(-1, x.shape[-1]).contiguous()
    result = torch.empty_like(matrix)
    lightop.rms_norm(matrix, (weight.float() * out_scale).contiguous(), eps, out=result)
    result = result.view_as(x)
    if return_origin_out:
        origin.copy_(result)
    if not do_fp8_cast:
        out.copy_(result)
    else:
        q, packed = per_token_cast(result.to(x.dtype).reshape(-1, x.shape[-1]), 'e4m3',
                                  num_per_channels, use_tma_aligned_col_major_sf=col_packed,
                                  round_sf=round_sf, use_packed_ue8m0=use_packed_ue8m0)
        out.copy_(q.view_as(out))
        if use_packed_ue8m0:
            raw = packed.contiguous().view(torch.uint8).reshape(*x.shape[:-1], -1)
            if col_packed:
                sf.copy_(raw.view_as(sf))
            else:
                sf.copy_(raw[..., :sf.shape[-1]])
        else:
            sf.copy_(packed.view_as(sf))
    _count('norm.native')


def swiglu_forward_and_per_token_cast(x, fmt, num_per_channels, psum_num_tokens_per_expert=None,
                                      topk_weights=None, routed_scaling_factor=None,
                                      use_tma_aligned_col_major_sf=False, round_sf=False,
                                      use_packed_ue8m0=False, clamp_value=None, clamped_count=None):
    import lightop

    if clamped_count is not None or fmt not in ('e4m3', 'bf16', 'fp32'):
        raise NotImplementedError('Unsupported LightOp SwiGLU format or clamp counting')
    if x.ndim != 2 or not x.is_contiguous() or x.shape[1] % 2:
        raise ValueError('SwiGLU requires a contiguous [M,2H] matrix')
    values = x.float()
    h = x.shape[1] // 2
    if clamp_value is not None:
        values = torch.cat((values[:, :h].clamp(max=clamp_value),
                            values[:, h:].clamp(-clamp_value, clamp_value)), -1)
    result = torch.empty((x.shape[0], h), device=x.device, dtype=torch.float32)
    if x.shape[0]:
        lightop.fuse_silu_and_mul(values, result)
        _count('swiglu.native')
    if topk_weights is not None:
        weights = topk_weights.float()
        if routed_scaling_factor is not None:
            weights = weights * routed_scaling_factor
        result *= weights[:, None]
    if fmt != 'e4m3':
        return result.to(torch.bfloat16 if fmt == 'bf16' else torch.float32)
    return per_token_cast(result, fmt, num_per_channels,
                          use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
                          round_sf=round_sf, use_packed_ue8m0=use_packed_ue8m0)


def swiglu_forward(x, fmt, psum_num_tokens_per_expert=None, topk_weights=None,
                   routed_scaling_factor=None, clamp_value=None, clamped_count=None):
    return swiglu_forward_and_per_token_cast(
        x, fmt, None, psum_num_tokens_per_expert, topk_weights, routed_scaling_factor,
        clamp_value=clamp_value, clamped_count=clamped_count)


def moe_topk_gate(logits, num_topk, use_shared_as_routed, num_shared_experts, routed_scaling_factor, ep_rank,
                  scoring_func='sqrtsoftplus', mask=None, bias=None, image_bias=None, image_token_mask=None,
                  fix_routing_mask=None, to_physical_map=None, logical_count=None, unmapped_topk_idx=None,
                  force_random=None, out=None, scores=None):
    import lightop

    if scoring_func != 'sqrtsoftplus' or scores is not None:
        raise NotImplementedError('Unsupported LightOp routing policy')
    tokens, routed = logits.shape
    if force_random is not None:
        if isinstance(force_random, torch.Tensor):
            if force_random.dtype != torch.bool or force_random.shape != (tokens,):
                raise ValueError('Routing metadata requires a boolean vector per token')
        elif force_random is not False:
            raise NotImplementedError('Forced random routing is not enabled by this candidate')
        # Match the existing HCU normal-routing policy. Upstream supplies this
        # vector even with FORCE_LOAD_BALANCE=0; the platform guard still rejects
        # that benchmark switch. Do not synchronize a metadata tensor on host.
        _count('router.force_random_metadata_compatibility')
    shared = num_shared_experts if use_shared_as_routed else 0
    if not tokens:
        return (torch.empty((0, num_topk + shared), device=logits.device, dtype=torch.int64),
                torch.empty((0, num_topk + shared), device=logits.device, dtype=torch.float32)) if out is None else out
    data = logits.float().contiguous()
    rank_bias = torch.zeros(routed, device=logits.device) if bias is None else bias.float().contiguous()
    if routed == 128:
        from hcu_adapter.kernels.draft_router_triton import sqrtsoftplus_topk

        def route(rank_bias):
            return sqrtsoftplus_topk(data, rank_bias, num_topk, routed_scaling_factor)

        _count('router.draft_triton')
    elif routed in (256, 384):
        def route(rank_bias):
            return lightop.moe_fused_gate_sqrtsoftplus_v41_hcu(
                data, rank_bias, num_topk, 0, True, routed_scaling_factor, True)

        _count('router.native')
    else:
        raise NotImplementedError(f'LightOp routing unsupported expert count: {routed}')
    weights, selected = route(rank_bias)
    selected = selected.long()
    if image_token_mask is not None:
        image_weights, image_selected = route(image_bias.float().contiguous())
        selected = torch.where(image_token_mask[:, None], image_selected.long(), selected)
        weights = torch.where(image_token_mask[:, None], image_weights, weights)
    if fix_routing_mask is not None and unmapped_topk_idx is not None:
        selected = torch.where(fix_routing_mask[:, None], unmapped_topk_idx.long(), selected)
        picked = torch.nn.functional.softplus(data).sqrt().gather(1, selected.clamp_min(0))
        weights = picked / (picked.sum(-1, keepdim=True) + 1e-20) * routed_scaling_factor
    active = torch.ones(tokens, device=logits.device, dtype=torch.bool) if mask is None else mask
    if unmapped_topk_idx is not None:
        unmapped_topk_idx.copy_(torch.where(active[:, None], selected, -1))
    if shared:
        ids = torch.arange(routed, routed + shared, device=logits.device).expand(tokens, -1)
        selected = torch.cat((selected, ids), -1)
        weights = torch.cat((weights, torch.ones((tokens, shared), device=logits.device)), -1)
    if to_physical_map is not None:
        row = torch.arange(tokens, device=logits.device)[:, None]
        logical = selected.clamp_min(0)
        duplicate = (ep_rank + row * 23333) % logical_count.long()[logical]
        selected = to_physical_map.long()[logical, duplicate]
    selected, weights = torch.where(active[:, None], selected, -1), torch.where(active[:, None], weights, 0.)
    if out is not None:
        out[0].copy_(selected)
        out[1].copy_(weights)
        selected, weights = out
    return selected, weights


def apply_rotary(query, cos_sin_cache, key=None, positions=None, interleaved=False,
                 conjugate=False, seqlen_offset=0):
    import lightop

    for x in (query,) if key is None else (query, key):
        if x.shape[-1] != cos_sin_cache.shape[1] or x.ndim not in (3, 4):
            raise ValueError('Unsupported LightOp rotary shape')
        length = x.shape[-3]
        if positions is None:
            pos = torch.arange(length, device=x.device)
            if x.ndim == 4:
                pos = pos.repeat(x.shape[0])
        else:
            pos = positions.reshape(-1).long()
        work = x.float().reshape(-1, x.shape[-2], x.shape[-1]).contiguous()
        cache = cos_sin_cache.float().contiguous()
        if conjugate:
            half = cache.shape[1] // 2
            cache = torch.cat((cache[:, :half], -cache[:, half:]), -1)
        lightop.rotary_embedding_fuse(pos.long() + seqlen_offset, work, None,
                                      x.shape[-1], cache, not interleaved)
        x.copy_(work.view_as(x))
        _count('rope.native')
