"""Fused attention-row transform + paged KV store, mirroring the pinned CUDA kernel.

Stage semantics follow tilelang_ops/cuda/fused_transform.py @774ce24: each
stage rounds to BF16; the norm stage reuses the norm_tl provider so the fused
and unfused paths share one reduction order; RoPE is FP32
fma(x0, cos, -(x1 * sin)); the cast quantizes the BF16 stage output with
hcu_adapter.kernels.quant (UE8M0 multiplier in BF16, E4M3-scale path in FP32). No
deepinfer imports.
"""

import torch

from .quant_emu import dequantize_groups, quantize_groups


def _norm(x, weight, eps, backend):
    from .elementwise import norm

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    # Unused operands get real tensors as placeholders; the kernel never reads them.
    norm(x, out, weight, x, out, x, x, out, backend=backend, has_residual=False, eps=eps,
         out_scale=1.0, do_fp8_cast=False, num_per_channels=0, col_packed=False,
         round_sf=False, use_packed_ue8m0=False, return_origin_out=False, shared_sf_block_m=0)
    return out


def _rope(x, positions, cos, sin, rotary_dim, offset, is_neox, conjugate, backend):
    if backend == 'lightop':
        from .rope_lightop import apply_rotary

        out = x.clone(memory_format=torch.contiguous_format)
        apply_rotary(out[..., offset:offset + rotary_dim], torch.cat((cos, sin), -1),
                     positions=positions, interleaved=not is_neox, conjugate=conjugate)
        return out
    half = rotary_dim // 2
    part = x[..., offset:offset + rotary_dim]
    if is_neox:
        x0, x1 = part[..., :half].float(), part[..., half:].float()
    else:
        x0, x1 = part[..., 0::2].float(), part[..., 1::2].float()
    c = cos.index_select(0, positions.long()).unsqueeze(1)
    s = sin.index_select(0, positions.long()).unsqueeze(1)
    if conjugate:
        y0 = torch.addcmul(x1 * s, x0, c)
        y1 = torch.addcmul(x1 * c, x0, -s)
    else:
        y0 = torch.addcmul(-(x1 * s), x0, c)
        y1 = torch.addcmul(x1 * c, x0, s)
    out = x.clone(memory_format=torch.contiguous_format)
    target = out[..., offset:offset + rotary_dim]
    if is_neox:
        target[..., :half] = y0.to(torch.bfloat16)
        target[..., half:] = y1.to(torch.bfloat16)
    else:
        target[..., 0::2] = y0.to(torch.bfloat16)
        target[..., 1::2] = y1.to(torch.bfloat16)
    return out


def _store(cache, slot_mapping, values, block_size):
    """Scatter values[n, width] by slot; slots < 0 or past the cache are skipped.

    Graph-safe (no host sync): skipped rows and duplicate slot-0 writers all
    write one deterministic row to slot 0 — the first valid slot-0 row if any,
    else slot 0's previous contents.
    """
    num_slots = cache.shape[0] * block_size
    valid = (slot_mapping >= 0) & (slot_mapping < num_slots)
    slots = torch.where(valid, slot_mapping, 0).long()
    zero = valid & (slots == 0)
    first_zero = zero.to(torch.int32).argmax().reshape(1)
    # index_select keeps the row lookup on device; tensor indexing would sync.
    fill = torch.where(zero.any(), values.index_select(0, first_zero)[0], cache[0, 0])
    rows = torch.where((valid & ~zero).unsqueeze(1), values, fill.unsqueeze(0))
    cache[slots // block_size, slots % block_size] = rows


def fused_transform(x, weight, positions, cos, sin, out_bf16, out_quant, out_sf,
                    cache_quant, cache_sf, slot_mapping, *, backend, do_norm, norm_eps, do_rope,
                    rotary_dim, rope_offset, is_neox, conjugate, cast_fmt, num_per_channels,
                    sf_format, return_bf16, bf16_stage, return_quant, store_cache, block_size):
    work = x
    if do_norm:
        work = _norm(work, weight, norm_eps, backend)
    if return_bf16 and bf16_stage == "norm":
        out_bf16.copy_(work)
    if do_rope:
        work = _rope(work, positions, cos, sin, rotary_dim, rope_offset, is_neox, conjugate, backend)
    if return_bf16 and bf16_stage == "transform":
        out_bf16.copy_(work)
    if not cast_fmt:
        return
    if (backend == 'lightop' and cast_fmt == 'e4m3' and sf_format == 'ue8m0'
            and num_per_channels in (32, 128) and work.shape[-1] % 128 == 0):
        from .quant_lightop import per_token_cast

        quant, packed = per_token_cast(work.reshape(-1, work.shape[-1]), 'e4m3',
                                      num_per_channels, round_sf=True, use_packed_ue8m0=True)
        quant = quant.view_as(work)
        groups = work.shape[-1] // num_per_channels
        sf = packed.contiguous().view(torch.uint8)[:, :groups].reshape(*work.shape[:-1], groups)
        sf_float = (sf.to(torch.int32) << 23).view(torch.float32)
    else:
        quant, sf, sf_float = quantize_groups(work, cast_fmt, num_per_channels, sf_format)
    if return_bf16 and bf16_stage == "cast_back":
        out_bf16.copy_(dequantize_groups(quant, sf_float, cast_fmt, num_per_channels))
    if return_quant:
        out_quant.copy_(quant)
        out_sf.copy_(sf)
    if store_cache:
        _store(cache_quant, slot_mapping, quant[:, 0], block_size)
        _store(cache_sf, slot_mapping, sf[:, 0, :cache_sf.shape[-1]], block_size)


def fused_transform_lightop(*args, **kwargs):
    """Native norm/RoPE/FP8 stages with the original paged-store layout."""
    kwargs['backend'] = 'lightop'
    return fused_transform(*args, **kwargs)


def load_paged(load_segments, kv, kv_sf, cache_value, cache_sf, *, block_size, mode,
               fmt, group, sf_format, valid_groups):
    """Gather [src_block, begin, end, dst] segments from a paged cache (graph-safe).

    mode "cast": dequantize to BF16 kv[total_k, D] as bf16(float(q) * sf) (pinned
    CUDA kernel); "index": copy quantized bytes and scales verbatim; "bf16": copy
    plain rows. src_block == -1 writes zeros. Rows not covered keep their contents.
    """
    if not load_segments.shape[0]:
        return
    seg = load_segments.long()
    offsets = torch.arange(block_size, device=seg.device)
    lengths = seg[:, 2] - seg[:, 1]
    valid = offsets < lengths[:, None]
    src_offset = (seg[:, 1, None] + offsets).clamp(0, block_size - 1)
    src_block = seg[:, 0, None].clamp_min(0).expand_as(src_offset)
    nonempty = (seg[:, 0] != -1)[:, None, None]
    total_k = kv.shape[0]
    # Invalid lanes are routed to a scratch row (index total_k) dropped after the scatter.
    dst = torch.where(valid, seg[:, 3, None] + offsets, total_k)

    def gather(cache):
        rows = cache[src_block, src_offset]
        return torch.where(nonempty, rows, torch.zeros((), dtype=rows.dtype, device=rows.device))

    def scatter(out, rows):
        flat = rows.reshape(-1, *rows.shape[2:])
        staged = torch.cat((out, out[:1]), 0)
        staged.index_copy_(0, dst.reshape(-1), flat.to(out.dtype))
        out.copy_(staged[:total_k])

    if mode == "bf16":
        scatter(kv, gather(cache_value))
        return
    values = gather(cache_value.view(torch.uint8)).view(cache_value.dtype)
    scales = gather(cache_sf.view(torch.uint8) if cache_sf.dtype == torch.float8_e4m3fn else cache_sf)
    if mode == "index":
        scatter(kv.view(torch.uint8), values.view(torch.uint8))
        if kv_sf.dim() == 1:
            scales = scales.squeeze(-1)
        scatter(kv_sf.view(torch.uint8) if kv_sf.dtype == torch.float8_e4m3fn else kv_sf, scales)
        return
    scales = scales[..., :valid_groups]
    if sf_format == "ue8m0":
        sf_float = (scales.to(torch.int32) << 23).view(torch.float32)
    else:
        sf_float = scales.view(torch.float8_e4m3fn).float() if sf_format == "e4m3" else scales.float()
    scatter(kv, dequantize_groups(values, sf_float, fmt, group))


def _store_quantize(kv, fmt, group, sf_format, sf_width):
    """Quantize BF16 rows exactly as tilelang_ops/cuda/store_kv_cache.py @774ce24.

    Differs from the fused transform: UE8M0 scales use division (amax / max),
    FP8 multiplies in FP32, FP4-UE8M0 multiplies the BF16 value by a BF16-rounded
    power of two, and padded scale slots beyond the valid groups are zero.
    """
    from .quant_emu import FLOAT8E4M3_MIN_POSITIVE, _narrow

    blocks = kv.unflatten(-1, (-1, group))
    amax = blocks.abs().amax(-1).float()
    finfo_max = 448.0 if fmt == "e4m3" else 6.0
    if sf_format == "e4m3":
        sf = (amax.clamp_min(FLOAT8E4M3_MIN_POSITIVE * finfo_max) * (1.0 / finfo_max)).clamp_max(
            448.0).to(torch.float8_e4m3fn)
        scaled = blocks.float() * (1.0 / sf.float()).unsqueeze(-1)
    else:
        amax = amax.clamp_min(1e-4 if fmt == "e4m3" else 6.0 * 2.0 ** -126)
        bits = (amax / finfo_max).view(torch.int32)
        biased = ((bits - 1) >> 23) + 1
        multiplier = ((254 - biased) << 23).view(torch.float32)
        if fmt == "e4m3":
            scaled = blocks.float() * multiplier.unsqueeze(-1)
        else:
            scaled = (blocks * multiplier.to(torch.bfloat16).unsqueeze(-1)).float()
        sf = biased.to(torch.uint8) if sf_format == "ue8m0" else (biased << 23).view(torch.float32)
    if sf.shape[-1] < sf_width:
        sf = torch.nn.functional.pad(sf.view(torch.uint8) if sf.dtype == torch.float8_e4m3fn else sf,
                                     (0, sf_width - sf.shape[-1]))
    return _narrow(scaled.flatten(-2), fmt), sf


def store_paged(kv, cache_value, cache_sf, slot_mapping, *, block_size, fmt, group, sf_format):
    """Store BF16 rows kv[n, D] into a paged cache by slot (-1 / out of range skipped)."""
    if not kv.shape[0]:
        return
    if fmt is None:
        _store(cache_value, slot_mapping, kv.to(cache_value.dtype), block_size)
        return
    quant, sf = _store_quantize(kv, fmt, group, sf_format, cache_sf.shape[-1])
    _store(cache_value, slot_mapping, quant, block_size)
    _store(cache_sf, slot_mapping, sf.view(cache_sf.dtype) if sf.dtype != cache_sf.dtype else sf, block_size)
