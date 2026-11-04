# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Torch/emu paged KV gather, quantization and scatter implementations."""

import torch
from .quant_emu import dequantize_groups


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
