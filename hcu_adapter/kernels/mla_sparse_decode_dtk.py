# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Native sparse decode over vLLM's page-flat 584-byte fp8_ds_mla caches.

Only selectors/Q-head padding are materialized. KV bytes stay in place; no KV
BF16 gather and no numerical reference implementation are used in production.
"""
from __future__ import annotations

import torch


SUPPORTED_ARCHES = frozenset({'gfx936', 'gfx938'})


def _supported_device(q):
    return (q.device.type == 'cuda' and torch.version.hip is not None
            and torch.cuda.get_device_properties(q.device).gcnArchName.split(':')[0]
            in SUPPORTED_ARCHES)


def _cache_supported(cache, q):
    # Page pitch may carry interleaved layers/padding: accept any outer stride
    # that still covers a dense page, and pass it through to the vendor.
    return (isinstance(cache, torch.Tensor) and cache.ndim == 3
            and cache.shape[0] > 0 and cache.shape[1] in (32, 64, 128)
            and cache.shape[2] == 584 and cache.dtype == torch.uint8
            and cache.device == q.device and cache.stride(2) == 1
            and cache.stride(1) == 584
            and cache.stride(0) >= cache.shape[1] * 584)


def _selectors_supported(dense, lengths, flat, indptr, q):
    rows = q.shape[0]
    if flat is not None or indptr is not None:
        # v0.33 core allocates CSR capacity = rows * maximum row width, even
        # when indptr[-1] is smaller. This host-known capacity bounds repacking.
        return (flat is not None and indptr is not None and flat.ndim == 1
                and indptr.shape == (rows + 1,) and flat.numel() % rows == 0
                and flat.dtype == indptr.dtype == torch.int32
                and flat.device == indptr.device == q.device)
    return (dense is not None and dense.ndim in (2, 3)
            and dense.shape[0] == rows and (dense.ndim == 2 or dense.shape[1] == 1)
            and dense.dtype == torch.int32 and dense.device == q.device
            and lengths is not None and lengths.shape == (rows,)
            and lengths.dtype == torch.int32 and lengths.device == q.device)


def _selectors(dense, lengths, flat, indptr, rows, num_slots):
    """CSR to padded dense, then compact invalid slots before vendor lengths.

    Never read indptr[-1] or any dynamic length on the host. Keep duplicate
    slots, and use the actual indptr offsets rather than reshaping CSR storage.
    """
    if flat is not None:
        width = flat.numel() // rows
        columns = torch.arange(width, device=flat.device)
        offsets = indptr[:-1, None].long() + columns[None, :]
        live = columns[None, :] < (indptr[1:] - indptr[:-1])[:, None]
        if flat.numel():
            dense = flat[offsets.clamp(0, flat.numel() - 1)]
            live = live & (offsets >= 0) & (offsets < flat.numel())
        else:
            dense = torch.empty((rows, 0), dtype=torch.int32, device=flat.device)
    else:
        dense = dense.reshape(rows, -1)
        width = dense.shape[1]
        columns = torch.arange(width, device=dense.device)
        live = columns[None, :] < lengths[:, None]
    valid = live & (dense >= 0) & (dense < num_slots)
    counts = valid.sum(-1, dtype=torch.int32)
    sentinel = torch.iinfo(torch.int32).max
    compact = torch.where(valid, dense, sentinel).sort(dim=-1).values
    padded_width = max(64, (width + 63) // 64 * 64)
    padded = torch.full((rows, 1, padded_width), -1, dtype=torch.int32, device=dense.device)
    padded[:, 0, :width] = torch.where(compact == sentinel, -1, compact)
    return padded, counts


def try_decode(*, q, kv_cache, swa_k_cache, swa_only, topk_indices, topk_lens,
               swa_indices, swa_lens, swa_ragged_indices, swa_ragged_indptr,
               topk_ragged_indices, topk_ragged_indptr, attn_sink, scale,
               head_dim, nope_head_dim, rope_head_dim, output,
               extra_cache_nan_free=False, adaptive_splits=False,
               inv_rope_positions=None, inv_rope_cos_sin_cache=None,
               output_mxfp8=None, provider, metadata_factory, epilogue):
    """Return rotated-row count on selection, or None before any output write.

    BF16 output owes the caller its inverse-RoPE (return 0). MXFP8 delegates
    inverse-RoPE/quantization to the original core epilogue (return every row).
    A fresh native plan per invocation prevents stale lengths across layers,
    warmup/capture and forwards; core metadata/schedulers are not modified.
    """
    if ((head_dim, nope_head_dim, rope_head_dim) != (512, 448, 64)
            or q.ndim != 3 or not 0 < q.shape[1] <= 128 or q.shape[0] == 0
            or q.shape[-1] != 512 or q.dtype != torch.bfloat16 or q.stride(-1) != 1
            or adaptive_splits or not _supported_device(q)):
        return None
    rows, heads, _ = q.shape
    if not _cache_supported(swa_k_cache, q):
        return None
    if not _selectors_supported(swa_indices, swa_lens, swa_ragged_indices, swa_ragged_indptr, q):
        return None
    if not swa_only and (not _cache_supported(kv_cache, q) or not _selectors_supported(
            topk_indices, topk_lens, topk_ragged_indices, topk_ragged_indptr, q)):
        return None
    if attn_sink is not None and (attn_sink.ndim != 1 or attn_sink.numel() < heads
            or attn_sink.dtype != torch.float32 or attn_sink.device != q.device):
        return None
    if output_mxfp8 is not None:
        if (output is not None or not isinstance(output_mxfp8, tuple) or len(output_mxfp8) != 2
                or inv_rope_positions is None or inv_rope_cos_sin_cache is None):
            return None
        data, scales = output_mxfp8
        if (data.shape != (rows, heads * 512) or scales.shape != (rows, heads * 16)
                or data.dtype != torch.float8_e4m3fn or scales.dtype != torch.uint8
                or data.device != q.device or scales.device != q.device
                or data.stride(-1) != 1 or scales.stride(-1) != 1
                or inv_rope_positions.shape != (rows,)
                or inv_rope_positions.dtype not in (torch.int32, torch.int64)
                or inv_rope_positions.device != q.device
                or inv_rope_cos_sin_cache.ndim != 2 or inv_rope_cos_sin_cache.shape[-1] != 64
                or inv_rope_cos_sin_cache.dtype != torch.float32
                or inv_rope_cos_sin_cache.device != q.device):
            return None
    elif (output is None or output.shape != q.shape or output.dtype != torch.bfloat16
          or output.device != q.device or output.stride(-1) != 1):
        return None

    main_ids, main_lens = _selectors(swa_indices, swa_lens, swa_ragged_indices,
        swa_ragged_indptr, rows, swa_k_cache.shape[0] * swa_k_cache.shape[1])
    extra_ids, extra_lens = None, None
    if not swa_only:
        extra_ids, extra_lens = _selectors(topk_indices, topk_lens, topk_ragged_indices,
            topk_ragged_indptr, rows, kv_cache.shape[0] * kv_cache.shape[1])
    # DTK D512 native decode consumes H64/H128. Pad queries/sinks only, so TP
    # slices (H4/H8/H16/...) use the same packed cache without any KV gather.
    native_heads = 64 if heads <= 64 else 128
    queries = torch.zeros((rows, 1, native_heads, 512), dtype=q.dtype, device=q.device)
    queries[:, 0, :heads].copy_(q)
    sink = None
    if attn_sink is not None:
        sink = torch.full((native_heads,), -torch.inf, dtype=torch.float32, device=q.device)
        sink[:heads].copy_(attn_sink[:heads])
    result, _ = provider(q=queries, k_cache=swa_k_cache.unsqueeze(-2),
        block_table=None, cache_seqlens=None, head_dim_v=512,
        tile_scheduler_metadata=metadata_factory()[0], softmax_scale=scale,
        causal=False, is_fp8_kvcache=True, indices=main_ids,
        topk_length=main_lens, attn_sink=sink,
        extra_k_cache=None if swa_only else kv_cache.unsqueeze(-2),
        extra_indices_in_kvcache=extra_ids, extra_topk_length=extra_lens)
    result = result[:, 0, :heads]
    nonempty = main_lens > 0 if swa_only else main_lens + extra_lens > 0
    result = torch.where(nonempty[:, None, None], result, 0.)
    if output_mxfp8 is not None:
        epilogue(result, inv_rope_positions, inv_rope_cos_sin_cache, rope_head_dim,
                 output_mxfp8[0], output_mxfp8[1])
        return rows
    output.copy_(result)
    return 0
