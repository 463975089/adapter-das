# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""LightOp MQA math with explicit V4.1 scale/cache and sparse-index bridges.

Qualified MXFP4 layouts use the packed LightOp paged/sparse kernels found in
the SGLang trace. Other layouts retain the counted BF16 compatibility bridge.
"""
import torch

from . import mla_attention_emu as reference
from .lightop_usage import _count

_CHUNK_VALUES = 1 << 24
_HEADS = (8, 16, 32, 64)


def _supported(q):
    data, sf = q
    width = data.shape[-1] * (2 if data.dtype == torch.int8 else 1)
    return sf is not None and width == 128 and data.shape[-2] in _HEADS


def _packed(q):
    return (q[0].shape[0] > 0 and q[0].dtype == torch.int8
            and q[0].shape[-1] == 64 and q[0].shape[-2] in (8, 16, 32)
            and q[1] is not None and q[1].dtype == torch.int32)


def _selected(query, weights, values, valid, output_dtype):
    from lightop.gemmopt import paged_mqa_logits

    rows, width, dim = values.shape
    if not rows or not width:
        return torch.zeros((rows, width), device=query.device, dtype=output_dtype)
    pages_per_row = (width + 63) // 64
    staged = torch.zeros((rows, pages_per_row * 64, dim), device=query.device, dtype=torch.bfloat16)
    staged[:, :width] = torch.where(valid[..., None], values, 0.).to(torch.bfloat16)
    table = torch.arange(rows * pages_per_row, device=query.device, dtype=torch.int32).view(rows, -1)
    lengths = torch.full((rows,), width, device=query.device, dtype=torch.int32)
    output = paged_mqa_logits(query.unsqueeze(1).contiguous(), staged.view(-1, 64, 1, dim),
                             weights.float().contiguous(), lengths, table, None,
                             pages_per_row * 64, True)
    return torch.where(valid, output[:, :width], 0.).to(output_dtype)


def _rows(query, weights, width, gather, output_dtype):
    rows = query.shape[0]
    output = torch.empty((rows, width), device=query.device, dtype=output_dtype)
    for column in range(0, width, _CHUNK_VALUES // 128):
        stop_column = min(width, column + _CHUNK_VALUES // 128)
        columns = slice(column, stop_column)
        padded_columns = (stop_column - column + 63) // 64 * 64
        chunk = max(1, _CHUNK_VALUES // (padded_columns * 128))
        for start in range(0, rows, chunk):
            selected = slice(start, min(rows, start + chunk))
            values, valid = gather(selected, columns)
            output[selected, columns] = _selected(query[selected], weights[selected], values, valid, output_dtype)
    return output


def fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end,
                       max_seqlen_k, schedule_meta=None, *, output_dtype=torch.bfloat16):
    if not _supported(q):
        _count('mqa.dense.compatibility')
        return reference.fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start,
                                           cu_seq_len_k_end, max_seqlen_k, schedule_meta, output_dtype=output_dtype)
    if kv[0].dtype == torch.int8 and max_seqlen_k > 0 and max_seqlen_k % 4 == 0:
        return packed_dense(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end,
                            max_seqlen_k, schedule_meta, output_dtype=output_dtype)
    query = reference._dequant_mqa_operand(*q)
    keys = reference._dequant_mqa_operand(*kv)
    columns = torch.arange(max_seqlen_k, device=query.device)
    def gather(rows, selected_columns):
        index = cu_seq_len_k_start[rows, None] + columns[selected_columns]
        valid = index < cu_seq_len_k_end[rows, None]
        safe = index.clamp(0, keys.shape[0] - 1).long()
        return keys[safe], valid
    output = _rows(query, weights, max_seqlen_k, gather, output_dtype)
    _count('mqa.dense_bridge.native')
    return output


def fp8_fp4_sparse_mqa_logits(q, kv, weights, metadata, num_max_sparse_blocks,
                              sparse_block_kv, use_unaligned_ks=False, *, output_dtype=torch.bfloat16):
    if not _supported(q):
        _count('mqa.sparse.compatibility')
        return reference.fp8_fp4_sparse_mqa_logits(q, kv, weights, metadata,
                                                  num_max_sparse_blocks, sparse_block_kv,
                                                  use_unaligned_ks, output_dtype=output_dtype)
    if (not isinstance(metadata, dict) or metadata['block_kv'] != sparse_block_kv
            or metadata['unaligned'] != use_unaligned_ks):
        raise ValueError('Sparse MQA metadata does not match this call')
    if (_packed(q) and kv[0].dtype == torch.int8 and kv[0].shape[0] > 0
            and sparse_block_kv in (8, 16) and 0 < num_max_sparse_blocks <= 4096
            and num_max_sparse_blocks % 4 == 0):
        from .packed_indexer_triton import sparse_linear

        result = sparse_linear(q, kv, weights, metadata, num_max_sparse_blocks,
                               sparse_block_kv, output_dtype=output_dtype)
        _count('mqa.sparse_packed.native')
        return result
    query = reference._dequant_mqa_operand(*q)
    keys = reference._dequant_mqa_operand(*kv)
    blocks = metadata['blocks'][:, :num_max_sparse_blocks].long()
    ks, ke = metadata['ks'].long(), metadata['ke'].long()
    base = ks % sparse_block_kv if use_unaligned_ks else torch.zeros_like(ks)
    offsets = torch.arange(sparse_block_kv, device=query.device)
    def gather(rows, columns):
        index = (blocks[rows, :, None] * sparse_block_kv + base[rows, None, None] + offsets).flatten(1)[:, columns]
        valid = ((index >= ks[rows, None]) & (index < ke[rows, None])
                 & (blocks[rows] >= 0).repeat_interleave(sparse_block_kv, 1)[:, columns])
        return keys[index.clamp(0, keys.shape[0] - 1)], valid
    output = _rows(query, weights, num_max_sparse_blocks * sparse_block_kv, gather, output_dtype)
    _count('mqa.sparse_bridge.native')
    return output


def fp8_fp4_paged_mqa_logits(q, fused_kv_cache, weights, context_lens, block_table,
                             schedule_meta, max_context_len, indices=None, *, output_dtype=torch.bfloat16):
    if (_packed(q) and q[0].ndim == 4 and q[0].shape[1] == 1
            and fused_kv_cache.shape[1:] == (64, 1, 68)
            and max_context_len > 0 and indices is not None):
        from .packed_indexer_triton import paged_dense

        result = paged_dense(q, fused_kv_cache, weights, context_lens,
                             block_table, max_context_len, output_dtype=output_dtype)
        _count('mqa.paged_packed.native')
        return result
    if not _supported(q) or q[0].shape[1] != 1 or indices is None:
        _count('mqa.paged.compatibility')
        return reference.fp8_fp4_paged_mqa_logits(q, fused_kv_cache, weights, context_lens,
                                                 block_table, schedule_meta, max_context_len, indices, output_dtype=output_dtype)
    query = reference._dequant_mqa_operand(q[0][:, 0], q[1][:, 0])
    lengths = context_lens.reshape(-1)
    positions = torch.arange(max_context_len, device=query.device)
    def gather(rows, columns):
        index = positions[columns].expand(query[rows].shape[0], -1)
        valid = index < lengths[rows, None]
        values = reference._paged_tokens(fused_kv_cache, block_table[rows], index, valid,
                                         q[0].dtype == torch.int8)
        return values, valid
    output = _rows(query, weights, max_context_len, gather, output_dtype)
    _count('mqa.paged_bridge.native')
    return output


def fp8_fp4_paged_sparse_mqa_logits(q, fused_kv_cache, weights, metadata,
                                    num_max_sparse_blocks, sparse_block_kv, *, output_dtype=torch.bfloat16):
    if not _supported(q):
        _count('mqa.paged_sparse.compatibility')
        return reference.fp8_fp4_paged_sparse_mqa_logits(q, fused_kv_cache, weights,
                                                        metadata, num_max_sparse_blocks, sparse_block_kv, output_dtype=output_dtype)
    if not isinstance(metadata, dict) or metadata['block_kv'] != sparse_block_kv:
        raise ValueError('Paged sparse MQA metadata does not match this call')
    if (_packed(q) and sparse_block_kv in (8, 16)
            and fused_kv_cache.shape[1] % sparse_block_kv == 0
            and 0 < num_max_sparse_blocks <= 4096 and num_max_sparse_blocks % 4 == 0
            and 'indices' in metadata):
        from .packed_indexer_triton import paged_sparse

        result = paged_sparse(q, fused_kv_cache, weights, metadata,
                              num_max_sparse_blocks, sparse_block_kv, output_dtype=output_dtype)
        _count('mqa.paged_sparse_packed.native')
        return result
    query = reference._dequant_mqa_operand(q[0][:, 0], q[1][:, 0])
    blocks = metadata['blocks'][:, :num_max_sparse_blocks].long()
    lengths = metadata['lens'].reshape(-1).long()
    offsets = torch.arange(sparse_block_kv, device=query.device)
    def gather(rows, columns):
        index = (blocks[rows, :, None] * sparse_block_kv + offsets).flatten(1)[:, columns]
        valid = ((index >= 0) & (index < lengths[rows, None])
                 & (blocks[rows] >= 0).repeat_interleave(sparse_block_kv, 1)[:, columns])
        values = reference._paged_tokens(fused_kv_cache, metadata['block_table'][rows], index,
                                         valid, q[0].dtype == torch.int8)
        return values, valid
    output = _rows(query, weights, num_max_sparse_blocks * sparse_block_kv, gather, output_dtype)
    _count('mqa.paged_sparse_bridge.native')
    return output


def packed_dense(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end,
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
