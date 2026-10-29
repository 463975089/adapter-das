"""Compatibility MQA entry points; implementations live in mqa_logits_lightop."""

import torch


def fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end,
                       max_seqlen_k, schedule_meta=None, *, output_dtype=torch.bfloat16):
    from .mqa_logits_lightop import fp8_fp4_mqa_logits as implementation

    return implementation(
        q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k, schedule_meta,
        output_dtype=output_dtype)


def fp8_fp4_sparse_mqa_logits(q, kv, weights, metadata, num_max_sparse_blocks,
                              sparse_block_kv, use_unaligned_ks=False, *, output_dtype=torch.bfloat16):
    from .mqa_logits_lightop import fp8_fp4_sparse_mqa_logits as implementation

    return implementation(
        q, kv, weights, metadata, num_max_sparse_blocks, sparse_block_kv, use_unaligned_ks,
        output_dtype=output_dtype)


def fp8_fp4_paged_mqa_logits(q, fused_kv_cache, weights, context_lens, block_table,
                             schedule_meta, max_context_len, indices=None, *, output_dtype=torch.bfloat16):
    from .mqa_logits_lightop import fp8_fp4_paged_mqa_logits as implementation

    return implementation(
        q, fused_kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len,
        indices, output_dtype=output_dtype)


def fp8_fp4_paged_sparse_mqa_logits(q, fused_kv_cache, weights, metadata,
                                    num_max_sparse_blocks, sparse_block_kv, *, output_dtype=torch.bfloat16):
    from .mqa_logits_lightop import fp8_fp4_paged_sparse_mqa_logits as implementation

    return implementation(
        q, fused_kv_cache, weights, metadata, num_max_sparse_blocks, sparse_block_kv,
        output_dtype=output_dtype)
