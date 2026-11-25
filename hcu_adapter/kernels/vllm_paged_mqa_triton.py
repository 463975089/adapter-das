# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Non-MX vLLM ABI over the existing official paged-MQA Triton launcher.

This is NOT the SGLang preshuffled reader. The public vLLM helper reads
page64 row-major FP8 bytes plus page-tail FP32 scales, including shared-page
strides. Its output is workspace-backed and undefined in padding lanes;
return a fresh masked tensor to preserve the generic adapter contract.
"""


def fp32_paged_mqa_logits(q, cache, weights, lengths, table, metadata,
                          max_model_len, indices=None):
    import torch
    from hcu_adapter.kernels import mla_attention_emu as emu

    if q[1] is not None:
        raise NotImplementedError("vLLM paged MQA Triton requires non-MX FP8")
    query = q[0]
    emu._validate_fp32_paged_mqa(query, cache, weights, lengths, table)
    if max_model_len < 0:
        raise ValueError("MQA max_model_len must be nonnegative")
    # Keep all valid generic ABI faces: the official fast launcher assumes
    # power-of-two head tiles and contiguous inner axes. Unsupported faces
    # use the existing active Torch implementation, not a native guess.
    if (query.device.type != "cuda" or not torch.version.hip
            or query.shape[2] not in (32, 64) or query.stride(-1) != 1
            or weights.stride(-1) != 1 or table.stride(-1) != 1
            or not lengths.is_contiguous() or cache.storage_offset() % 4
            or cache.stride(0) % 4 or not query.shape[0] or not max_model_len):
        return emu.fp8_fp4_paged_mqa_logits(
            q, cache, weights, lengths, table, metadata, max_model_len, indices
        )
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_fp8_paged_mqa_logits_triton,
    )

    if current_platform.fp8_dtype() != query.dtype:
        raise NotImplementedError("vLLM paged MQA Triton requires E4M3FN platform semantics")
    # Workspace ownership remains entirely with vLLM. Never create/reset its
    # global manager here, catch kernel failures, or return its aliased scratch.
    scores = rocm_fp8_paged_mqa_logits_triton(
        query, cache, weights, lengths, table, max_model_len
    )
    from hcu_adapter.kernels._paged_mqa_mask_triton import owned_logits_mask

    return owned_logits_mask(scores, lengths)
