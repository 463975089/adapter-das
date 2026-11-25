# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Non-MX page64 scoring through the HCU AITER Opus interface."""

from functools import lru_cache
import logging


@lru_cache(maxsize=32)
def _report_provider(provider, page_stride):
    logging.getLogger("vllm." + __name__).info(
        "HCU paged MQA provider=%s cache_page_stride=%s", provider, page_stride,
    )


def fp32_paged_mqa_logits(q, cache, weights, lengths, table, metadata,
                          max_model_len, indices=None):
    import torch
    from hcu_adapter.kernels import mla_attention_emu as emu
    from hcu_adapter.kernels.vllm_paged_mqa_triton import (
        fp32_paged_mqa_logits as triton_logits,
    )

    if q[1] is not None:
        raise NotImplementedError("vLLM AITER paged MQA requires non-MX FP8")
    query = q[0]
    emu._validate_fp32_paged_mqa(query, cache, weights, lengths, table)
    if max_model_len < 0:
        raise ValueError("MQA max_model_len must be nonnegative")
    # Shared physical pages can contain another layer's payload. Copying the
    # entire allocated cache per decode step would grow with cache capacity.
    supported = (
        query.device.type == "cuda" and torch.version.hip
        and query.shape[2] in (32, 64) and 0 < query.shape[0] <= 65535
        and 0 < max_model_len <= 2147483647 and cache.numel() <= 2147483647
        and (cache.is_contiguous() or (
            cache.stride(0) >= 64 * 132 and cache.stride(0) % 4 == 0
        ))
        and cache.data_ptr() % 4 == 0
        and query.data_ptr() % 4 == 0
        and not query.requires_grad and not weights.requires_grad
    )
    if supported:
        arch = torch.cuda.get_device_properties(query.device).gcnArchName.split(":")[0]
        supported = arch in ("gfx936", "gfx938")
    if not supported:
        _report_provider("triton (unsupported AITER layout/device)", cache.stride(0))
        return triton_logits(q, cache, weights, lengths, table, metadata,
                             max_model_len, indices=indices)

    try:
        import aiter
    except ModuleNotFoundError as error:
        if error.name not in ("aiter", "aiter.ops", "aiter.ops.opus",
                              "aiter.ops.opus.paged_mqa_logits"):
            raise
        return triton_logits(q, cache, weights, lengths, table, metadata,
                             max_model_len, indices=indices)

    from hcu_adapter._base import Missing

    paged_mqa_logits = getattr(aiter, "paged_mqa_logits", None)
    if isinstance(paged_mqa_logits, Missing) or not callable(paged_mqa_logits):
        return triton_logits(q, cache, weights, lengths, table, metadata,
                             max_model_len, indices=indices)

    supports_stride = (
        arch == "gfx938"
        and getattr(paged_mqa_logits, "supports_page_stride", False) is True
    )
    if not cache.is_contiguous() and not supports_stride:
        _report_provider("triton (AITER lacks page-stride support)", cache.stride(0))
        return triton_logits(q, cache, weights, lengths, table, metadata,
                             max_model_len, indices=indices)

    _report_provider("aiter.paged_mqa_logits", cache.stride(0))

    from hcu_adapter.kernels._paged_mqa_mask_triton import owned_logits_mask

    # AITER requires positive sequence lengths. Graph padding may have zero
    # lengths and invalid tables. Substitute a valid page for those rows and
    # mask using the original lengths, entirely on the current device stream.
    with torch.cuda.device(query.device), torch.inference_mode():
        row_lengths = lengths.reshape(-1).contiguous()
        safe_lengths = row_lengths.clamp(min=1, max=min(max_model_len, table.shape[1] * 64))
        safe_table = torch.where(row_lengths[:, None] > 0, table, 0).contiguous()
        scores = paged_mqa_logits(
            query.contiguous(), cache, weights.contiguous(), safe_lengths,
            safe_table, max_model_len, clean_logits=False,
        )
        return owned_logits_mask(scores, row_lengths)
