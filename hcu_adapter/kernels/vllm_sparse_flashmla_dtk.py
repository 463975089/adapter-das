# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Restore Core's sparse FlashMLA calls using the DTK vendor ABI.

The native metadata builder, cache reader/writer, PCP path, query padding and
sink handling remain Core-owned. Private function namespaces bind the vendor
callables and Core's existing decode-row capacity helper; no Core module global
or source is rewritten.
Remove these bindings when Core exposes an operator-provider seam.
"""
from __future__ import annotations

from hcu_adapter.infra.native_namespace import bind_native


def sparse_entry(provider):
    """Older DTK providers lack Core's optional destination-buffer argument."""
    def sparse(q, kv, indices, sm_scale, d_v=512, attn_sink=None,
               topk_length=None, out=None):
        result = provider(q, kv, indices, sm_scale, d_v, attn_sink, topk_length)
        if out is None:
            return result
        output, maximum, lse = result
        out.copy_(output)
        return out, maximum, lse
    return sparse


def make_index_group_builder(logical_topk_indices):
    """Use Core's decode-row bound for its persistent conversion tables."""
    from vllm.config import get_current_vllm_config
    from vllm.v1.attention.backends.mla.index_group import (
        SparseMLAIndexGroupBuilder, get_sparse_mla_index_group_max_rows,
    )
    rows = get_sparse_mla_index_group_max_rows(get_current_vllm_config())
    return SparseMLAIndexGroupBuilder(logical_topk_indices, max_decode_rows=rows)


def build_backend(vendor):
    from vllm.model_executor.layers.attention.sparse_mla_attention import SparseMLACommonImpl
    from vllm.models.hy_v4.nvidia.flashmla_sparse import (
        HYV4FlashMLASparseBackend as NativeBackend,
        HYV4FlashMLASparseImpl as NativeImpl,
    )
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseMetadataBuilder as NativeBuilder,
    )

    class HcuIndexCapacityImpl(SparseMLACommonImpl):
        __init__ = bind_native(SparseMLACommonImpl.__init__,
            {'SparseMLAIndexGroupBuilder': make_index_group_builder})

    class HcuSparseFlashMLAImpl(NativeImpl, HcuIndexCapacityImpl):
        # DTK reports the key-only LSE with sinks. Do not advertise the native
        # CUDA DCP merge until its single-sink normalization is restored.
        supports_dcp = False
        _bf16_flash_mla_kernel = bind_native(NativeImpl._bf16_flash_mla_kernel,
            {'flash_mla_sparse_fwd': sparse_entry(vendor.flash_mla_sparse_fwd)})
        _fp8_flash_mla_kernel = bind_native(NativeImpl._fp8_flash_mla_kernel,
            {'flash_mla_with_kvcache': vendor.flash_mla_with_kvcache})

    class HcuSparseFlashMLAMetadataBuilder(NativeBuilder):
        _build_fp8_mixed_decode_prefill = bind_native(
            NativeBuilder._build_fp8_mixed_decode_prefill,
            {'get_mla_metadata': vendor.get_mla_metadata})
        _build_fp8_separate_prefill_decode = bind_native(
            NativeBuilder._build_fp8_separate_prefill_decode,
            {'get_mla_metadata': vendor.get_mla_metadata})

    class HcuSparseFlashMLABackend(NativeBackend):
        @classmethod
        def get_supported_head_sizes(cls):
            return [576]

        @staticmethod
        def get_impl_cls():
            return HcuSparseFlashMLAImpl

        @staticmethod
        def get_builder_cls():
            return HcuSparseFlashMLAMetadataBuilder

    # The public backend registry resolves this stable import path. No other
    # Core model/implementation class is rebound or replaced.
    for cls in (HcuSparseFlashMLAImpl, HcuSparseFlashMLAMetadataBuilder,
                HcuSparseFlashMLABackend):
        cls.__qualname__ = cls.__name__
        globals()[cls.__name__] = cls
    return HcuSparseFlashMLABackend
