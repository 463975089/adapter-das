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

from copy import copy

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


class _QueryWorkspaceConfig:
    """Constructor-local query capacity; all other config reads stay native."""

    def __init__(self, original, rows):
        self.original = original
        self.scheduler_config = copy(original.scheduler_config)
        self.scheduler_config.max_num_batched_tokens = rows

    def __getattr__(self, name):
        return getattr(self.original, name)


def bind_query_workspace_constructor(constructor, query_rows):
    """Bound native Q/output reservations without shrinking global KV gathers."""
    anchors = ('get_current_vllm_config', 'get_prefill_workspace_size')
    if any(name not in constructor.__code__.co_names for name in anchors):
        raise RuntimeError('HY4 query workspace constructor anchor changed')
    get_config = constructor.__globals__[anchors[0]]
    get_kv_rows = constructor.__globals__[anchors[1]]

    def query_config():
        cfg = get_config()
        rows = query_rows(cfg)
        if rows is None:
            return cfg
        if type(rows) is not int or not 0 < rows <= cfg.scheduler_config.max_num_batched_tokens:
            raise ValueError('Invalid HY4 query workspace row capacity')
        return _QueryWorkspaceConfig(cfg, rows)

    def kv_workspace_size(cfg):
        original = cfg.original if isinstance(cfg, _QueryWorkspaceConfig) else cfg
        return get_kv_rows(original)

    return bind_native(constructor, {
        anchors[0]: query_config, anchors[1]: kv_workspace_size})


def _pcp_query_rows(cfg):
    from hcu_adapter.features.hy4.pcp_profile import matches, local_profile_tokens
    return local_profile_tokens(cfg) if matches(cfg) else None


def build_backend(vendor):
    from vllm.model_executor.layers.attention.sparse_mla_attention import SparseMLACommonImpl
    from vllm.models.hy_v4.nvidia.flashmla_sparse import (
        HYV4FlashMLASparseBackend as NativeBackend,
        HYV4FlashMLASparseImpl as NativeImpl,
    )
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseImpl, FlashMLASparseMetadataBuilder as NativeBuilder,
    )
    from vllm.v1.attention.ops.metadata import compute_token_to_req_indices

    class HcuIndexCapacityImpl(SparseMLACommonImpl):
        __init__ = bind_native(SparseMLACommonImpl.__init__,
            {'SparseMLAIndexGroupBuilder': make_index_group_builder})

    class HcuQueryCapacityImpl(FlashMLASparseImpl):
        __init__ = bind_query_workspace_constructor(
            FlashMLASparseImpl.__init__, _pcp_query_rows)

    class HcuSparseFlashMLAImpl(NativeImpl, HcuQueryCapacityImpl, HcuIndexCapacityImpl):
        # DTK reports the key-only LSE with sinks. Do not advertise the native
        # CUDA DCP merge until its single-sink normalization is restored.
        supports_dcp = False
        _bf16_flash_mla_kernel = bind_native(NativeImpl._bf16_flash_mla_kernel,
            {'flash_mla_sparse_fwd': sparse_entry(vendor.flash_mla_sparse_fwd)})
        _fp8_flash_mla_kernel = bind_native(NativeImpl._fp8_flash_mla_kernel,
            {'flash_mla_with_kvcache': vendor.flash_mla_with_kvcache})

    class HcuSparseFlashMLAMetadataBuilder(NativeBuilder):
        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            self.supports_draft_decode_metadata_update = (
                self.pcp_world_size == self.dcp_world_size == 1
                and not self.use_hisparse
                and vllm_config.cache_config.cache_dtype == 'fp8_ds_mla')

        def update_draft_decode_metadata(self, metadata):
            if not self.supports_draft_decode_metadata_update:
                return super().update_draft_decode_metadata(metadata)
            rows = metadata.num_decode_tokens
            if (rows == metadata.num_actual_tokens == metadata.num_prefills == 0):
                return
            if (metadata.num_prefills != 0
                    or rows != metadata.num_actual_tokens
                    or rows != metadata.num_decodes
                    or metadata.max_query_len != 1
                    or metadata.decode_max_query_len != 1):
                raise ValueError('DTK draft update requires uniform one-token decode metadata')
            # Seq lengths, slots and block tables already view runner buffers.
            # FP8 sparse scheduling uses fixed max-model cache lengths and no
            # topk_length; causal validity comes from the changing indices.
            # Only the builder-owned token/request map needs regeneration.
            compute_token_to_req_indices(metadata.query_start_loc,
                metadata.req_id_per_token, rows, rows)

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
