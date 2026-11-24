# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU ownership/contract tests; GPU metadata and kernel replay are separate."""
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_adapter.kernels import vllm_sparse_flashmla_dtk as kernel


def make_builder(monkeypatch, *, pcp=1, dcp=1, dtype='fp8_ds_mla', hisparse=False):
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseMetadataBuilder as NativeBuilder,
    )

    def native_init(self, spec, names, cfg, device):
        # Only heavy device allocation is replaced in this CPU contract test.
        self.pcp_world_size, self.dcp_world_size = pcp, dcp
        self.use_hisparse = hisparse
        self.native_inputs = (spec, names, cfg, device)

    monkeypatch.setattr(NativeBuilder, '__init__', native_init)
    vendor = NS(flash_mla_sparse_fwd=lambda *a, **k: None,
        flash_mla_with_kvcache=lambda *a, **k: None,
        get_mla_metadata=lambda: (object(), None))
    cls = kernel.build_backend(vendor).get_builder_cls()
    cfg = NS(cache_config=NS(cache_dtype=dtype))
    spec, names, device = object(), ['draft.attn'], object()
    builder = cls(spec, names, cfg, device)
    assert builder.native_inputs == (spec, names, cfg, device)
    assert cls.build is NativeBuilder.build
    return builder


def metadata(rows=3, **overrides):
    data = dict(num_actual_tokens=rows, num_decodes=rows, num_decode_tokens=rows,
        num_prefills=0, max_query_len=1, decode_max_query_len=1,
        query_start_loc=torch.arange(rows + 1, dtype=torch.int32),
        req_id_per_token=torch.full((max(rows, 1),), -99, dtype=torch.int32),
        block_table=object(), slot_mapping=object(), seq_lens=object(),
        fp8_extra_metadata=object())
    data.update(overrides)
    return NS(**data)


def test_owned_builder_advertises_native_draft_update(monkeypatch):
    builder = make_builder(monkeypatch)
    assert builder.supports_draft_decode_metadata_update


@pytest.mark.parametrize('settings', [dict(pcp=16), dict(dcp=2),
    dict(hisparse=True), dict(dtype='auto'), dict(dtype='bfloat16'),
    dict(dtype='fp8_e4m3'), dict(dtype='nvfp4_ds_mla')])
def test_other_metadata_scopes_keep_original_fallback(monkeypatch, settings):
    builder = make_builder(monkeypatch, **settings)
    assert not builder.supports_draft_decode_metadata_update
    with pytest.raises(NotImplementedError):
        builder.update_draft_decode_metadata(metadata())


def test_callback_delegates_mapping_without_replacing_metadata(monkeypatch):
    from vllm.v1.attention.ops import metadata as native
    calls = []

    def map_requests(query_start_loc, out, mapped, total):
        calls.append((query_start_loc, out, mapped, total))
        out[:total] = torch.arange(total, dtype=torch.int32)
        return out[:total]

    monkeypatch.setattr(native, 'compute_token_to_req_indices', map_requests)
    builder = make_builder(monkeypatch)
    md = metadata()
    before = vars(md).copy()
    assert builder.update_draft_decode_metadata(md) is None
    assert calls == [(md.query_start_loc, md.req_id_per_token, 3, 3)]
    assert torch.equal(md.req_id_per_token, torch.arange(3, dtype=torch.int32))
    assert all(getattr(md, key) is value for key, value in before.items())


def test_empty_callback_does_not_launch(monkeypatch):
    from vllm.v1.attention.ops import metadata as native
    monkeypatch.setattr(native, 'compute_token_to_req_indices',
        lambda *a, **k: pytest.fail('empty batch launched mapping'))
    builder = make_builder(monkeypatch)
    builder.update_draft_decode_metadata(metadata(0))


def test_empty_collective_placeholder_does_not_launch(monkeypatch):
    from vllm.v1.attention.ops import metadata as native
    monkeypatch.setattr(native, 'compute_token_to_req_indices',
        lambda *a, **k: pytest.fail('empty placeholder launched mapping'))
    builder = make_builder(monkeypatch)
    builder.update_draft_decode_metadata(metadata(0, num_decodes=1,
        max_query_len=0, decode_max_query_len=0))


@pytest.mark.parametrize('change', [dict(num_prefills=1), dict(num_decodes=2),
    dict(num_decode_tokens=2), dict(max_query_len=2), dict(decode_max_query_len=2)])
def test_nonuniform_or_prefill_metadata_is_rejected_before_launch(monkeypatch, change):
    from vllm.v1.attention.ops import metadata as native
    monkeypatch.setattr(native, 'compute_token_to_req_indices',
        lambda *a, **k: pytest.fail('invalid metadata launched mapping'))
    builder = make_builder(monkeypatch)
    with pytest.raises((AssertionError, ValueError), match='decode|uniform|token'):
        builder.update_draft_decode_metadata(metadata(**change))
