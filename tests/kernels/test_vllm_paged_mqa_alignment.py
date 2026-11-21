# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Shared paged-MQA reader: page-tail bytes need dtype-view alignment."""

import pytest
import torch

from hcu_adapter.kernels import mla_attention_emu as emu
from hcu_adapter.kernels.vllm_paged_mqa_triton import fp32_paged_mqa_logits


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires HCU/CUDA numerical runtime")
    return request.param


def _cache(value_bytes, scales, offset, padding):
    # Independent page-tail writer. Retain a possibly odd shared-page stride;
    # never use the production reader to construct the reference bytes.
    pages, page_kv, width = value_bytes.shape
    page_bytes = page_kv * (width + 4)
    storage = torch.full((offset + pages * (page_bytes + padding),), 0xA5,
                         dtype=torch.uint8, device=value_bytes.device)
    flat = storage.as_strided((pages, page_bytes), (page_bytes + padding, 1), offset)
    flat[:, :page_kv * width].copy_(value_bytes.reshape(pages, -1))
    flat[:, page_kv * width:].copy_(scales.view(torch.uint8).reshape(pages, -1))
    return flat.view(pages, page_kv, 1, width + 4), storage


@pytest.mark.parametrize("pages", [1, 2, 3])
@pytest.mark.parametrize("offset", [0, 1, 2, 3])
@pytest.mark.parametrize("padding", [0, 257])
@pytest.mark.parametrize("is_fp4", [False, True])
def test_split_index_cache_alignment(pages, offset, padding, is_fp4, device):
    width = 64 if is_fp4 else 128
    payload = torch.arange(pages * 64 * width, device=device).to(torch.uint8).reshape(pages, 64, width)
    packed = (torch.arange(pages * 64, device=device, dtype=torch.int32)
              + 0x3F000000).reshape(pages, 64)
    cache, storage = _cache(payload, packed, offset, padding)
    before = storage.clone()
    values, scales = emu._split_index_cache(cache, is_fp4)
    assert values.dtype == (torch.int8 if is_fp4 else torch.float8_e4m3fn)
    assert scales.dtype == torch.int32
    torch.testing.assert_close(values.view(torch.uint8), payload, rtol=0, atol=0)
    torch.testing.assert_close(scales, packed, rtol=0, atol=0)
    torch.testing.assert_close(storage, before, rtol=0, atol=0)


@pytest.mark.parametrize("pages", [1, 2, 3])
@pytest.mark.parametrize("offset", [0, 1, 2, 3])
@pytest.mark.parametrize("padding", [0, 257])
@pytest.mark.parametrize("entrypoint", ["emu", "triton_fallback"])
def test_non_mx_page_tail_alignment(pages, offset, padding, entrypoint, device):
    keys = ((torch.arange(pages * 64 * 128, device=device) % 7) - 3).float().reshape(pages, 64, 128)
    scales = torch.exp2((torch.arange(pages * 64, device=device) % 5 - 2).float()).reshape(pages, 64)
    cache, storage = _cache(keys.to(torch.float8_e4m3fn).view(torch.uint8), scales, offset, padding)
    before = storage.clone()
    # Three heads force the documented generic fallback on both CPU and HCU;
    # no dependency on an installed vLLM or its workspace is needed here.
    q = torch.ones((2, 1, 3, 128), device=device).to(torch.float8_e4m3fn)
    weights = torch.tensor([[1., 2., 3.], [2., 1., 4.]], device=device)
    lengths = torch.tensor([pages * 64 - 1, 1], dtype=torch.int32, device=device)
    table = torch.arange(pages - 1, -1, -1, device=device, dtype=torch.int32).repeat(2, 1)
    width = pages * 64 + 1
    score = emu.fp8_fp4_paged_mqa_logits if entrypoint == "emu" else fp32_paged_mqa_logits
    actual = score((q, None), cache, weights, lengths, table, None, width)
    # Reference uses the original keys/scales, not any unpacked cache result.
    per_token = keys.sum(-1).relu() * scales
    logical = per_token.flip(0).reshape(-1)
    expected = torch.zeros((2, width), device=device)
    expected[0, :pages * 64 - 1] = logical[:-1] * weights[0].sum()
    expected[1, 0] = logical[0] * weights[1].sum()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(storage, before, rtol=0, atol=0)


def test_aligned_single_page_scales_keep_storage():
    payload = torch.zeros((1, 64, 128), dtype=torch.uint8)
    packed = torch.full((1, 64), 0x3F800000, dtype=torch.int32)
    # Nonzero but aligned offset: do not allocate on the already valid path.
    cache, _ = _cache(payload, packed, offset=4, padding=0)
    _, scales = emu._split_index_cache(cache, False)
    assert scales.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert scales.storage_offset() == (4 + 64 * 128) // 4
