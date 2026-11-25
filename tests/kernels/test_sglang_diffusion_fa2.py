# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.fixture
def fa2_impl():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("Requires a HIP device and the DTK FlashAttention wheel")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx936":
        pytest.skip("HCU diffusion FA2 is registered for gfx936")
    pytest.importorskip("flash_attn")
    pytest.importorskip("sglang")
    from hcu_adapter.bridges.sglang_diffusion_fa2 import HcuFlashAttention2Backend

    return HcuFlashAttention2Backend.get_impl_cls()(
        num_heads=4, head_size=128, causal=False, softmax_scale=0.09
    )


@pytest.fixture(params=["FA", "FA2"])
def packed_impl(request, fa2_impl):
    if request.param == "FA2":
        return fa2_impl

    from hcu_adapter.bridges.sglang_diffusion_fa2 import HcuFlashAttentionBackend

    return HcuFlashAttentionBackend.get_impl_cls()(
        num_heads=4, head_size=128, causal=False, softmax_scale=0.09
    )


def reference(query, key, value, scale):
    q, k, v = (tensor.transpose(-3, -2).float() for tensor in (query, key, value))
    return ((q @ k.transpose(-1, -2) * scale).softmax(dim=-1) @ v).transpose(-3, -2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dense_fa2_matches_reference(fa2_impl, dtype):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    q, k, v = [
        torch.randn((2, 19, 4, 128), dtype=dtype, device="cuda", generator=generator)
        for _ in range(3)
    ]
    actual = fa2_impl.forward(q, k, v, None)
    expected = reference(q, k, v, fa2_impl.softmax_scale)
    torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.02)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("lengths", [(17, 12), (19, 0)])
def test_packed_fa2_matches_reference(packed_impl, dtype, lengths):
    generator = torch.Generator(device="cuda").manual_seed(5678)
    qkv = torch.randn(
        (sum(lengths), 3, 4, 128), dtype=dtype, device="cuda", generator=generator
    )
    q, k, v = qkv.unbind(dim=1)
    assert not q.is_contiguous()
    cu_seqlens = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int64)
    actual = packed_impl.forward_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max(lengths)
    )
    expected = torch.cat([
        reference(q_segment, k_segment, v_segment, packed_impl.softmax_scale)
        for q_segment, k_segment, v_segment in zip(
            q.split(lengths), k.split(lengths), v.split(lengths)
        )
        if q_segment.shape[0]
    ])
    torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.02)
