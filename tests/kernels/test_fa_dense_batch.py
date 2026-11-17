# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.fixture(scope='module')
def native_attention():
    from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import FlashAttentionImpl
    from hcu_adapter import fa_vendor
    fa_vendor.install()
    from sglang.kernels.ops.attention import flash_attention
    from sglang.kernels.ops.attention import flash_attention_v3
    # The native consumer was imported before installation. Its public
    # dispatcher must use the newly installed provider, not a cached alias.
    assert flash_attention.fa3_flash_attn_varlen_func is flash_attention_v3.flash_attn_varlen_func
    return FlashAttentionImpl, flash_attention.flash_attn_varlen_func


def reference(q, k, v, causal):
    q, k, v = [x.cpu().double().transpose(1, 2) for x in (q, k, v)]
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = (q @ k.transpose(-1, -2)) * q.shape[-1]**-0.5
    if causal:
        mask = torch.arange(k.shape[-2])[None, :] <= torch.arange(q.shape[-2])[:, None] + k.shape[-2] - q.shape[-2]
        scores.masked_fill_(~mask, -float('inf'))
    return (scores.softmax(-1) @ v).transpose(1, 2).float(), scores.logsumexp(-1).float()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires HCU/CUDA')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('causal', [False, True])
def test_dense_batch_public_dispatch_output_lse_and_changed_input_graph(native_attention, dtype, batch, causal):
    _, public = native_attention
    torch.manual_seed(691)
    q, k, v = [torch.randn(batch, length, heads, 66, device='cuda', dtype=dtype)[..., 1:65] for length, heads in [(17, 4), (29, 2), (29, 2)]]
    output_storage = torch.full((batch, 17, 4, 66), float('nan'), device='cuda', dtype=dtype)
    out = output_storage[..., 1:65]
    def execute():
        result = public(q, k, v, None, None, max_seqlen_q=17, max_seqlen_k=29, causal=causal, return_softmax_lse=True, out=out, ver=3)
        assert result[0] is out
        return result
    execute()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = execute()
    for phase in range(3):
        fresh = [torch.randn_like(x) for x in (q, k, v)]
        expected, expected_lse = reference(*fresh, causal)
        eager = None
        for mode in ['eager', 'graph']:
            for dst, src in zip((q, k, v), fresh): dst.copy_(src)
            result = execute() if mode == 'eager' else (graph.replay() or captured)
            torch.cuda.synchronize()
            actual, lse = [x.float().cpu() for x in result]
            assert actual.shape == expected.shape and lse.shape == expected_lse.shape
            torch.testing.assert_close(actual, expected, atol=.02 if dtype == torch.bfloat16 else .003, rtol=.02 if dtype == torch.bfloat16 else .003)
            torch.testing.assert_close(lse, expected_lse, atol=.001, rtol=.001)
            assert torch.isfinite(actual).all() and torch.isfinite(lse).all()
            assert torch.isnan(output_storage[..., :1]).all() and torch.isnan(output_storage[..., 65:]).all()
            for current, unchanged in zip((q, k, v), fresh): assert torch.equal(current, unchanged)
            if eager is None: eager = actual.clone()
            else: assert torch.equal(actual, eager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires HCU/CUDA')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('single', [False, True])
def test_native_diffusion_packed_consumer_uses_dense_and_ragged_contracts(native_attention, dtype, single):
    from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context, set_forward_context
    cls, _ = native_attention
    impl = cls(2, 64, False, 64**-0.5)
    torch.manual_seed(919)
    q, k, v = [torch.randn(32, 2, 64, device='cuda', dtype=dtype) for _ in range(3)]
    bounds = (0, 32) if single else (0, 8, 32)
    cu = torch.tensor(bounds, dtype=torch.int32, device='cuda')
    with set_forward_context(0, None):
        assert getattr(get_forward_context(), 'quality', 'exact') == 'exact'
        actual = impl.forward_varlen(q, k, v, cu_seqlens=cu, max_seqlen=32, cu_seqlens_host=bounds)
    refs = [reference(q[a:b][None], k[a:b][None], v[a:b][None], False)[0][0] for a, b in zip(bounds, bounds[1:])]
    torch.testing.assert_close(actual.float().cpu(), torch.cat(refs), atol=.02, rtol=.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires HCU/CUDA')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_packed_public_output_lse_and_changed_input_graph(native_attention, dtype):
    _, public = native_attention
    q, k, v = [torch.randn(31, heads, 66, device='cuda', dtype=dtype)[..., 1:65]
               for heads in (4, 2, 2)]
    cu = torch.tensor([0, 7, 31], device='cuda', dtype=torch.int32)
    storage = torch.full((31, 4, 66), float('nan'), device='cuda', dtype=dtype)
    out = storage[..., 1:65]

    def execute():
        result = public(q, k, v, cu, cu, max_seqlen_q=24, max_seqlen_k=24,
                        causal=True, return_softmax_lse=True, out=out, ver=3)
        assert len(result) == 2 and result[0] is out
        return result

    execute()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = execute()
    for _ in range(3):
        fresh = [torch.randn_like(t) for t in (q, k, v)]
        refs = [reference(*(t[a:b][None] for t in fresh), True)
                for a, b in ((0, 7), (7, 31))]
        expected = torch.cat([r[0][0] for r in refs])
        expected_lse = torch.cat([r[1][0].transpose(0, 1) for r in refs])
        eager = None
        for mode in ('eager', 'graph'):
            for dst, src in zip((q, k, v), fresh):
                dst.copy_(src)
            result = execute() if mode == 'eager' else (graph.replay() or captured)
            torch.cuda.synchronize()
            actual, lse = [t.float().cpu() for t in result]
            torch.testing.assert_close(actual, expected, atol=.02 if dtype == torch.bfloat16 else .003,
                                       rtol=.02 if dtype == torch.bfloat16 else .003)
            torch.testing.assert_close(lse, expected_lse, atol=.001, rtol=.001)
            assert torch.isnan(storage[..., :1]).all() and torch.isnan(storage[..., 65:]).all()
            for current, unchanged in zip((q, k, v), fresh):
                assert torch.equal(current, unchanged)
            if eager is None:
                eager = actual.clone()
            else:
                assert torch.equal(actual, eager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires HCU/CUDA')
@pytest.mark.parametrize('length', [4121, 9253])
def test_long_dense_diffusion_sampled_fp64_and_fresh_input_graph(native_attention, length):
    _, public = native_attention
    q, k, v = [torch.randn(length, 56, 128, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    cu = torch.tensor([0, length], device='cuda', dtype=torch.int32)
    indices = [0, 1, 31, length // 2, length - 1]

    def execute():
        return public(q, k, v, cu, cu, max_seqlen_q=length, max_seqlen_k=length, ver=3)

    execute()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = execute()
    for _ in range(3):
        for tensor in (q, k, v):
            tensor.copy_(torch.randn_like(tensor))
        query = q[indices].cpu().double().transpose(0, 1)
        key, value = [t.cpu().double().transpose(0, 1) for t in (k, v)]
        scores = (query @ key.transpose(1, 2)) * 128**-0.5
        expected = (scores.softmax(-1) @ value).transpose(0, 1).float()
        eager = execute()[indices].float().cpu()
        graph.replay()
        torch.cuda.synchronize()
        actual = captured[indices].float().cpu()
        torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
        assert (actual - expected).norm() / expected.norm() < .005
        assert torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0) > .9999
        assert torch.equal(eager, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires HCU/CUDA')
def test_explicit_vendor_public_dispatch_reaches_real_provider(native_attention, monkeypatch):
    from hcu_adapter import hcu_env, kill_switch
    if hcu_env.resolve('HCU_FA_IMPL')[0] != 'vendor' or kill_switch.op_disabled('vendor_fa'):
        pytest.skip('only the explicit vendor arm')
    import flash_attn
    real = flash_attn.flash_attn_varlen_func
    calls = []

    def observe(*args, **kwargs):
        calls.append(tuple(args[0].shape))
        return real(*args, **kwargs)

    monkeypatch.setattr(flash_attn, 'flash_attn_varlen_func', observe)
    _, public = native_attention
    q, k, v = [torch.randn(1, 17, 4, 64, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    actual = public(q, k, v, None, None, max_seqlen_q=17, max_seqlen_k=17, ver=3)
    expected, _ = reference(q, k, v, False)
    torch.testing.assert_close(actual.float().cpu(), expected, atol=.02, rtol=.02)
    assert calls == [(17, 4, 64)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires HCU/CUDA')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('dense', [False, True])
def test_vendor_operand_rejection_and_neutral_defaults_graph(native_attention, dtype, dense):
    from hcu_adapter import hcu_env, kill_switch
    from sgl_kernel.flash_attn import flash_attn_varlen_func
    if hcu_env.resolve('HCU_FA_IMPL')[0] != 'vendor' or kill_switch.op_disabled('vendor_fa'):
        pytest.skip('only the explicit vendor arm')
    q, k, v = [torch.randn(2, length, heads, 64, device='cuda', dtype=dtype)
               for length, heads in ((7, 4), (11, 2), (11, 2))]
    cu_q, cu_k = (None, None) if dense else (
        torch.tensor([0, 7, 14], device='cuda', dtype=torch.int32),
        torch.tensor([0, 11, 22], device='cuda', dtype=torch.int32),
    )
    operands = (q, k, v) if dense else tuple(t.flatten(0, 1) for t in (q, k, v))
    out = torch.full_like(operands[0], float('nan'))

    def execute(**kwargs):
        return flash_attn_varlen_func(*operands, cu_q, cu_k, 7, 11, out=out, **kwargs)

    for extra in ({'alibi_slopes': torch.ones(4, device='cuda')}, {'dropout_p': 0.25}):
        with pytest.raises(NotImplementedError, match='ALiBi|dropout'):
            execute(**extra)
        assert torch.isnan(out).all()
    with pytest.raises(TypeError, match='unexpected_bias'):
        execute(unexpected_bias=torch.ones(4, device='cuda'))
    assert torch.isnan(out).all()
    execute(alibi_slopes=None, dropout_p=0.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = execute(alibi_slopes=None, dropout_p=0.0)
    for _ in range(3):
        for tensor in (q, k, v):
            tensor.copy_(torch.randn_like(tensor))
        expected = reference(q, k, v, False)[0]
        expected = expected if dense else expected.flatten(0, 1)
        eager = execute(alibi_slopes=None, dropout_p=0.0).clone()
        graph.replay()
        torch.cuda.synchronize()
        assert captured is out
        tolerance = .02 if dtype == torch.bfloat16 else .003
        torch.testing.assert_close(out.float().cpu(), expected, atol=tolerance, rtol=tolerance)
        assert torch.equal(out, eager)
