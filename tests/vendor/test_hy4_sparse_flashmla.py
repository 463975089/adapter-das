# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Actual Core sink wrapper + DTK kernel; not whole-model qualification."""
import pytest
import torch


@pytest.mark.parametrize('rows,heads', [(1, 64), (8, 64), (17, 128)])
@pytest.mark.parametrize('with_sink', [False, True])
@pytest.mark.parametrize('empty_first_row', [False, True])
def test_hy4_bf16_prefill_against_cpu_reference(rows, heads, with_sink, empty_first_row):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires HCU')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] not in ('gfx936', 'gfx938'):
        pytest.skip('requires gfx936/gfx938')
    from flash_mla import flash_mla_interface as vendor
    from hcu_adapter.kernels.vllm_sparse_flashmla_dtk import build_backend

    torch.manual_seed(734)
    q = torch.randn(rows, heads, 576, dtype=torch.bfloat16) * .1
    kv = torch.randn(257, 1, 576, dtype=torch.bfloat16) * .1
    ids = torch.randint(0, kv.shape[0], (rows, 2048), dtype=torch.int32)
    lengths = torch.tensor([min(2048, 7 + 31 * i) for i in range(rows)], dtype=torch.int32)
    if empty_first_row:
        lengths[0] = 0
    sinks = torch.linspace(-1, 1, heads) if with_sink else None
    columns = torch.arange(ids.shape[1])
    valid = columns[None, :] < lengths[:, None]
    # Retain duplicates and a nonuniform gather; invalid columns must not
    # contribute to attention or its normalization.
    safe_ids = torch.where(valid, ids, -1)
    keys = kv[:, 0].float()[ids.long()]
    scores = torch.einsum('rhd,rkd->rhk', q.float(), keys) * .25
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    key_lse = torch.logsumexp(scores, dim=-1)
    denominator = key_lse
    if sinks is not None:
        denominator = torch.logaddexp(denominator, sinks[None, :])
    probs = torch.nan_to_num(torch.exp(scores - denominator[:, :, None]))
    expected = torch.einsum('rhk,rkd->rhd', probs, keys[..., :512])

    impl_type = build_backend(vendor).get_impl_cls()
    impl = object.__new__(impl_type)
    impl.num_heads = heads
    impl.prefill_padding = 64
    impl.softmax_scale = .25
    impl.sinks = None if sinks is None else sinks.cuda()
    actual, lse = impl._bf16_flash_mla_kernel(
        q.cuda(), kv.cuda(), safe_ids.cuda(), lengths.cuda())
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=.03, atol=.001)
    # DTK returns key-only LSE for nonempty rows, and a non-mathematical
    # sentinel for empty rows (their output is checked above). Core consumes
    # neither in PCP16/DCP1. DCP is refused until its normalization is adapted.
    nonempty = lengths > 0
    torch.testing.assert_close(lse.cpu()[nonempty], key_lse[nonempty],
                               rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('heads', [64, 128])
@pytest.mark.parametrize('with_sink', [False, True])
def test_hy4_packed_fp8_decode_real_writer_and_changed_input_graph(heads, with_sink):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires HCU')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] not in ('gfx936', 'gfx938'):
        pytest.skip('requires gfx936/gfx938')
    from flash_mla import flash_mla_interface as vendor
    from hcu_adapter.kernels.vllm_sparse_flashmla_dtk import build_backend
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata

    torch.manual_seed(734)
    rows, pages, page = 3, 4, 64
    cache = torch.zeros(pages, page, 656, dtype=torch.uint8, device='cuda')
    latent = (torch.randn(pages * page, 512, device='cuda') * .1).bfloat16()
    rope = (torch.randn(pages * page, 64, device='cuda') * .1).bfloat16()
    ops.concat_and_cache_mla(latent, rope, cache,
        torch.arange(pages * page, device='cuda', dtype=torch.int64),
        'fp8_ds_mla', torch.ones(1, device='cuda'))
    # Independent CPU reader of the D576 packed cache; no framework gather.
    packed = cache.cpu().reshape(-1, 656)
    data = packed[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = packed[:, 512:528].contiguous().view(torch.float32)
    decoded = torch.cat((data * scales.repeat_interleave(128, dim=-1),
                        packed[:, 528:].contiguous().view(torch.bfloat16).float()), dim=-1)
    assert torch.isfinite(decoded).all()
    q = (torch.randn(1, rows, heads, 576, device='cuda') * .1).bfloat16()
    ids = torch.full((1, rows, 2048), -1, dtype=torch.int32, device='cuda')
    selected = torch.arange(128, device='cuda', dtype=torch.int32)
    selected[1] = selected[0]
    ids[:, :, :128] = selected
    sink = torch.linspace(-1, 1, heads, device='cuda') if with_sink else None
    impl = object.__new__(build_backend(vendor).get_impl_cls())
    impl.num_heads = heads
    impl.fp8_decode_padded_heads = heads
    impl.softmax_scale = .25
    impl.sinks = sink
    scheduler, _ = vendor.get_mla_metadata()
    md = FlashMLASparseMetadata.FP8KernelMetadata(scheduler_metadata=scheduler,
        dummy_block_table=torch.zeros(1, 1, dtype=torch.int32, device='cuda'),
        cache_lens=torch.full((1,), pages * page, dtype=torch.int32, device='cuda'))

    def reference():
        keys = decoded[selected.cpu().long()]
        scores = torch.einsum('rhd,kd->rhk', q[0].cpu().float(), keys) * .25
        denom = torch.logsumexp(scores, dim=-1)
        if sink is not None:
            denom = torch.logaddexp(denom, sink.cpu()[None, :])
        probabilities = torch.exp(scores - denom[:, :, None])
        return torch.einsum('rhk,kd->rhd', probabilities, keys[:, :512])

    # Warm the vendor scheduler before capture, retaining its actual ABI.
    for _ in range(3):
        result, _ = impl._fp8_flash_mla_kernel(q, cache, ids, md)
    torch.cuda.synchronize()
    torch.testing.assert_close(result[0].cpu().float(), reference(), rtol=.03, atol=.001)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, _ = impl._fp8_flash_mla_kernel(q, cache, ids, md)
    for factor in (.5, -1.):
        q.mul_(factor)
        if sink is not None:
            sink.add_(.2)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured[0].cpu().float(), reference(), rtol=.03, atol=.001)
