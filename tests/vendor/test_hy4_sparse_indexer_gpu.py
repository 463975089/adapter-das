# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Actual Core indexer, native cache ABI and changed-input Graph; not model acceptance."""
import pytest
import torch


@pytest.mark.parametrize('heads', [32, 64])
def test_hy4_indexer_native_body_and_changed_input_graph(heads):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires HCU')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] not in ('gfx936', 'gfx938'):
        pytest.skip('requires gfx936/gfx938')
    from types import SimpleNamespace as NS
    from vllm import _custom_ops as ops
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import override_forward_context
    from vllm.v1.worker.workspace import init_workspace_manager
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadata as Metadata,
        DeepseekV32IndexerPrefillMetadata as Prefill,
        DeepseekV32IndexerPrefillChunkMetadata as Chunk,
        DeepSeekV32IndexerDecodeMetadata as Decode,
    )
    from hcu_adapter.features.dsa import hy4_sparse_indexer as owned
    from hcu_adapter.kernels.compact_mqa_logits_triton import compact

    torch.manual_seed(42)
    init_workspace_manager(torch.device('cuda:0'))
    n, rows, dim = 96, 4, 128
    k = torch.randn((n, dim), device='cuda', dtype=torch.bfloat16)
    cache = torch.zeros((2, 64, 132), device='cuda', dtype=torch.uint8)
    slots = torch.cat((torch.arange(32), torch.arange(64, 128))).to('cuda')
    ops.indexer_k_quant_and_cache(k, cache, slots, 128, 'ue8m0')
    values = torch.empty((n, dim), dtype=torch.float8_e4m3fn, device='cuda')
    sf = torch.empty((n, 4), dtype=torch.uint8, device='cuda')
    table = torch.tensor([[0], [1]], dtype=torch.int32, device='cuda')
    cu = torch.tensor([0, 32, 96], dtype=torch.int32, device='cuda')
    ops.cp_gather_indexer_k_quant_cache(cache, values, sf, table, cu)
    scale = sf.view(torch.float32).flatten()
    q = torch.randn((rows, heads, dim), device='cuda').to(torch.float8_e4m3fn)
    w = torch.randn((rows, heads), device='cuda', dtype=torch.float32)
    starts = torch.tensor([0, 0, 32, 32], device='cuda', dtype=torch.int32)
    ends = torch.tensor([10, 32, 80, 96], device='cuda', dtype=torch.int32)
    provider = owned._provider()[1]
    prefill = owned.make_prefill(provider, compact, 2 * n * 4)

    def reference():
        keys = values.cpu().float() * scale.cpu()[:, None]
        full = torch.einsum('mhd,nd->mhn', q.cpu().float(), keys).relu()
        full = (full * w.cpu()[:, :, None]).sum(1)
        columns = torch.arange(64)[None] + starts.cpu()[:, None]
        valid = columns < ends.cpu()[:, None]
        return full.gather(1, columns.clamp(max=n - 1).long()).masked_fill(~valid, -float('inf'))

    actual = prefill((q, None), (values, scale), w, starts, ends, 64)
    torch.testing.assert_close(actual.cpu(), reference(), atol=0.08, rtol=0.005)
    for _ in range(3):
        prefill((q, None), (values, scale), w, starts, ends, 64)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = prefill((q, None), (values, scale), w, starts, ends, 64)
    for factor in (0.75, -0.25, 1.5):
        w.mul_(factor)
        q.copy_(torch.randn(q.shape, device=q.device).to(q.dtype))
        scale.mul_(1.125)
        ends.copy_(torch.tensor([12, 30, 70, 94], device='cuda', dtype=torch.int32))
        graph.replay()
        torch.testing.assert_close(captured.cpu(), reference(), atol=0.08, rtol=0.005)

    ops.cp_gather_indexer_k_quant_cache(cache, values, sf, table, cu)
    output = torch.empty((rows, 8), dtype=torch.int32, device='cuda')
    layer = NS(use_fp4_cache=False, compress_ratio=1, dcp_world_size=1,
        candidate_blocks=None, candidate_block_size=0, candidate_write=False,
        k_cache=NS(prefix='probe.k_cache', kv_cache=cache), quant_block_size=128,
        scale_fmt='ue8m0', topk_tokens=8, head_dim=128, max_model_len=128,
        max_total_seq_len=n, topk_indices_buffer=output, skip_k_cache_insert=True,
        use_pcp=False, pcp_shard_decode_requests=False, dense_mha_metadata_layer_name='',
        dcp_rank=0, _parallel_config=NS(cp_kv_cache_interleave_size=1), topk_backend='auto')
    hidden = torch.empty((rows, 8), device='cuda')
    with override_forward_context(NS(attn_metadata=None)):
        result = owned.forward(layer, hidden, q, None, w)
    assert result.data_ptr() == output.data_ptr()
    chunk = Chunk(table, starts, ends, cu, torch.tensor([0, 0, 1, 1], device='cuda'),
        n, 64, 0, rows, 2, local_cu_seq_lens=cu, local_total_seq_lens=n,
        max_local_total_seq_lens=n)
    metadata = Metadata(torch.tensor([32, 64], device='cuda'), 64,
        torch.empty((0,), dtype=torch.int64, device='cuda'), 0, 0, 2, rows,
        prefill=Prefill([chunk]))
    with override_forward_context(NS(attn_metadata={'probe.k_cache': metadata},
                                     cudagraph_runtime_mode=CUDAGraphMode.NONE)):
        result = owned.forward(layer, hidden, q, None, w)
    expected = reference().topk(8, dim=-1).indices
    assert torch.equal(result.cpu().sort(-1).values, expected.int().sort(-1).values)

    # Compile the actual public route and opaque Core-body operator, rather
    # than only capturing its vendor leaf. Preparation is outside tracing.
    owned.prepare()
    def run_indexer(hidden, query, weights):
        return owned.forward(layer, hidden, query, None, weights)
    compiled = torch.compile(run_indexer, backend='eager', fullgraph=True)
    with override_forward_context(NS(attn_metadata={'probe.k_cache': metadata},
                                     cudagraph_runtime_mode=CUDAGraphMode.NONE)):
        compiled_result = compiled(hidden, q, w)
    assert torch.equal(compiled_result.cpu().sort(-1).values, expected.int().sort(-1).values)

    # Ragged verification needs identical per-request padding for Q and weights.
    starts.copy_(torch.tensor([0, 32, 32, 32], device='cuda', dtype=torch.int32))
    ends.copy_(torch.tensor([12, 66, 80, 94], device='cuda', dtype=torch.int32))
    layer.topk_indices_buffer = torch.empty((6, 8), device='cuda', dtype=torch.int32)
    decode = Decode(table,
        torch.tensor([[12, 0, 0], [34, 48, 62]], device='cuda', dtype=torch.int32),
        torch.tensor([1, 3], device='cuda', dtype=torch.int32), True,
        torch.empty((0, 2), device='cuda', dtype=torch.int32),
        decode_is_uniform=False, write_max_decode_len=3)
    metadata = Metadata(torch.tensor([32, 64], device='cuda'), 64,
        torch.empty((0,), dtype=torch.int64, device='cuda'), 2, rows, 0, 0, decode=decode)
    with override_forward_context(NS(attn_metadata={'probe.k_cache': metadata},
                                     cudagraph_runtime_mode=CUDAGraphMode.NONE)):
        result = owned.forward(layer, hidden, q, None, w)
    expected = reference().topk(8, dim=-1).indices
    assert torch.equal(result[:rows].cpu().sort(-1).values, expected.int().sort(-1).values)

    context = NS(attn_metadata={'probe.k_cache': metadata},
                 cudagraph_runtime_mode=CUDAGraphMode.NONE)
    with override_forward_context(context):
        for _ in range(3):
            owned.forward(layer, hidden, q, None, w)
        torch.cuda.synchronize()
        decode_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(decode_graph):
            captured = owned.forward(layer, hidden, q, None, w)
        # The same addresses now describe different request lengths and values.
        decode.decode_lens.copy_(torch.tensor([3, 1], device='cuda', dtype=torch.int32))
        decode.seq_lens.copy_(torch.tensor([[12, 16, 20], [62, 0, 0]],
                                          device='cuda', dtype=torch.int32))
        starts.copy_(torch.tensor([0, 0, 0, 32], device='cuda', dtype=torch.int32))
        ends.copy_(torch.tensor([12, 16, 20, 94], device='cuda', dtype=torch.int32))
        q.copy_(torch.randn(q.shape, device=q.device).to(q.dtype))
        w.mul_(-0.75)
        decode_graph.replay()
    expected = reference().topk(8, dim=-1).indices
    assert torch.equal(captured[:rows].cpu().sort(-1).values, expected.int().sort(-1).values)
