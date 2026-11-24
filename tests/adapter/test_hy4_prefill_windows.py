# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU routing for bounded full-logit windows; GPU numerics are separate."""
import torch
import pytest
from types import SimpleNamespace as NS


def test_prefill_can_delegate_bounded_full_result_without_compaction():
    from hcu_adapter.features.dsa.hy4_sparse_indexer import make_prefill
    q = torch.ones((2, 2, 4), dtype=torch.float8_e4m3fn)
    keys = torch.ones((8, 4), dtype=q.dtype)
    scale = torch.ones(8)
    weights = torch.ones((2, 2))
    starts = torch.tensor([0, 4], dtype=torch.int32)
    ends = torch.tensor([4, 8], dtype=torch.int32)
    full = torch.arange(16).float().reshape(2, 8)
    calls = []

    def provider(*args):
        calls.append('provider')
        return full

    class Windows:
        def try_prefill(self, function, *args):
            calls.append('window')
            return function(*args)

    def compact(*args):
        raise AssertionError('bounded window must not be copied')

    function = make_prefill(provider, compact, 64, windows=Windows())
    assert function((q, None), (keys, scale), weights, starts, ends, 4) is full
    assert calls == ['window', 'provider']


def fixture(rows=2, budget=64, topk=None):
    from hcu_adapter.features.dsa.hy4_sparse_indexer import make_prefill, PrefillWindows
    calls = []
    if topk is None:
        def topk(logits, starts, ends, *args, **kwargs):
            calls.append((logits, starts, ends))
            return logits
    windows = PrefillWindows(NS(top_k_per_row_prefill=topk, untouched='native'))
    q = torch.ones((rows, 2, 4), dtype=torch.float8_e4m3fn)
    keys = torch.ones((8, 4), dtype=q.dtype)
    scale, weights = torch.ones(8), torch.ones((rows, 2))
    starts = (torch.arange(rows, dtype=torch.int32) % 2) * 4
    ends = starts + 4
    def provider(query, *args):
        return torch.arange(8).float().repeat(query.shape[0], 1)
    def compact(scores, output, begin, end, offset):
        cols = begin[:, None].long() + torch.arange(output.shape[1])
        output[offset:offset+len(scores)] = scores.gather(1, cols)
    prefill = make_prefill(provider, compact, budget, windows=windows)
    def run(dcp_world_size=1, candidate_blocks=None, use_fp4_cache=False):
        scores = prefill((q, None), (keys, scale), weights, starts, ends, 4)
        return windows.top_k_per_row_prefill(scores, torch.zeros_like(starts), ends-starts)
    return windows, run, calls, starts, ends


def test_direct_topk_receives_original_absolute_windows_and_delegates_ops():
    windows, run, calls, starts, ends = fixture()
    result = windows.scope(run)()
    assert result.shape == (2, 8)
    assert calls[0][1] is starts and calls[0][2] is ends
    assert windows.untouched == 'native' and windows.current.get() is None


@pytest.mark.parametrize('kwargs', [dict(dcp_world_size=2),
    dict(candidate_blocks=object()), dict(use_fp4_cache=True)])
def test_other_layouts_keep_compacted_scores(kwargs):
    windows, run, calls, starts, ends = fixture()
    result = windows.scope(run)(**kwargs)
    assert result.shape == (2, 4)
    assert torch.equal(calls[0][1], torch.zeros_like(starts))
    assert torch.equal(calls[0][2], ends-starts)
    assert windows.current.get() is None


def test_budget_overflow_retains_bounded_compaction():
    windows, run, calls, starts, ends = fixture(rows=4, budget=64)
    result = windows.scope(run)()
    assert result.shape == (4, 4)
    assert torch.equal(calls[0][1], torch.zeros_like(starts))


def test_without_native_scope_retains_compaction():
    windows, run, calls, starts, ends = fixture()
    assert run().shape == (2, 4)
    assert windows.current.get() is None


def test_topk_exception_does_not_leak_pending_state():
    def fail(*args):
        raise ValueError('native top-k failure')
    windows, run, _, _, _ = fixture(topk=fail)
    with pytest.raises(ValueError, match='native top-k failure'):
        windows.scope(run)()
    assert windows.current.get() is None


def test_reentrant_scope_restores_parent_pending_window():
    windows, child, calls, _, _ = fixture()
    def parent(dcp_world_size=1, candidate_blocks=None, use_fp4_cache=False):
        state = windows.current.get()
        marker = object()
        state.pending = marker
        windows.scope(child)()
        assert windows.current.get() is state and state.pending is marker
    windows.scope(parent)()
    assert len(calls) == 1 and windows.current.get() is None


def test_changed_native_signature_refuses_activation():
    windows, _, _, _, _ = fixture()
    with pytest.raises(RuntimeError, match='anchors changed'):
        windows.scope(lambda x: x)


def test_wrong_logit_owner_is_rejected():
    windows, _, _, _, _ = fixture()
    def body(dcp_world_size=1, candidate_blocks=None, use_fp4_cache=False):
        windows.current.get().pending = (object(), object(), object())
        windows.top_k_per_row_prefill(object(), None, None)
    with pytest.raises(RuntimeError, match='ownership'):
        windows.scope(body)()
    assert windows.current.get() is None


@pytest.mark.parametrize('budget,columns', [(256, 8), (64, 4)])
@pytest.mark.parametrize('rank,multiple_chunks', [(None, False), (0, True), (1, True), (2, True)])
def test_complete_native_prefill_body_uses_owned_windows(budget, columns, rank, multiple_chunks):
    """Execute Core's real loop, isolating device kernels and workspace leaves."""
    import inspect
    from vllm.model_executor.layers import sparse_attn_indexer as native
    from hcu_adapter.infra.native_namespace import bind_native
    from hcu_adapter.features.dsa.hy4_sparse_indexer import make_prefill, PrefillWindows
    from vllm.config import CUDAGraphMode
    starts = torch.tensor([0, 0, 0, 4, 4, 4], dtype=torch.int32)
    ends = torch.tensor([3, 4, 4, 7, 8, 8], dtype=torch.int32)
    keys = torch.arange(32).reshape(8, 4).to(torch.float8_e4m3fn)
    scales = torch.ones((8, 1))
    q = torch.ones((6, 2, 4), dtype=keys.dtype)
    weights = torch.ones((6, 2))
    chunk = NS(cu_seqlen_ks=starts, cu_seqlen_ke=ends, max_seqlen_k=4,
        token_start=0, token_end=6, local_cu_seq_lens=torch.tensor([0, 4, 8]),
        max_local_total_seq_lens=8, local_total_seq_lens=8, skip_kv_gather=True,
        pcp_deinterleave_idx=None)
    chunks = [chunk]
    if multiple_chunks:
        chunks = [NS(**(vars(chunk) | dict(token_start=lo, token_end=hi,
            cu_seqlen_ks=starts[lo:hi], cu_seqlen_ke=ends[lo:hi])))
            for lo, hi in [(0, 3), (3, 6)]]
    metadata = native.DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([4, 4]), max_seq_len=4,
        slot_mapping=torch.arange(6), num_decodes=0, num_decode_tokens=0,
        num_prefills=2, num_prefill_tokens=6,
        prefill=NS(chunks=chunks, row_shard_sizes=None if rank is None else [2, 4, 0]))
    context = NS(attn_metadata={'layer': metadata}, cudagraph_runtime_mode=CUDAGraphMode.NONE)
    topk_calls = []
    def topk(logits, begin, end, output, rows, stride0, stride1, count):
        topk_calls.append((logits.shape[1], begin.clone(), end.clone()))
        for row in range(rows):
            lo, hi = int(begin[row]), int(end[row])
            output[row] = logits[row, lo:hi].topk(count).indices.int()
    windows = PrefillWindows(NS(top_k_per_row_prefill=topk))
    def provider(query, k, sf, w, *args):
        return torch.einsum('mhd,nd->mhn', query.float(), k.float()).relu().mul(
            w[..., None]).sum(1).mul(sf.squeeze(-1))
    def compact(scores, output, begin, end, offset):
        cols = torch.arange(output.shape[1])[None] + begin[:, None]
        output[offset:offset+len(scores)] = scores.gather(1, cols.long()).masked_fill(
            cols >= end[:, None], -float('inf'))
    original = inspect.unwrap(native.sparse_attn_indexer)
    original_ops = original.__globals__['ops']
    exchanges = []
    class CPUShard(native.PrefillRowShard):
        @classmethod
        def from_metadata(cls, prefill, num_decodes):
            return None if rank is None else cls(num_decodes, prefill.row_shard_sizes, rank)
        def exchange_topk(self, buffer, width):
            exchanges.append((self.start, self.stop, width))
    body = bind_native(original, {
        'get_forward_context': lambda: context,
        'current_platform': NS(fp8_dtype=lambda: torch.float8_e4m3fn, is_xpu=lambda: False),
        'current_workspace_manager': lambda: NS(get_simultaneous=lambda *args: (keys, scales.view(torch.uint8))),
        'mqa_logits_use_mx_scales': lambda: False,
        'mqa_logits_mx_inputs': lambda query, scale, w: (scale, w),
        'fp8_fp4_mqa_logits': make_prefill(provider, compact, budget, windows=windows),
        'PrefillRowShard': CPUShard,
        'ops': windows})
    output = torch.full((6, 2), -99, dtype=torch.int32)
    result = windows.scope(body)(torch.ones(6, 4), 'layer', torch.empty(0), q,
        None, None, weights, 128, 'ue8m0', 2, 4, 8, 8, output, True,
        False, False, '')
    assert result is output
    expected = torch.tensor([[2, 1], [3, 2], [3, 2], [2, 1], [3, 2], [3, 2]], dtype=torch.int32)
    active = (0, 6) if rank is None else [(0, 2), (2, 6), (6, 6)][rank]
    expected[:active[0]] = -1
    expected[active[1]:] = -1
    assert torch.equal(output, expected)
    expected_calls = []
    for c in chunks:
        lo, hi = max(c.token_start, active[0]), min(c.token_end, active[1])
        if lo < hi:
            width = 8 if (hi-lo)*8*4 <= budget else 4
            expected_calls.append((width, starts[lo:hi] if width == 8 else torch.zeros_like(starts[lo:hi]),
                                   ends[lo:hi] if width == 8 else ends[lo:hi]-starts[lo:hi]))
    assert len(topk_calls) == len(expected_calls)
    for call, expected_call in zip(topk_calls, expected_calls):
        assert call[0] == expected_call[0]
        assert torch.equal(call[1], expected_call[1])
        assert torch.equal(call[2], expected_call[2])
    assert exchanges == ([] if rank is None else [(*active, 2)])
    assert windows.current.get() is None
    assert original.__globals__['ops'] is original_ops
