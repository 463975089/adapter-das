# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Sparse prefill ABI contracts. CPU doubles do NOT qualify the DTK kernel.

Run CPU wiring/preparation: pytest tests/kernels/test_mla_sparse_prefill.py -k 'not native'
Run native numerical/Graph gates on an idle HCU: pytest tests/kernels/test_mla_sparse_prefill.py -k native -v
Reference: FP32 dot/softmax, explicit zero-valued sink, duplicate selectors retained.
"""
import math

import pytest
import torch

from hcu_adapter.kernels import mla_sparse_prefill_dtk as kernel


def inputs(rows=3, heads=4, width=128, kv_rows=129, device='cpu', lengths=True):
    generator = torch.Generator().manual_seed(311)
    q = (torch.randn(rows, heads, 512, generator=generator) * .3).bfloat16().to(device)
    kv = (torch.randn(kv_rows, 1, 512, generator=generator) * .3).bfloat16().to(device)
    ids = (torch.arange(rows * width).reshape(rows, width) % (kv_rows - 1)).int().to(device)
    ids[:, 1] = ids[:, 0]  # Keep duplicated entries in the softmax multiset.
    ids[:, 3] = -1
    lens = ((torch.arange(rows) * 7 % (width - 1)) + 5).clamp(max=width).int().to(device)
    sink = torch.linspace(-1, 1, heads + 4, device=device)[::1]
    return dict(q=q, kv=kv, indices=ids, topk_length=lens if lengths else None,
                scale=1 / math.sqrt(512), head_dim=512, nope_head_dim=448,
                rope_head_dim=64, attn_sink=sink, output=torch.full_like(q, 9.))


def reference(a):
    q, kv = a['q'].cpu().float(), a['kv'].cpu().float()
    ids = a['indices'].cpu().reshape(q.shape[0], -1)
    lens = a['topk_length']
    sink = a['attn_sink']
    result = torch.empty_like(q)
    for row in range(q.shape[0]):
        limit = ids.shape[1] if lens is None else int(lens[row])
        valid = [int(x) for x in ids[row, :limit] if 0 <= int(x) < kv.shape[0]]
        keys = kv[valid, 0]
        scores = q[row] @ keys.T * a['scale']
        values = keys
        if sink is not None:
            scores = torch.cat((scores, sink[:q.shape[1]].cpu()[:, None]), dim=1)
            values = torch.cat((values, torch.zeros(1, 512)), dim=0)
        result[row] = torch.nan_to_num(torch.softmax(scores, dim=-1)) @ values
    return result


def vector_provider(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None):
    """CPU provider double: independent vectorization of vendor input contract."""
    flat = indices[:, 0].long()
    valid = (flat >= 0) & (flat < kv.shape[0])
    if topk_length is not None:
        valid &= torch.arange(flat.shape[1])[None, :] < topk_length[:, None]
    bank = kv[:, 0][flat.clamp(0, kv.shape[0] - 1)].float()
    bank = torch.where(valid[..., None], bank, 0.)
    scores = torch.einsum('mhd,mkd->mhk', q.float(), bank) * sm_scale
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    if attn_sink is not None:
        scores = torch.cat((scores, attn_sink[None, :, None].expand(q.shape[0], -1, 1)), -1)
        bank = torch.cat((bank, torch.zeros(q.shape[0], 1, 512)), 1)
    weights = torch.nan_to_num(torch.softmax(scores, -1))
    return torch.einsum('mhk,mkd->mhd', weights, bank).to(q.dtype), None, None


def assert_numerics(out, expected):
    out, expected = out.cpu().float(), expected.float()
    assert torch.isfinite(out).all()
    cosine = torch.nn.functional.cosine_similarity(out.flatten(), expected.flatten(), dim=0)
    # Elementwise max-relative error; only exact-zero denominators are guarded.
    maxrel = ((out - expected).abs() / expected.abs().clamp_min(torch.finfo(torch.float32).tiny)).max()
    assert cosine >= .9999, float(cosine)
    assert maxrel <= .05, float(maxrel)


@pytest.mark.parametrize('rows,heads', [(1,4), (17,8), (64,16), (129,4), (3,64), (1,128)])
@pytest.mark.parametrize('lengths', [False, True])
def test_cpu_preparation_contract(monkeypatch, rows, heads, lengths):
    monkeypatch.setattr(kernel, '_supported_device', lambda q: True)
    a = inputs(rows, heads, lengths=lengths)
    original_ids = a['indices'].clone()
    expected = reference(a)
    assert kernel.try_prefill(**a, provider=vector_provider)
    assert_numerics(a['output'], expected)
    assert torch.equal(a['indices'], original_ids)


@pytest.mark.parametrize('lengths', [False, True])
def test_caller_indices_protected_from_vendor_sort(monkeypatch, lengths):
    monkeypatch.setattr(kernel, '_supported_device', lambda q: True)
    a = inputs(lengths=lengths)
    before = a['indices'].clone()
    def mutate(q, kv, ids, *args):
        ids.fill_(55)
        return torch.zeros_like(q), None, None
    assert kernel.try_prefill(**a, provider=mutate)
    assert torch.equal(a['indices'], before)


def test_mask_nan_padding_and_empty_rows(monkeypatch):
    monkeypatch.setattr(kernel, '_supported_device', lambda q: True)
    a = inputs()
    a['kv'][-1] = torch.nan
    a['topk_length'][0] = 0
    a['indices'][:, 10:] = a['kv'].shape[0] - 1
    a['topk_length'][1:] = 5
    assert kernel.try_prefill(**a, provider=vector_provider)
    assert_numerics(a['output'], reference(a))
    assert torch.count_nonzero(a['output'][0]) == 0


@pytest.mark.parametrize('change', ['ragged', 'dims', 'dtype', 'width', 'heads', 'length_dtype', 'empty', 'sink'])
def test_unsupported_falls_back_without_writes(monkeypatch, change):
    monkeypatch.setattr(kernel, '_supported_device', lambda q: True)
    a = inputs()
    if change == 'ragged': a['ragged_indices'] = torch.zeros(1)
    if change == 'dims': a['head_dim'] = 576
    if change == 'dtype': a['q'] = a['q'].float()
    if change == 'width': a['indices'] = a['indices'][:, :-1]
    if change == 'heads': a['q'] = a['q'][:, :3]; a['output'] = a['output'][:, :3]
    if change == 'length_dtype': a['topk_length'] = a['topk_length'].long()
    if change == 'empty': a['q'] = a['q'][:0]; a['output'] = a['output'][:0]
    if change == 'sink': a['attn_sink'] = torch.zeros(1)
    original = a['output'].clone()
    assert not kernel.try_prefill(**a, provider=lambda *args: pytest.fail('unexpected launch'))
    assert torch.equal(a['output'], original)


def test_real_cpu_input_is_not_admitted():
    a = inputs()
    assert not kernel.try_prefill(**a, provider=lambda *args: pytest.fail('unexpected launch'))


def test_vendor_errors_propagate(monkeypatch):
    monkeypatch.setattr(kernel, '_supported_device', lambda q: True)
    a = inputs()
    def broken(*args): raise RuntimeError('vendor launch failure')
    with pytest.raises(RuntimeError, match='vendor launch failure'):
        kernel.try_prefill(**a, provider=broken)
    assert (a['output'] == 9).all()


@pytest.fixture
def native_provider():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires native HCU; CPU tests do not qualify vendor math')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] not in kernel.SUPPORTED_ARCHES:
        pytest.skip('requires gfx936/gfx938')
    # A missing/broken vendor on an admitted HCU is a failure, never a skip.
    from flash_mla.flash_mla_interface import flash_mla_sparse_fwd
    return flash_mla_sparse_fwd


@pytest.mark.parametrize('rows,heads,kv_rows', [(1,4,7), (17,8,129), (64,16,257), (129,4,1023), (3,64,129), (1,128,129)])
@pytest.mark.parametrize('lengths', [False, True])
def test_native_numerical_matrix(native_provider, rows, heads, kv_rows, lengths):
    a = inputs(rows, heads, kv_rows=kv_rows, device='cuda', lengths=lengths)
    before = {k: v.clone() for k, v in a.items() if isinstance(v, torch.Tensor) and k != 'output'}
    expected = reference(a)
    assert kernel.try_prefill(**a, provider=native_provider)
    assert_numerics(a['output'], expected)
    for key, value in before.items():
        assert torch.equal(a[key], value), key


def test_native_graph_fresh_inputs(native_provider):
    a = inputs(17, 8, device='cuda')
    for _ in range(3):
        assert kernel.try_prefill(**a, provider=native_provider)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert kernel.try_prefill(**a, provider=native_provider)
    graph.replay()  # Capture itself does not execute the graph.
    assert_numerics(a['output'], reference(a))
    for step in range(3):
        a['q'].add_(.05)
        a['kv'].mul_(-.8)
        a['indices'].copy_((a['indices'] + step + 2).remainder(a['kv'].shape[0]))
        a['topk_length'].add_(1)
        graph.replay()
        assert_numerics(a['output'], reference(a))
