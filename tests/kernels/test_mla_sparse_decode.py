# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Packed sparse decode ABI/preparation contracts. CPU doubles prove no vendor math.

Run: HCU_ADAPTER_HOST=none pytest tests/kernels/test_mla_sparse_decode.py -q
The provider double is not evidence that the DTK kernel, Graph or model is correct.
"""
import math

import pytest
import torch

from hcu_adapter.kernels import mla_sparse_decode_dtk as kernel


def caches(pages=5, page=32, rows=4, width=128, device='cpu'):
    generator = torch.Generator().manual_seed(907)
    swa = torch.randint(0, 256, (pages, page, 584), generator=generator,
                        dtype=torch.uint8).to(device)
    extra = torch.randint(0, 256, (pages, page, 584), generator=generator,
                          dtype=torch.uint8).to(device)
    return swa, extra


def selectors(rows, width, slots, device='cpu'):
    dense = (torch.arange(rows * width).reshape(rows, width) % slots).int().to(device)
    dense[:, 1] = dense[:, 0]          # Duplicates are distinct softmax entries.
    dense[:, 3] = -1                   # Invalid sentinels must be compacted away.
    dense[:, 5] = slots + 7            # Out-of-range sentinels too.
    lens = ((torch.arange(rows) * 13 % (width - 1)) + 4).clamp(max=width).int().to(device)
    return dense, lens


class Recorder:
    """Provider double: capture arguments, return a shaped result, never compute."""
    def __init__(self, rows, heads, native, fill=0.5):
        self.calls = []
        self.rows, self.heads, self.native, self.fill = rows, heads, native, fill

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        q = kwargs['q']
        out = torch.full((q.shape[0], 1, q.shape[2], 512), self.fill, dtype=q.dtype)
        return out, None


def factory():
    return (object(), None)


def epilogue(result, positions, cos_sin, rope_dim, data, scale):
    epilogue.calls.append((result.shape, positions.shape, rope_dim, data.shape, scale.shape))


epilogue.calls = []


def arguments(rows=4, heads=64, width=128, mxfp8=False, swa_only=False, device='cpu'):
    swa, extra = caches(rows=rows, width=width, device=device)
    dense, lens = selectors(rows, width, swa.shape[0] * swa.shape[1], device)
    q = torch.full((rows, heads, 512), .25, dtype=torch.bfloat16, device=device)
    base = dict(q=q, kv_cache=None if swa_only else extra, swa_k_cache=swa,
        swa_only=swa_only, topk_indices=None if swa_only else dense,
        topk_lens=None if swa_only else lens, swa_indices=dense, swa_lens=lens,
        swa_ragged_indices=None, swa_ragged_indptr=None,
        topk_ragged_indices=None, topk_ragged_indptr=None,
        attn_sink=torch.linspace(-1, 1, heads).float().to(device),
        scale=1 / math.sqrt(512), head_dim=512, nope_head_dim=448,
        rope_head_dim=64, output=None if mxfp8 else torch.full_like(q, 7.),
        extra_cache_nan_free=False, adaptive_splits=False,
        inv_rope_positions=torch.arange(rows, device=device).int() if mxfp8 else None,
        inv_rope_cos_sin_cache=torch.zeros(64, 64).float().to(device) if mxfp8 else None,
        output_mxfp8=(
            torch.zeros((rows, heads * 512), dtype=torch.float8_e4m3fn, device=device),
            torch.zeros((rows, heads * 16), dtype=torch.uint8, device=device),
        ) if mxfp8 else None)
    return base


@pytest.fixture
def admitted(monkeypatch):
    monkeypatch.setattr(kernel, '_supported_device', lambda q: True)
    epilogue.calls = []


@pytest.mark.parametrize('heads,native', [(4, 64), (64, 64), (128, 128), (16, 64)])
def test_head_padding_keeps_cache_bytes(admitted, heads, native):
    a = arguments(heads=heads)
    record = Recorder(a['q'].shape[0], heads, native)
    assert kernel.try_decode(**a, provider=record, metadata_factory=factory,
                             epilogue=epilogue) == 0
    call = record.calls[0]
    # No BF16 KV gather: the packed uint8 cache goes straight to the vendor.
    assert call['k_cache'].dtype == torch.uint8 and call['k_cache'].shape[2:] == (1, 584)
    assert call['is_fp8_kvcache'] is True
    assert call['q'].shape == (a['q'].shape[0], 1, native, 512)
    assert torch.equal(call['q'][:, 0, :heads], a['q'])
    assert torch.count_nonzero(call['q'][:, 0, heads:]) == 0
    assert call['attn_sink'].shape == (native,)
    assert torch.isinf(call['attn_sink'][heads:]).all()
    assert (call['attn_sink'][heads:] < 0).all()
    assert torch.equal(a['output'], torch.full_like(a['output'], .5))


def test_mxfp8_delegates_epilogue_and_returns_all_rows(admitted):
    a = arguments(heads=64, mxfp8=True)
    record = Recorder(a['q'].shape[0], 64, 64)
    assert kernel.try_decode(**a, provider=record, metadata_factory=factory,
                             epilogue=epilogue) == a['q'].shape[0]
    assert len(epilogue.calls) == 1
    shape, positions, rope_dim, data, scale = epilogue.calls[0]
    assert shape == (a['q'].shape[0], 64, 512) and rope_dim == 64
    assert positions == (a['q'].shape[0],) and data == a['output_mxfp8'][0].shape


def test_zero_rotation_is_a_success_not_a_fallback(admitted):
    a = arguments()
    record = Recorder(a['q'].shape[0], 64, 64)
    original = kernel.try_decode(**a, provider=record, metadata_factory=factory,
                                 epilogue=epilogue)
    assert original == 0 and original is not None and len(record.calls) == 1


def test_empty_rows_zero_their_output(admitted):
    a = arguments(rows=4)
    a['swa_lens'] = torch.zeros(4, dtype=torch.int32)
    a['topk_lens'] = torch.zeros(4, dtype=torch.int32)
    record = Recorder(4, 64, 64, fill=9.)
    assert kernel.try_decode(**a, provider=record, metadata_factory=factory,
                             epilogue=epilogue) == 0
    assert torch.count_nonzero(a['output']) == 0


def test_csr_selectors_are_repacked_without_host_reads(admitted):
    a = arguments(rows=3, width=128)
    slots = a['swa_k_cache'].shape[0] * a['swa_k_cache'].shape[1]
    dense, lens = selectors(3, 128, slots)
    rows, width = dense.shape
    flat = torch.empty(rows * width, dtype=torch.int32)
    indptr = torch.empty(rows + 1, dtype=torch.int32)
    # Fill CSR with the same rows the dense form carries, real lengths included.
    indptr[0] = 0
    for row in range(rows):
        count = int(lens[row])
        flat[row * width:row * width + count] = dense[row, :count]
        indptr[row + 1] = indptr[row] + count
    a['swa_indices'] = None
    a['swa_lens'] = None
    a['swa_ragged_indices'] = flat
    a['swa_ragged_indptr'] = indptr
    record = Recorder(3, 64, 64)
    assert kernel.try_decode(**a, provider=record, metadata_factory=factory,
                             epilogue=epilogue) == 0
    got, got_lens = record.calls[0]['indices'], record.calls[0]['topk_length']
    assert got.shape[0] == 3 and got.shape[-1] % 64 == 0
    for row in range(rows):
        valid = [int(x) for x in got[row, 0, :got_lens[row]] if int(x) >= 0]
        assert valid == sorted(valid), 'valid slots must precede -1 padding'
        assert len(valid) <= int(lens[row])


@pytest.mark.parametrize('change', ['dims', 'dtype', 'heads', 'cache_dtype', 'cache_rows',
                                    'adaptive', 'no_output', 'both_outputs', 'sink',
                                    'swa_missing', 'extra_missing', 'mxfp8_data',
                                    'mxfp8_scale', 'positions'])
def test_unsupported_falls_back_before_any_write(admitted, change):
    a = arguments(mxfp8=change.startswith('mxfp8') or change in ('positions', 'both_outputs'))
    if change == 'dims': a['head_dim'] = 576
    if change == 'dtype': a['q'] = a['q'].float(); a['output'] = a['output'].float()
    if change == 'heads':
        # 129 heads exceeds the validated native width (H64/H128).
        a['q'] = torch.zeros((4, 129, 512), dtype=torch.bfloat16)
        a['output'] = torch.zeros_like(a['q'])
    if change == 'cache_dtype': a['swa_k_cache'] = a['swa_k_cache'].view(torch.int8)
    if change == 'cache_rows': a['swa_k_cache'] = a['swa_k_cache'][:, :, :512]
    if change == 'adaptive': a['adaptive_splits'] = True
    if change == 'no_output': a['output'] = None
    if change == 'both_outputs':
        a['output'] = torch.zeros((4, 64, 512), dtype=torch.bfloat16)
    if change == 'sink': a['attn_sink'] = torch.zeros(1)
    if change == 'swa_missing': a['swa_indices'] = None
    if change == 'extra_missing': a['topk_indices'] = None
    if change == 'mxfp8_data':
        a['output_mxfp8'] = (torch.zeros(4, 64 * 512), a['output_mxfp8'][1])
    if change == 'mxfp8_scale':
        a['output_mxfp8'] = (a['output_mxfp8'][0], torch.zeros(4, 64))
    if change == 'positions': a['inv_rope_positions'] = None
    before = None if a['output'] is None else a['output'].clone()
    record = Recorder(4, 64, 64)
    assert kernel.try_decode(**a, provider=record, metadata_factory=factory,
                             epilogue=epilogue) is None
    assert record.calls == [] and epilogue.calls == []
    if before is not None:
        assert torch.equal(a['output'], before)


def test_swa_only_skips_the_extra_cache(admitted):
    a = arguments(swa_only=True)
    record = Recorder(a['q'].shape[0], 64, 64)
    assert kernel.try_decode(**a, provider=record, metadata_factory=factory,
                             epilogue=epilogue) == 0
    assert record.calls[0]['extra_k_cache'] is None
    assert record.calls[0]['extra_indices_in_kvcache'] is None


def test_vendor_errors_propagate(admitted):
    a = arguments()
    def broken(**kwargs): raise RuntimeError('vendor decode failure')
    with pytest.raises(RuntimeError, match='vendor decode failure'):
        kernel.try_decode(**a, provider=broken, metadata_factory=factory, epilogue=epilogue)
    assert torch.equal(a['output'], torch.full_like(a['output'], 7.))


def test_real_cpu_input_is_not_admitted():
    a = arguments()
    assert kernel.try_decode(**a, provider=Recorder(4, 64, 64),
                             metadata_factory=factory, epilogue=epilogue) is None


@pytest.fixture
def native_provider():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires native HCU; CPU doubles do not qualify vendor decode')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] not in ('gfx936', 'gfx938'):
        pytest.skip('packed decode candidate requires gfx936/gfx938')
    from flash_mla.flash_mla_interface import flash_mla_with_kvcache, get_mla_metadata
    return flash_mla_with_kvcache, get_mla_metadata


@pytest.mark.parametrize('heads', [64, 128])
def test_native_decode_graph_fresh_plans(native_provider, heads):
    provider, meta = native_provider
    rows, width = 8, 128
    a = arguments(rows=rows, heads=heads, device='cuda')
    swa, extra = caches(rows=rows, width=width, device='cuda')
    a['swa_k_cache'], a['kv_cache'] = swa, extra
    slots = swa.shape[0] * swa.shape[1]
    a['swa_indices'], a['swa_lens'] = selectors(rows, width, slots, 'cuda')
    a['topk_indices'], a['topk_lens'] = selectors(rows, width, extra.shape[0] * extra.shape[1], 'cuda')
    a['q'] = torch.randn(rows, heads, 512, device='cuda').bfloat16() * .2
    a['output'] = torch.zeros_like(a['q'])
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inverse_rope_mxfp8_rows
    for _ in range(3):
        assert kernel.try_decode(**a, provider=provider, metadata_factory=meta,
                                 epilogue=rocm_inverse_rope_mxfp8_rows) == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert kernel.try_decode(**a, provider=provider, metadata_factory=meta,
                                 epilogue=rocm_inverse_rope_mxfp8_rows) == 0
    graph.replay()
    assert torch.isfinite(a['output'].float()).all()
    for step in range(3):
        a['q'].add_(.03)
        a['swa_indices'].add_(step + 1).remainder_(slots)
        a['topk_lens'].add_(1)
        graph.replay()
        assert torch.isfinite(a['output'].float()).all()
