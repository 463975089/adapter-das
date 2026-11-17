# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sgl_kernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires HCU/CUDA')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('rotary_dim', [64, 96])
@pytest.mark.parametrize('key_mode', ['separate', 'same', 'none'])
def test_partial_rotary_width_preserves_packed_storage_and_fresh_graph(dtype, rotary_dim, key_mode):
    tokens, head_size, q_heads, k_heads = 17, 128, 11, 3
    q_end = 7 + q_heads * head_size
    k_end = q_end + k_heads * head_size
    generator = torch.Generator().manual_seed(731)
    batches = []
    for phase in range(3):
        packed = torch.randn(tokens, k_end + 9, generator=generator).to(dtype)
        packed[:, :7] = float('nan')
        packed[:, k_end:] = float('nan')
        positions = (torch.arange(tokens) * 7 + phase * 11) % 127
        cache = torch.randn(127, rotary_dim, generator=generator).to(dtype)
        batches.append((packed, positions, cache))
    storage, positions, cache = [v.cuda() for v in batches[0]]
    q = storage[:, 7:q_end]
    k = storage[:, q_end:k_end] if key_mode == 'separate' else q if key_mode == 'same' else None
    pointers = (q.data_ptr(), None if k is None else k.data_ptr())

    def execute():
        assert sgl_kernel.rotary_embedding(positions, q, k, head_size, cache, True) is None

    def verify(inputs, indices, coefficients):
        actual = storage.cpu()
        for start, stop, heads in [(7, q_end, q_heads)] + ([(q_end, k_end, k_heads)] if key_mode == 'separate' else []):
            before = inputs[:, start:stop].reshape(tokens, heads, head_size).double()
            observed = actual[:, start:stop].reshape(tokens, heads, head_size)
            half = rotary_dim // 2
            cosine, sine = coefficients[indices].double().chunk(2, dim=-1)
            cosine, sine = cosine[:, None], sine[:, None]
            first, second = before[..., :half], before[..., half:rotary_dim]
            expected = torch.cat((first*cosine-second*sine, first*sine+second*cosine), dim=-1)
            bound = torch.cat((first.abs()*cosine.abs()+second.abs()*sine.abs(), first.abs()*sine.abs()+second.abs()*cosine.abs()), dim=-1) * (8 * 2**-24)
            lower, upper = (expected-bound).to(dtype).double(), (expected+bound).to(dtype).double()
            rotated = observed[..., :rotary_dim].double()
            assert torch.isfinite(rotated).all()
            assert ((rotated >= lower) & (rotated <= upper)).all()
            torch.testing.assert_close(observed[..., rotary_dim:], before[..., rotary_dim:].to(dtype), rtol=0, atol=0)
        torch.testing.assert_close(actual[:, :7], inputs[:, :7], rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(actual[:, k_end:], inputs[:, k_end:], rtol=0, atol=0, equal_nan=True)
        if key_mode != 'separate':
            torch.testing.assert_close(actual[:, q_end:k_end], inputs[:, q_end:k_end], rtol=0, atol=0)
        assert torch.equal(positions.cpu(), indices)
        assert torch.equal(cache.cpu(), coefficients)
        assert pointers == (q.data_ptr(), None if k is None else k.data_ptr())
        return actual

    execute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    for inputs, indices, coefficients in batches:
        results = []
        for mode in ['eager', 'graph']:
            storage.copy_(inputs); positions.copy_(indices); cache.copy_(coefficients)
            execute() if mode == 'eager' else graph.replay()
            torch.cuda.synchronize()
            results.append(verify(inputs, indices, coefficients))
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0, equal_nan=True)
    sgl_kernel.rotary_embedding(positions[:0], q[:0], None, head_size, cache, True)
