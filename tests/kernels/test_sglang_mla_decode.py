# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Exercise community Triton MLA decode with the real compressed-cache layout.

This complements upstream's contiguous, split-four grouped-kernel comparisons:
32 heads, shared 576-wide K / 512-wide V, physical remapping, empty splits, and
same-address graph updates. It is a component contract, not model acceptance.
"""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU and community SGLang"
)


def _reference(q, cache, indices, length, scale):
    # Independent dense CPU FP32 softmax, without either device decode kernel.
    keys = cache.cpu().float()[indices.cpu().long()[:length], 0]
    scores = q.cpu().float()[0] @ keys.T * scale
    return (scores.softmax(dim=-1) @ keys[:, :512]).unsqueeze(0)


@pytest.mark.parametrize("length", [33, 437, 460])
def test_compressed_mla_cache_remap_and_graph(length):
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

    torch.manual_seed(1741 + length)
    heads, width, value_width, capacity, max_splits = 32, 576, 512, 1024, 128
    scale = 0.07216878364870322
    q = torch.empty((1, heads, width), device="cuda", dtype=torch.bfloat16)
    cache = torch.empty((capacity, 1, width), device="cuda", dtype=q.dtype)
    values = cache[..., :value_width]
    assert values.stride(0) == width and not values.is_contiguous()
    indices = torch.empty(length, device="cuda", dtype=torch.int32)
    indptr = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    splits = torch.tensor([7], device="cuda", dtype=torch.int32)
    partial = torch.empty(
        (1, heads, max_splits, value_width), device="cuda", dtype=torch.float32
    )
    lse = torch.empty((1, heads, max_splits), device="cuda", dtype=torch.float32)
    output = torch.empty((1, heads, value_width), device="cuda", dtype=q.dtype)

    def run(v=values, dest=output):
        decode_attention_fwd(
            q, cache, v, dest, indptr, indices, partial, lse, splits,
            max_splits, scale, 1.0, 1.0, has_mla=True, page_size=1,
        )

    def update(active_length, split_count):
        q.normal_()
        # Unreferenced physical rows and unused scratch slots stay poisoned.
        cache.fill_(float("nan"))
        mapping = torch.randperm(capacity, device="cuda")[:length].int()
        indices.copy_(mapping)
        rows = torch.randn(
            (active_length, 1, width), device="cuda", dtype=q.dtype
        )
        cache.index_copy_(0, mapping[:active_length].long(), rows)
        indptr[1:].fill_(active_length)
        splits.fill_(split_count)
        partial.fill_(float("nan"))
        lse.fill_(float("nan"))
        output.fill_(float("nan"))

    def check(active_length):
        actual = output.cpu().float()
        reference = _reference(q, cache, indices, active_length, scale)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, atol=1e-2, rtol=1e-2)
        rel = (actual - reference).norm() / reference.norm()
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), reference.flatten(), dim=0
        )
        assert rel < 1e-2, f"relative L2: {rel}"
        assert cosine > 0.9999, f"cosine: {cosine}"
        # Copying V preserves the math but removes its physical row pitch.
        # Equality isolates stride handling from reference rounding differences.
        contiguous_output = torch.empty_like(output)
        run(values.contiguous(), contiguous_output)
        assert torch.equal(output, contiguous_output)
        return actual

    update(length, 7)
    run()
    initial = check(length)
    # Warm the exact captured signature on a side stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    graph.replay()
    assert torch.equal(check(length), initial)
    for active_length, split_count in ((length - 1, 1), (length, 7)):
        update(active_length, split_count)
        graph.replay()
        changed = check(active_length)
        assert not torch.equal(changed, initial), "graph replay used stale inputs"
