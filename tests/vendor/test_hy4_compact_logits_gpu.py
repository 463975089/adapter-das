# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Request-local layout parity, strides and changed-input graph replay."""
import pytest
import torch


@pytest.mark.parametrize('width', [0, 1, 255, 256, 257, 511, 512, 513, 1001, 84835])
def test_compact_request_windows_strides_and_graph(width):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires HCU')
    from hcu_adapter.kernels.compact_mqa_logits_triton import compact

    rows, columns = 5, max(23, width + 19)
    source = torch.randn((rows, columns + 8), device='cuda')[:, :columns]
    backing = torch.full((rows + 2, width + 12), 17., device='cuda')
    output = backing[:, 3:width + 3]
    starts = torch.tensor([0, 7, -3, columns - 2, columns + 1], device='cuda', dtype=torch.int32)
    ends = torch.tensor([columns, 9, 11, columns + 1, columns + 4], device='cuda', dtype=torch.int32)

    def check():
        absolute = starts.cpu().long()[:, None] + torch.arange(width)[None]
        valid = (absolute >= 0) & (absolute < ends.cpu()[:, None]) & (absolute < columns)
        expected = source.cpu().gather(1, absolute.clamp(0, columns - 1))
        expected.masked_fill_(~valid, -float('inf'))
        torch.testing.assert_close(output[1:rows + 1].cpu(), expected, rtol=0, atol=0)
        assert torch.all(backing[0] == 17.) and torch.all(backing[-1] == 17.)
        assert torch.all(backing[:, :3] == 17.) and torch.all(backing[:, width + 3:] == 17.)

    for _ in range(3): compact(source, output, starts, ends, 1)
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compact(source, output, starts, ends, 1)
    for offset in [2, -4, 0]:
        source.mul_(-0.5).add_(0.25)
        starts.add_(offset)
        ends.add_(offset - 1)
        graph.replay()
        check()


def test_compact_empty_query_does_not_touch_output():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires HCU')
    from hcu_adapter.kernels.compact_mqa_logits_triton import compact
    output = torch.full((2, 600), 17., device='cuda')
    compact(output[:0], output, None, None, 0)
    assert torch.all(output == 17.)
