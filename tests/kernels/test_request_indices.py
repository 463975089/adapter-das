"""Request padding reference and changing-metadata graph regression."""

import pytest
import torch

from hcu_adapter.kernels.request_indices_emu import padded_request_indices


@pytest.mark.parametrize("requests,lengths,size", [
    ([16], [6], 8),
    ([4, 8, 2], [0, 3, 1], 8),
    ([4, 8, 2], [2, 0, 0], 4),
    ([], [], 4),
    ([], [], 0),
    ([4], [0], 0),
    ([4, 8], [2, 3], 5),
])
def test_padded_indices_reference(requests, lengths, size):
    req = torch.tensor(requests, dtype=torch.int64)
    lens = torch.tensor(lengths, dtype=torch.int64)
    expected = [r for r, n in zip(requests, lengths) for _ in range(n)]
    expected += [0] * (size - len(expected))
    assert padded_request_indices(req, lens, size).tolist() == expected


@pytest.mark.parametrize("lengths,size", [([6], 4), ([-1], 8)])
def test_bad_lengths_rejected(lengths, size):
    with pytest.raises(ValueError, match="nonnegative"):
        padded_request_indices(torch.tensor([16]), torch.tensor(lengths), size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_request_padding_graph_changes_and_valid_row_isolation():
    req = torch.tensor([16, 7, 3], dtype=torch.int64, device="cuda")
    lens = torch.tensor([0, 3, 1], dtype=torch.int64, device="cuda")

    def run():
        return padded_request_indices(req, lens, 8)

    assert run().cpu().tolist() == [7, 7, 7, 3, 0, 0, 0, 0]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = run()
    graph.replay()
    torch.cuda.synchronize()
    assert output.cpu().tolist() == [7, 7, 7, 3, 0, 0, 0, 0]
    req.copy_(torch.tensor([2, 5, 9], device="cuda"))
    lens.copy_(torch.tensor([2, 0, 6], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    assert output.cpu().tolist() == [2, 2, 9, 9, 9, 9, 9, 9]
    lens.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert output.cpu().tolist() == [0] * 8
    # Dummy request zero has separate storage: padded writes cannot overwrite
    # either live request's slot when the first token of a request is position 0.
    slots = torch.zeros((17, 2), device="cuda")
    slots[7, 0], slots[3, 0] = 17, 23
    slots[output, 0] = -1
    assert slots[7, 0].item() == 17 and slots[3, 0].item() == 23
