"""Top-k tie, scalar/batched parameter and graph replay contracts."""

import pytest
import torch

from sgl_kernel.sampling import top_k_renorm_prob


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    return request.param


@pytest.mark.parametrize("parameter", [2, torch.tensor(2), torch.tensor([2, 2])])
def test_top_k_ties_keep_exactly_k_entries(device, parameter):
    probs = torch.tensor([[0.25] * 4, [0.5, 0.25, 0.25, 0]], device=device)
    actual = top_k_renorm_prob(probs, parameter)
    expected = torch.tensor([[0.5, 0.5, 0, 0], [2 / 3, 1 / 3, 0, 0]], device=device)
    torch.testing.assert_close(actual, expected)
    assert torch.equal((actual > 0).sum(-1), torch.tensor([2, 2], device=device))


def test_top_k_per_row_parameters_and_clamping(device):
    probs = torch.full((4, 4), 0.25, device=device)
    ks = torch.tensor([0, 1, 3, 9], device=device)
    expected = torch.tensor(
        [[1, 0, 0, 0], [1, 0, 0, 0], [1 / 3, 1 / 3, 1 / 3, 0], [0.25] * 4],
        device=device,
    )
    torch.testing.assert_close(top_k_renorm_prob(probs, ks), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_top_k_graph_replay_uses_fresh_probs_and_k():
    probs = torch.full((2, 4), 0.25, device="cuda")
    ks = torch.tensor([1, 2], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            top_k_renorm_prob(probs, ks)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = top_k_renorm_prob(probs, ks)
    for values, counts, expected in [
        ([[0.1, 0.3, 0.3, 0.3], [0.4, 0.4, 0.1, 0.1]], [2, 1],
         [[0, 0.5, 0.5, 0], [1, 0, 0, 0]]),
        ([[0.25] * 4, [0.1, 0.1, 0.4, 0.4]], [1, 2],
         [[1, 0, 0, 0], [0, 0, 0.5, 0.5]]),
    ]:
        probs.copy_(torch.tensor(values, device="cuda"))
        ks.copy_(torch.tensor(counts, device="cuda"))
        graph.replay()
        torch.testing.assert_close(actual, torch.tensor(expected, device="cuda"))
