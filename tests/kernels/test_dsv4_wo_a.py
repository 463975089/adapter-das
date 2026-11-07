"""Original grouped BF16 output projection with an independent CPU oracle."""

import pytest
import torch


def _batch(phase, rows, live):
    generator = torch.Generator().manual_seed(831 + phase)
    x = (torch.randn(rows, 2, 4096, generator=generator) / 8).bfloat16()
    weight = (torch.randn(2, 1024, 4096, generator=generator) / 64).bfloat16()
    x[live:] = torch.nan
    # Each group has its own weights; neither folding nor broadcasting the
    # group axis may substitute for the grouped contraction.
    expected = torch.stack([
        x[:live, group].double() @ weight[group].double().T
        for group in range(2)
    ], dim=1)
    return x, weight, expected


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="Requires a real HIP device",
)
@pytest.mark.parametrize("rows,live", [(4, 1), (8, 6)])
def test_grouped_wo_a_fresh_input_graph(rows, live):
    from sglang.srt.models.deepseek_v4 import _apply_wo_a_bf16_matmul

    batches = [_batch(phase, rows, live) for phase in range(3)]
    # Keep a padded token stride, as with a view of a larger attention output.
    # The unused groups must not be read as part of the next token.
    storage = torch.full((rows, 8, 4096), torch.nan, device="cuda",
                         dtype=torch.bfloat16)
    x = storage[:, :2]
    weight = batches[0][1].cuda()
    x.copy_(batches[0][0])

    def execute():
        return _apply_wo_a_bf16_matmul(
            x, weight, is_decode=rows == 4, is_prefill=rows == 8,
            is_target_verify=False, fast_path=True,
            fuse_mxfp8_quant=False, fp8_grid=False, emit_fp8=False,
        )

    execute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = execute()
    torch.cuda.synchronize()

    for mode in ("eager", "graph"):
        for inputs, weights, expected in batches:
            x.copy_(inputs)
            weight.copy_(weights)
            if mode == "graph":
                graph.replay()
                actual = captured
            else:
                actual = execute()
            assert actual.shape == (rows, 2, 1024)
            assert actual.dtype == torch.bfloat16
            observed = actual[:live].cpu().double()
            assert torch.isfinite(observed).all()
            delta = observed - expected
            cosine = torch.nn.functional.cosine_similarity(
                observed.flatten(), expected.flatten(), dim=0)
            assert cosine >= 0.9999
            assert delta.norm() / expected.norm() <= 0.005
            assert delta.abs().max() / expected.abs().max() <= 0.05
            # Padding outputs are not consumed; they may contain NaN.
            # The public inputs, including unused storage, must stay unchanged.
            torch.testing.assert_close(
                x.cpu(), inputs, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(
                weight.cpu(), weights, rtol=0, atol=0)
            assert torch.isnan(storage[:, 2:]).all()
