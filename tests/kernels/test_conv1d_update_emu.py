# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""causal_conv1d_update torch-emulator gate — §3.4b.

The gfx938 triton kernel returns correlation noise (cos 0.26 vs the W=4
non-spec reference; weight-flip variant 0.50 — neither matches), while this
emulator passes exactly vs the same reference. Runs on every KDA layer per
decode step (42 x steps) in GLM-5.x — the decode-side word-soup driver once
prefill chunk_kda is emulated. Also covers the in-place state slide and the
conv_state_indices continuous-batching indexing path.
"""

import math

import pytest
import torch

from hcu_adapter.conv1d_update_emu import causal_conv1d_update_emu


def _case():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(3, 96, generator=generator).bfloat16()
    weight = (torch.randn(96, 4, generator=generator) * 0.3).bfloat16()
    bias = (torch.randn(96, generator=generator) * 0.1).bfloat16()
    state = torch.randn(6, 96, 3, generator=generator).bfloat16()
    slots = torch.tensor([4, 1, 5], dtype=torch.int32)
    return x, weight, bias, state, slots


def _reference(x, weight, bias, state, slots):
    """Scalar FP64 convolution and logical cache update, independent of the emu."""
    expected = torch.empty(x.shape, dtype=torch.float64)
    after = state.clone()
    for row, slot in enumerate(slots.tolist()):
        for channel in range(x.shape[1]):
            history = state[slot, channel].double().tolist() + [float(x[row, channel])]
            value = math.fsum(a * b for a, b in zip(
                history, weight[channel].double().tolist())) + float(bias[channel])
            expected[row, channel] = value / (1 + math.exp(-value))
            after[slot, channel] = torch.tensor(history[1:], dtype=state.dtype)
    return expected, after


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_conv1d_update_output_and_indexed_state(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a real GPU")
    inputs = _case()
    expected, after = _reference(*inputs)
    x, weight, bias, state, slots = [value.to(device).clone() for value in inputs]
    actual = causal_conv1d_update_emu(
        x, state, weight, bias, "silu", conv_state_indices=slots)
    observed = actual.cpu().double()
    cosine = torch.nn.functional.cosine_similarity(
        observed.flatten(), expected.flatten(), dim=0).item()
    assert cosine >= 0.9999
    assert (observed - expected).abs().max().item() <= 2e-2
    # Comparing the entire cache also checks unselected and permuted slots.
    torch.testing.assert_close(state.cpu(), after, rtol=0, atol=0)
    for value, original in zip((x, weight, bias, slots),
                               (inputs[0], inputs[1], inputs[2], inputs[4])):
        torch.testing.assert_close(value.cpu(), original, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
