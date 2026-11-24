# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Actual HIP qualification for owned paged-MQA score padding."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="requires actual HIP tensors and graph replay",
)


def helper():
    path = Path(__file__).resolve().parents[2] / "hcu_adapter/kernels/_paged_mqa_mask.py"
    spec = importlib.util.spec_from_file_location("_actual_paged_mqa_mask_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.owned_logits_mask


def scores_with_pitch(rows, width):
    patterns = torch.tensor(
        [0, -2147483648, 1, -4194303, 2143289345, 2139095040,
         -8388608, 1065353216, -1082130432],
        dtype=torch.int32, device="cuda",
    )
    index = torch.arange(rows * (width + 13) + 7, device="cuda")
    storage = patterns[index % len(patterns)]
    return storage.as_strided((rows, width), (width + 13, 1), 7).view(torch.float32), storage


@pytest.mark.parametrize("width", [1, 63, 1025, 262144])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("column_lengths", [False, True])
def test_exact_bits_owned_output_and_pitched_source(width, dtype, column_lengths):
    mask = helper()
    scores, storage = scores_with_pitch(4, width)
    before = storage.clone()
    values = [-1, 0, width // 2, width + 3] if dtype == torch.int32 else [-2**40, 0, width // 2, width + 2**40]
    lengths = torch.tensor(values, dtype=dtype, device="cuda")
    if column_lengths:
        lengths = lengths[:, None]
    expected = scores.masked_fill(
        torch.arange(width, device="cuda")[None] >= lengths.reshape(-1, 1), 0,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = mask(scores, lengths)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert actual.untyped_storage().data_ptr() != storage.untyped_storage().data_ptr()
    actual.fill_(17)
    assert torch.equal(storage, before)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_graph_replay_reads_new_lengths_and_new_score_bits(dtype):
    mask = helper()
    width = 1025
    scores, _ = scores_with_pitch(4, width)
    lengths = torch.full((4, 1), width, dtype=dtype, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        mask(scores, lengths)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = mask(scores, lengths)
    pointer = actual.data_ptr()
    for values in [[-1, 0, 7, width + 1], [width, 513, 1, 0], [0, width, -8, 1]]:
        lengths.copy_(torch.tensor(values, dtype=dtype, device="cuda")[:, None])
        scores.fill_(float(values[2]))
        graph.replay()
        expected = scores.masked_fill(
            torch.arange(width, device="cuda")[None] >= lengths.reshape(-1, 1), 0,
        )
        assert actual.data_ptr() == pointer
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("shape", [(0, 17), (5, 0)])
def test_empty_shape(shape):
    mask = helper()
    scores = torch.empty(shape, dtype=torch.float32, device="cuda")
    lengths = torch.zeros(shape[0], dtype=torch.int32, device="cuda")
    actual = mask(scores, lengths)
    assert actual.shape == shape and actual.dtype == torch.float32
    assert actual.numel() == 0
