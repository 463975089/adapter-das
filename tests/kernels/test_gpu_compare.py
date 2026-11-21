# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Self-tests for the GPU-native compare helper (runs on any device;
the helper's math is device-agnostic, so CPU tensors exercise it fully)."""

import pytest
import torch

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _gpu_compare import gpu_compare, gpu_maxdiff  # noqa: E402


def test_pass_within_atol():
    gpu_compare(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0 + 1e-9]), atol=1e-6)


def test_fail_reports_first_mismatch_neighborhood():
    ref = torch.tensor([1.0, 2.0, 3.0, 4.0])
    got = ref.clone()
    got[2] = 9.0
    with pytest.raises(AssertionError, match=r"first@2 got=9\.0 ref=3\.0 diff=6\.0"):
        gpu_compare(got, ref, atol=0.1)


def test_rtol_rule_is_elementwise():
    gpu_compare(torch.tensor([100.0]), torch.tensor([100.9]), rtol=0.01)


def test_bf16_ulp_scale_aware_bound():
    # one bf16 ulp at |ref|~16 is 0.125: diff 0.1 passes with bf16_ulp,
    # fails with a flat atol of 0.05.
    ref = torch.tensor([16.0])
    got = torch.tensor([16.1])
    with pytest.raises(AssertionError):
        gpu_compare(got, ref, atol=0.05)
    gpu_compare(got, ref, atol=0.05, bf16_ulp=True)


def test_cosine_floor():
    gpu_compare(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1e-3]), cosine=0.9999)
    with pytest.raises(AssertionError, match="cosine"):
        gpu_compare(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), cosine=0.5)


def test_shape_mismatch_message():
    with pytest.raises(AssertionError, match="shape mismatch"):
        gpu_compare(torch.zeros(2), torch.zeros(3))


def test_gpu_maxdiff_single_sync_pair():
    assert gpu_maxdiff(torch.tensor([1.0, -4.0]), torch.tensor([0.5, -2.0])) == [2.0, 2.0]
