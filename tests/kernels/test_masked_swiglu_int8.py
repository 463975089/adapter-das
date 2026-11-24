# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""FP32 legacy SwiGLU contract; padding contents are intentionally unspecified.

Run on HCU: pytest tests/kernels/test_masked_swiglu_int8.py -q
For an uninstalled source file set HCU_MASKED_SWIGLU_SOURCE to its absolute
path. No product sys.path changes or installation are needed. For the legacy
bitwise gate set HCU_MASKED_SWIGLU_OLD_SOURCE to the fixed diagnostic closure
hy4_w4a8_persistent_20e14f7_diag.py from commit 20e14f7. That independently
loaded historical source is test-only, never a production dependency.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires HCU/CUDA device",
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def provider():
    path = os.environ.get("HCU_MASKED_SWIGLU_SOURCE")
    if path:
        return _load(path, "masked_swiglu_under_test")
    from hcu_adapter.features.ct_w4a8 import masked_activation
    return masked_activation


@pytest.fixture(scope="module")
def old_provider():
    path = os.environ.get("HCU_MASKED_SWIGLU_OLD_SOURCE")
    if not path:
        pytest.skip("opt in with fixed 20e14f7 diagnostic source path")
    assert Path(path).is_file()
    module = _load(path, "masked_swiglu_fixed_old_reference")
    assert module.SOURCE_COMMIT == "20e14f7d81d0fc7c63543dd875a0515d0a63e53b"
    assert module.SOURCE_SHA256 == "2d5a97d318443b73b437d9d1589c24fad39a4e10118a47b0e37440ce24dbb829"
    return module.persistent_masked_m_silu_mul_clamp_int8_quant


def _reference(value):
    # Independent torch FP32 arithmetic, never a BF16 activation store.
    gate, up = value.float().chunk(2, dim=-1)
    gate = gate.clamp(max=10)
    up = up.clamp(-10, 10)
    active = gate * torch.sigmoid(gate) * up
    maximum = active.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    codes = (active * (127 / maximum)).round().clamp(-127, 127).to(torch.int8)
    return codes, maximum / 127


def _valid(counts, capacity):
    return torch.arange(capacity, device=counts.device)[None, :] < counts[:, None]


def test_cpu_value_rejected_before_allocation(provider, monkeypatch):
    value = torch.empty(2, 3, 8, dtype=torch.bfloat16)
    counts = torch.zeros(2, dtype=torch.int32)

    def no_allocation(*args, **kwargs):
        pytest.fail("invalid input must be rejected before allocating outputs")

    monkeypatch.setattr(torch, "empty", no_allocation)
    with pytest.raises(ValueError, match="CUDA|cuda|device"):
        provider.persistent_masked_m_silu_mul_clamp_int8_quant(value, counts, limit=10)


@requires_gpu
@pytest.mark.parametrize("case", [
    "rank", "odd", "zero_hidden", "dtype", "noncontiguous", "counts_rank",
    "counts_length", "counts_dtype", "counts_device", "limit_zero",
    "limit_negative", "limit_nan", "limit_inf", "parallel_zero",
    "parallel_negative", "parallel_float", "parallel_bool",
])
def test_static_invalid_contract_before_allocation(provider, monkeypatch, case):
    value = torch.empty(2, 3, 8, dtype=torch.bfloat16, device="cuda")
    counts = torch.zeros(2, dtype=torch.int32, device="cuda")
    kwargs = dict(limit=10, parallel_tokens=8)
    if case == "rank": value = value[0]
    elif case == "odd": value = value[..., :7].contiguous()
    elif case == "zero_hidden": value = value[..., :0]
    elif case == "dtype": value = value.float()
    elif case == "noncontiguous": value = value.transpose(0, 1)
    elif case == "counts_rank": counts = counts[:, None]
    elif case == "counts_length": counts = counts[:1]
    elif case == "counts_dtype": counts = counts.long()
    elif case == "counts_device": counts = counts.cpu()
    elif case.startswith("limit_"):
        kwargs["limit"] = {"zero": 0, "negative": -1, "nan": float("nan"),
                           "inf": float("inf")}[case[6:]]
    else:
        kwargs["parallel_tokens"] = {"zero": 0, "negative": -1,
                                     "float": 1.5, "bool": True}[case[9:]]

    def forbidden(*args, **kwargs):
        pytest.fail("static validation must precede allocation and counts readback")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(provider._persistent_masked_m_silu_mul_clamp_int8_quant,
                        "run", forbidden)
    with pytest.raises((ValueError, TypeError)):
        provider.persistent_masked_m_silu_mul_clamp_int8_quant(value, counts, **kwargs)


@requires_gpu
@pytest.mark.parametrize("experts,capacity", [(0, 512), (16, 0)])
def test_empty_shape_returns_without_kernel(provider, monkeypatch, experts, capacity):
    value = torch.empty(experts, capacity, 4096, dtype=torch.bfloat16, device="cuda")
    counts = torch.zeros(experts, dtype=torch.int32, device="cuda")

    def no_launch(*args, **kwargs):
        pytest.fail("empty expert/token dimension must not launch a kernel")

    monkeypatch.setattr(provider._persistent_masked_m_silu_mul_clamp_int8_quant,
                        "run", no_launch)
    q, s = provider.persistent_masked_m_silu_mul_clamp_int8_quant(value, counts, limit=10)
    assert q.shape == (experts, capacity, 2048)
    assert s.shape == (experts, capacity, 1)
    assert q.dtype == torch.int8 and s.dtype == torch.float32
    assert q.device == value.device == s.device


@requires_gpu
@pytest.mark.parametrize("parallel_tokens", [8, 64])
@pytest.mark.parametrize("hidden", [17, 2048])
def test_fp32_math_clamp_zero_and_strided_counts(provider, parallel_tokens, hidden):
    rng = torch.Generator(device="cuda").manual_seed(251)
    value = torch.randn(3, 17, hidden * 2, device="cuda", dtype=torch.bfloat16,
                        generator=rng) * 13
    value[0, 0].zero_()
    value[0, 1, :5] = torch.tensor([-100, -10, 0, 10, 100], device="cuda")
    value[0, 1, hidden:hidden + 5] = torch.tensor([-100, 100, 0, -100, 100], device="cuda")
    backing = torch.tensor([17, -999, 1, -999, 0, -999], device="cuda", dtype=torch.int32)
    counts = backing[::2]
    valid = _valid(counts, 17)
    value[~valid] = float("nan")
    q, s = provider.persistent_masked_m_silu_mul_clamp_int8_quant(
        value, counts, limit=10, parallel_tokens=parallel_tokens)
    expected_q, expected_s = _reference(value)
    # exp implementations may cross an isolated quantization midpoint; the
    # exact historical kernel gate below separately forbids mathematical drift.
    assert (q[valid].short() - expected_q[valid].short()).abs().max() <= 1
    torch.testing.assert_close(s[valid], expected_s[valid], rtol=2e-6, atol=1e-15)
    assert torch.equal(q[0, 0], torch.zeros(hidden, device="cuda", dtype=torch.int8))
    torch.testing.assert_close(s[0, 0], torch.tensor([1e-10 / 127], device="cuda"), rtol=1e-6, atol=0)
    # A BF16 activation intermediate changes scales: reject that mutation.
    if hidden == 2048:
        gate, up = value.float().chunk(2, -1)
        rounded = (gate.clamp(max=10) * torch.sigmoid(gate.clamp(max=10)) * up.clamp(-10, 10)).bfloat16().float()
        premature_s = rounded.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
        assert not torch.allclose(s[valid], premature_s[valid], rtol=2e-6, atol=1e-15)


@requires_gpu
def test_old_p64_codes_and_scales_bitwise(provider, old_provider):
    value = torch.randn(16, 512, 4096, dtype=torch.bfloat16, device="cuda") * 7
    counts = torch.full((16,), 512, dtype=torch.int32, device="cuda")
    old = old_provider(value, counts, limit=10, parallel_tokens=64)
    for parallel in (8, 64):
        new = provider.persistent_masked_m_silu_mul_clamp_int8_quant(
            value, counts, limit=10, parallel_tokens=parallel)
        assert torch.equal(new[0], old[0])
        assert torch.equal(new[1], old[1])


@requires_gpu
def test_same_capture_changes_counts_and_inputs(provider, monkeypatch):
    value = torch.empty(16, 512, 4096, dtype=torch.bfloat16, device="cuda")
    count_storage = torch.zeros(32, dtype=torch.int32, device="cuda")
    counts = count_storage[::2]
    value.normal_()
    run = provider.persistent_masked_m_silu_mul_clamp_int8_quant
    # Compile with nonzero counts, then capture with all-zero counts. If the
    # wrapper scans counts or specializes on their current content, replay fails.
    counts.fill_(1)
    run(value, counts, limit=10, parallel_tokens=64)
    counts.zero_()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    def no_item(*args, **kwargs):
        pytest.fail("provider must never read device counts on the host")

    monkeypatch.setattr(torch.Tensor, "item", no_item)
    with torch.cuda.graph(graph):
        captured = run(value, counts, limit=10, parallel_tokens=64)
    monkeypatch.undo()
    rng = torch.Generator(device="cuda").manual_seed(309)
    for n in (0, 1, 17, 128, 512, 0):
        for hot in (False, True):
            count_storage.fill_(-999)
            counts.zero_()
            if hot: counts[7] = n
            else: counts.fill_(n)
            value.normal_(generator=rng)
            valid = _valid(counts, 512)
            value[~valid] = float("nan")
            eager = run(value, counts, limit=10, parallel_tokens=64)
            # Poison preexisting valid output rows to expose omitted replay writes.
            captured[0].fill_(-128)
            captured[1].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured[0][valid], eager[0][valid])
            assert torch.equal(captured[1][valid], eager[1][valid])
            if n:
                expected_q, expected_s = _reference(value)
                assert (captured[0][valid].short() - expected_q[valid].short()).abs().max() <= 1
                torch.testing.assert_close(captured[1][valid], expected_s[valid], rtol=2e-6, atol=1e-15)
