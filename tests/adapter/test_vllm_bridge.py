"""vLLM custom-op compatibility bridge used by the sglang host chain.

Run: python3 -m pytest -q tests/adapter/test_vllm_bridge.py
"""

from __future__ import annotations

import inspect
import sys
from types import ModuleType

import torch

from hcu_adapter.bridges import vllm as bridge


def _install_with_fake_custom_ops(monkeypatch):
    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    custom_ops = ModuleType("vllm._custom_ops")
    fake_vllm._custom_ops = custom_ops
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", custom_ops)

    assert bridge.install_rms_norm_bridge() is True
    return custom_ops


def _rms_reference(value, weight, epsilon):
    value_fp32 = value.float()
    return (
        value_fp32
        * torch.rsqrt(value_fp32.square().mean(dim=-1, keepdim=True) + epsilon)
        * weight.float()
    ).to(value.dtype)


def test_plain_rms_norm_preserves_the_vllm_destination_signature(monkeypatch):
    custom_ops = _install_with_fake_custom_ops(monkeypatch)
    assert list(inspect.signature(custom_ops.rms_norm).parameters) == [
        "out", "input", "weight", "epsilon"
    ]
    value = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32)
    weight = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
    out = torch.empty_like(value)

    result = custom_ops.rms_norm(out, value, weight, 1e-6)

    assert result is None
    torch.testing.assert_close(out, _rms_reference(value, weight, 1e-6))


def test_fused_rms_norm_writes_distinct_destinations(monkeypatch):
    custom_ops = _install_with_fake_custom_ops(monkeypatch)
    assert list(inspect.signature(custom_ops.fused_add_rms_norm).parameters) == [
        "out", "input", "residual_out", "residual", "weight", "epsilon"
    ]
    value = torch.tensor([[1.0, 2.0, -1.0, -2.0]], dtype=torch.float32)
    residual = torch.tensor([[0.5, -0.5, 1.5, -1.5]], dtype=torch.float32)
    residual_before = residual.clone()
    residual_out = torch.empty_like(residual)
    weight = torch.tensor([1.0, 0.5, 1.5, 2.0], dtype=torch.float32)
    out = torch.empty_like(value)
    summed = residual_before + value

    result = custom_ops.fused_add_rms_norm(
        out, value, residual_out, residual, weight, 1e-6
    )

    assert result is None
    torch.testing.assert_close(residual, residual_before)
    torch.testing.assert_close(residual_out, summed)
    torch.testing.assert_close(out, _rms_reference(summed, weight, 1e-6))


def test_fused_rms_norm_supports_aliased_residual_destination(monkeypatch):
    custom_ops = _install_with_fake_custom_ops(monkeypatch)
    value = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    residual = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
    weight = torch.ones(2, dtype=torch.float32)
    out = torch.empty_like(value)
    summed = residual.clone() + value

    custom_ops.fused_add_rms_norm(
        out, value, residual, residual, weight, 1e-6
    )

    torch.testing.assert_close(residual, summed)
    torch.testing.assert_close(out, _rms_reference(summed, weight, 1e-6))


def test_sglang_calls_vllm_bridge_before_grammar_bridge():
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "hcu_adapter"
        / "bridges"
        / "sglang.py"
    ).read_text(encoding="utf-8")

    rms_call = source.index("_vllm_bridge.install_rms_norm_bridge()")
    grammar_bridge = source.index("apply_token_bitmask_inplace_cuda")
    assert rms_call < grammar_bridge
