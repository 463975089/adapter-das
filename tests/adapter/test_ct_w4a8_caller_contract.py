"""Negative controls for source-extracted caller audit; no host/GPU acceptance."""

import pytest
import torch

from tools import ct_w4a8_caller_contract as contract


def test_missing_caller_method_is_not_accepted():
    with pytest.raises(StopIteration):
        contract._load_method(b"class DeepseekV2MoE:\n    pass\n", "fixture.py")


def test_decorated_caller_is_not_silently_unwrapped():
    raw = (
        "class DeepseekV2MoE:\n"
        " @changes_semantics\n"
        " def forward_deepep(self, hidden_states, forward_batch):\n"
        "  return hidden_states\n"
    ).encode()
    with pytest.raises(ValueError, match="Decorated"):
        contract._load_method(raw, "decorator-control.py")


@pytest.mark.parametrize("multiplier", [0., 2.])
def test_missing_or_duplicate_scaling_fails_the_audit(multiplier):
    raw = (
        "class DeepseekV2MoE:\n"
        " def forward_deepep(self, hidden_states, forward_batch):\n"
        f"  return hidden_states * {multiplier}\n"
    ).encode()
    function, namespace = contract._load_method(raw, "negative-control.py")
    with pytest.raises(AssertionError, match="exactly once"):
        contract._exercise(function, namespace)


def test_cpu_audit_refuses_visible_gpu(monkeypatch):
    monkeypatch.setenv("HCU_ADAPTER_HOST", "none")
    for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4")
    with pytest.raises(RuntimeError, match="visibility"):
        contract._isolated_cpu()


def test_source_annotations_are_retained_without_host_imports():
    raw = (
        "class DeepseekV2MoE:\n"
        " def forward_deepep(self, hidden_states: MissingHostTensor,\n"
        "                    forward_batch: MissingHostBatch) -> MissingHostTensor:\n"
        "  return hidden_states\n"
    ).encode()
    function, namespace = contract._load_method(raw, "annotation-control.py")
    assert function.__annotations__["hidden_states"] == "MissingHostTensor"
    assert "MissingHostTensor" not in namespace
    value = torch.ones(1)
    assert function(None, value, None) is value
