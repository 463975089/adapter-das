# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The breakable graph query must use HIP without disabling graph capture."""

import sys
import types

import pytest
import torch

from hcu_adapter import prefill_bcg_gate as gate
from hcu_adapter.infra import registry


@pytest.fixture
def host(monkeypatch):
    monkeypatch.delenv("HCU_DISABLE_OPS", raising=False)
    monkeypatch.setattr(torch.version, "hip", "test-hip")
    monkeypatch.setattr(registry, "_RECORDS", {})
    monkeypatch.setattr(registry, "_BOUND", {})
    modules = {}
    for name in (
        "sglang", "sglang.srt", "sglang.srt.model_executor",
        "sglang.srt.model_executor.runner_backend_utils",
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph",
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, attr = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], attr, module)
    bcg = modules[
        "sglang.srt.model_executor.runner_backend_utils."
        "breakable_cuda_graph.breakable_cuda_graph"
    ]
    bcg.is_hip = lambda: False
    return bcg


def test_hip_query_uses_hardware_truth_and_is_idempotent(host):
    bcg = host
    gate.install()
    installed = bcg.is_hip
    assert installed()
    record = registry.get("cuda_graph")
    assert record["implementation"] == "prefill-bcg-driver-truth"
    assert record["mechanism"] == registry.MECH_REBIND
    assert record["target"].endswith("breakable_cuda_graph.is_hip")
    gate.install()
    assert bcg.is_hip is installed


def test_non_hip_leaves_host_untouched(host, monkeypatch):
    bcg = host
    original = bcg.is_hip
    monkeypatch.setattr(torch.version, "hip", None)
    gate.install()
    assert bcg.is_hip is original
    assert registry.get("cuda_graph") is None


def test_registration_failure_does_not_publish_binding(host, monkeypatch):
    bcg = host
    original = bcg.is_hip

    def reject(*args, **kwargs):
        raise ValueError("registration refused")

    monkeypatch.setattr(registry, "register", reject)
    with pytest.raises(ValueError, match="registration refused"):
        gate.install()
    assert bcg.is_hip is original


def test_disabled_leaves_current_host_binding(host, monkeypatch):
    original = host.is_hip
    monkeypatch.setenv("HCU_DISABLE_OPS", "prefill_graph_driver")
    gate.install()
    assert host.is_hip is original
    assert registry.get("cuda_graph") is None
