# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Driver gate dispatch and registration; no GPU or model acceptance claim."""

import sys
import types

import pytest
import torch

from hcu_adapter import prefill_bcg_gate as gate, registry


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test-hip")
    monkeypatch.setattr(registry, "_RECORDS", {})
    monkeypatch.setattr(registry, "_BOUND", {})
    modules = {}
    for name in (
        "sglang", "sglang.srt", "sglang.srt.model_executor",
        "sglang.srt.model_executor.model_runner",
        "sglang.srt.model_executor.model_runner_components",
        "sglang.srt.model_executor.model_runner_components.cuda_graph_setup",
        "sglang.srt.model_executor.cuda_graph_config",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, attr = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], attr, module)
    prefix = "sglang.srt.model_executor."
    mr = modules[prefix + "model_runner"]
    cgs = modules[prefix + "model_runner_components.cuda_graph_setup"]
    cfg = modules[prefix + "cuda_graph_config"]
    state = types.SimpleNamespace(breakable=True, calls=[])

    def original(**kwargs):
        state.calls.append(kwargs)
        return "original-result"

    mr.capture_prefill_graph = cgs.capture_prefill_graph = original
    cgs.GraphCapture = lambda **kwargs: kwargs
    cfg.Phase = types.SimpleNamespace(PREFILL="prefill")
    cfg.Backend = types.SimpleNamespace(BREAKABLE="breakable")

    def check(phase, backend):
        assert (phase, backend) == ("prefill", "breakable")
        return state.breakable

    cfg.check_cuda_graph_backend = check
    return mr, cgs, state, original


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("force", [False, True])
def test_breakable_driver_gate_registered_and_idempotent(host, draft, force):
    mr, cgs, state, original = host
    gate.install()
    installed = cgs.capture_prefill_graph
    assert installed is mr.capture_prefill_graph
    assert installed is not original
    record = registry.get("cuda_graph")
    assert record["implementation"] == "prefill-bcg-driver-truth"
    assert record["mechanism"] == registry.MECH_SEAM
    assert record["target"].endswith(".capture_prefill_graph")
    gate.install()
    assert cgs.capture_prefill_graph is installed
    runner, eager = types.SimpleNamespace(is_draft_worker=draft), object()
    result = installed(model_runner=runner, eager_runner=eager,
                       force_for_draft_worker=force)
    assert result == dict(runner=None if draft else eager,
                          memory_phase="draft_prefill" if draft else "prefill",
                          memory_usage_gb=0, capture_time=0)
    assert not state.calls


@pytest.mark.parametrize("force", [False, True])
def test_other_prefill_backends_delegate_unchanged(host, force):
    mr, cgs, state, original = host
    state.breakable = False
    gate.install()
    runner, eager = object(), object()
    assert mr.capture_prefill_graph(
        model_runner=runner, eager_runner=eager,
        force_for_draft_worker=force,
    ) == "original-result"
    assert state.calls == [dict(model_runner=runner, eager_runner=eager,
                                force_for_draft_worker=force)]


def test_non_hip_leaves_host_untouched(host, monkeypatch):
    mr, cgs, state, original = host
    monkeypatch.setattr(torch.version, "hip", None)
    gate.install()
    assert mr.capture_prefill_graph is cgs.capture_prefill_graph is original
    assert registry.get("cuda_graph") is None


def test_registration_failure_does_not_publish_bindings(host, monkeypatch):
    mr, cgs, state, original = host

    def reject(*args, **kwargs):
        raise ValueError("registration refused")

    monkeypatch.setattr(registry, "register", reject)
    with pytest.raises(ValueError, match="registration refused"):
        gate.install()
    assert mr.capture_prefill_graph is cgs.capture_prefill_graph is original
