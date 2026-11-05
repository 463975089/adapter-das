# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Admission ordering and real official loader failures; no GPU operations."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from hcu_adapter.bridges import vllm as bridge


@pytest.fixture
def startup(monkeypatch):
    calls = []
    failure = []

    class OfficialWrapper:
        def __init__(self):
            calls.append("official-runtime-constructor")
            if failure:
                raise failure[0]

    original = bridge.import_module

    def load(name):
        if name == "vllm.distributed.device_communicators.cuda_wrapper":
            return SimpleNamespace(CudaRTLibrary=OfficialWrapper)
        return original(name)

    monkeypatch.setattr(bridge, "import_module", load)
    monkeypatch.setattr(bridge, "detect_hcu", lambda: bridge.HcuDetection(True, "gfx936", ""))
    monkeypatch.setattr(bridge, "_ensure_vllm_compatible", lambda: "0.32.1+compatible")
    monkeypatch.setattr(bridge, "_ensure_hcu_platform_selected", lambda: None)
    monkeypatch.setattr(bridge, "configure_vllm_hip_runtime_path", lambda: calls.append("configure"))
    from hcu_adapter.bridges import vllm_allreduce, vllm_fp8
    monkeypatch.setattr(vllm_fp8, "install", lambda: calls.append("fp8"))
    monkeypatch.setattr(vllm_allreduce, "prepare_runtime", lambda: calls.append("ar-prepare"))
    monkeypatch.setattr(bridge, "_installed_adapter_version", lambda: "test")
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())
    # Native admission remains real or explicitly injected below; optional FP8
    # and other built-in operator registrations are outside this loader contract.
    from hcu_adapter.bridges import vllm_mhc, vllm_sparse_indexer
    monkeypatch.setattr(vllm_mhc, "install", lambda: False)
    monkeypatch.setattr(vllm_sparse_indexer, "install", lambda: False)

    def activate(context, *, health_check_sink):
        assert callable(health_check_sink)
        calls.append("extensions")
        return ()

    monkeypatch.setattr(bridge, "activate_extensions", activate)
    monkeypatch.delenv("HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS", raising=False)
    monkeypatch.delenv("HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES", raising=False)
    return calls, failure


def test_runtime_admission_precedes_registrations_and_runs_once(startup):
    calls, _ = startup
    bridge.hcu_runtime_plugin()
    bridge.hcu_runtime_plugin()
    assert calls == ["configure", "official-runtime-constructor", "fp8", "ar-prepare", "extensions"]


@pytest.mark.parametrize("error", [
    OSError("invalid ELF"), AttributeError("missing hipIpcOpenMemHandle"),
    ImportError("official wrapper unavailable"), KeyError("changed runtime mapping"),
])
def test_bad_runtime_latches_before_any_registration(startup, error):
    calls, failure = startup
    failure.append(error)
    for _ in range(2):
        with pytest.raises(bridge.LatchedVllmRuntimeError, match="HIP runtime ABI"):
            bridge.hcu_runtime_plugin()
    assert calls == ["configure", "official-runtime-constructor"]
    assert bridge.extension_reports() == ()


def test_non_hcu_does_not_enter_runtime_admission(startup, monkeypatch):
    calls, _ = startup
    monkeypatch.setattr(bridge, "detect_hcu", lambda: bridge.HcuDetection(False, None, "CPU control"))
    bridge.hcu_runtime_plugin()
    assert calls == []


def test_admission_preserves_explicit_operator_alias(startup, monkeypatch):
    calls, _ = startup
    monkeypatch.setenv("VLLM_CUDART_SO_PATH", "/operator/runtime-alias.so")
    bridge.hcu_runtime_plugin()
    import os
    assert os.environ["VLLM_CUDART_SO_PATH"] == "/operator/runtime-alias.so"
    assert "official-runtime-constructor" in calls


@pytest.mark.parametrize("kind", ["missing", "invalid-elf", "wrong-library"])
def test_real_official_constructor_rejects_invalid_native_library(startup, monkeypatch, tmp_path, kind):
    """Actual ctypes/target constructor; CPU platform/lookup metadata doubles."""
    calls, _ = startup
    official = importlib.import_module("vllm.distributed.device_communicators.cuda_wrapper")
    path = tmp_path / "invalid-runtime.so"
    if kind == "invalid-elf":
        path.write_bytes(b"not a shared library")
    elif kind == "wrong-library":
        # A real loadable system library has no HIP runtime exports.
        candidates = [Path(line.split()[-1]) for line in Path("/proc/self/maps").read_text().splitlines()
                      if line.split()[-1].endswith("/libc.so.6")]
        assert candidates, "Linux CPU control needs an actual mapped libc"
        path = candidates[0]
    monkeypatch.setattr(official, "current_platform", SimpleNamespace(is_rocm=lambda: True))
    monkeypatch.setattr(official, "find_loaded_library", lambda name: None)
    monkeypatch.setenv("VLLM_CUDART_SO_PATH", str(path))
    original = bridge.import_module
    monkeypatch.setattr(bridge, "import_module", lambda name: official if name == official.__name__ else original(name))
    with pytest.raises(bridge.LatchedVllmRuntimeError, match="HIP runtime ABI"):
        bridge.hcu_runtime_plugin()
    assert calls == ["configure"]
    assert bridge.extension_reports() == ()


def test_official_loaded_library_precedence_is_preserved(startup, monkeypatch, tmp_path):
    """Real target resolver; explicit CPU library/lookup doubles, no HIP call."""
    calls, _ = startup
    official = importlib.import_module("vllm.distributed.device_communicators.cuda_wrapper")
    selected = str(tmp_path / "already-loaded-runtime.so")
    libraries = []

    class Library:
        def __getattr__(self, name):
            def prohibited(*args):
                raise AssertionError("admission must not execute " + name)
            return prohibited

    def load_library(path):
        libraries.append(path)
        return Library()

    monkeypatch.setattr(official, "current_platform", SimpleNamespace(is_rocm=lambda: True))
    monkeypatch.setattr(official, "find_loaded_library", lambda name: selected)
    monkeypatch.setenv("VLLM_CUDART_SO_PATH", "/missing/operator-fallback.so")
    monkeypatch.setattr(official.ctypes, "CDLL", load_library)
    original = bridge.import_module
    monkeypatch.setattr(bridge, "import_module", lambda name: official if name == official.__name__ else original(name))
    bridge.hcu_runtime_plugin()
    assert libraries == [selected]
    assert calls == ["configure", "fp8", "ar-prepare", "extensions"]


def test_ar_prepare_failure_stops_extensions_and_runtime_retry(startup, monkeypatch):
    """Public runtime refusal persists; no Plugin/AR math is executed."""
    from hcu_adapter.bridges import vllm_allreduce
    calls, _ = startup

    def refuse():
        calls.append("ar-prepare")
        raise vllm_allreduce.AllReduceCompatibilityError("CPU AR preparation refusal")

    monkeypatch.setattr(vllm_allreduce, "prepare_runtime", refuse)
    for query in (bridge.hcu_runtime_plugin, bridge.require_runtime_healthy, bridge.hcu_runtime_plugin):
        with pytest.raises(bridge.LatchedVllmRuntimeError, match="AR preparation refusal"):
            query()
    assert calls == ["configure", "official-runtime-constructor", "fp8", "ar-prepare"]
    assert bridge.extension_reports() == ()
