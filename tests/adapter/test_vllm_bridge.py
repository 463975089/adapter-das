"""vLLM custom-op compatibility bridge used by the sglang host chain.

Run: python3 -m pytest -q tests/adapter/test_vllm_bridge.py
"""

from __future__ import annotations

import inspect
import importlib
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

from hcu_adapter.bridges import vllm as bridge


@pytest.mark.parametrize(
    "version",
    (
        "0.32.0",
        "0.32.0rc1",
        "0.32.0.dev7",
        "0.32.0.post1",
        "0.32.0+rocm723",
    ),
)
def test_vllm_release_gate_accepts_only_the_032_release_line(version):
    assert bridge.is_supported_vllm_version(version) is True


@pytest.mark.parametrize(
    "version",
    (None, "not-a-version", "1!0.32.0", "0.31.0", "0.32.1", "0.33.0"),
)
def test_vllm_release_gate_rejects_missing_malformed_or_other_releases(version):
    assert bridge.is_supported_vllm_version(version) is False


class _FakeAccelerator:
    def __init__(self, *, available=True, architecture="gfx938"):
        self.available = available
        self.architecture = architecture
        self.property_calls = 0

    def is_available(self):
        return self.available

    def get_device_properties(self, index):
        assert index == 0
        self.property_calls += 1
        return SimpleNamespace(gcnArchName=self.architecture)


def _fake_torch(*, hip="6.2", available=True, architecture="gfx938"):
    accelerator = _FakeAccelerator(
        available=available, architecture=architecture
    )
    return SimpleNamespace(
        version=SimpleNamespace(hip=hip), cuda=accelerator
    ), accelerator


@pytest.mark.parametrize("architecture", ("gfx928", "gfx936", "gfx938"))
def test_hcu_detection_accepts_the_validated_architectures(architecture):
    torch_module, _ = _fake_torch(architecture=architecture)

    result = bridge.detect_hcu(torch_module)

    assert result.detected is True
    assert result.architecture == architecture
    assert result.reason == ""


def test_hcu_detection_rejects_non_hip_without_querying_a_device():
    torch_module, accelerator = _fake_torch(hip=None, architecture="gfx938")

    result = bridge.detect_hcu(torch_module)

    assert result.detected is False
    assert "HIP" in result.reason
    assert accelerator.property_calls == 0


def test_hcu_detection_rejects_an_unvalidated_rocm_architecture():
    torch_module, _ = _fake_torch(architecture="gfx942")

    result = bridge.detect_hcu(torch_module)

    assert result.detected is False
    assert result.architecture == "gfx942"
    assert "unsupported" in result.reason


def test_dtk_runtime_guard_preserves_an_explicit_operator_path(tmp_path):
    explicit = tmp_path / "operator-selected.so"
    environment = {
        "DTKROOT": str(tmp_path / "dtk"),
        "VLLM_CUDART_SO_PATH": str(explicit),
    }

    configured = bridge.configure_vllm_hip_runtime_path(environment)

    assert configured == str(explicit)
    assert environment["VLLM_CUDART_SO_PATH"] == str(explicit)


def test_dtk_runtime_guard_derives_an_existing_library(tmp_path):
    root = tmp_path / "dtk"
    library = root / "lib" / "libamdhip64.so.4"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"hip runtime")
    (root / "lib" / "libamdhip64.so").symlink_to(library.name)
    environment = {"DTKROOT": str(root)}

    configured = bridge.configure_vllm_hip_runtime_path(environment)

    assert configured == str(library)
    assert environment["VLLM_CUDART_SO_PATH"] == str(library)


def test_process_actions_run_once_per_pid_and_reset_after_fork_like_change(
    monkeypatch,
):
    pid = 101
    monkeypatch.setattr(bridge.os, "getpid", lambda: pid)
    actions = bridge.ProcessLocalActions()
    calls = []

    actions.run_once("runtime", lambda: calls.append(pid))
    actions.run_once("runtime", lambda: calls.append(-1))
    pid = 202
    actions.run_once("runtime", lambda: calls.append(pid))

    assert calls == [101, 202]


def test_failed_process_action_is_latched_but_not_inherited(monkeypatch):
    pid = 101
    monkeypatch.setattr(bridge.os, "getpid", lambda: pid)
    actions = bridge.ProcessLocalActions()
    calls = []

    def fail():
        calls.append(pid)
        raise ValueError("incompatible vLLM API")

    with pytest.raises(bridge.LatchedVllmRuntimeError, match="incompatible"):
        actions.run_once("runtime", fail)
    with pytest.raises(bridge.LatchedVllmRuntimeError, match="incompatible"):
        actions.run_once("runtime", lambda: calls.append(-1))

    pid = 202
    actions.run_once("runtime", lambda: calls.append(pid))

    assert calls == [101, 202]


def test_platform_plugin_is_a_non_hcu_noop(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "detect_hcu",
        lambda: bridge.HcuDetection(False, None, "no supported HCU"),
    )
    monkeypatch.setattr(
        bridge,
        "_ensure_vllm_compatible",
        lambda: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    assert bridge.hcu_platform_plugin() is None


def test_platform_plugin_gates_once_and_returns_the_lazy_class_path(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "detect_hcu",
        lambda: bridge.HcuDetection(True, "gfx938", ""),
    )
    monkeypatch.setattr(
        bridge, "_ensure_vllm_compatible", lambda: calls.append("version")
    )
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())

    first = bridge.hcu_platform_plugin()
    repeated = bridge.hcu_platform_plugin()

    assert first == "hcu_adapter.bridges.vllm_platform.HcuPlatform"
    assert repeated == first
    assert calls == ["version"]


def test_runtime_plugin_runs_the_required_vllm_sequence_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "detect_hcu",
        lambda: bridge.HcuDetection(True, "gfx938", ""),
    )
    monkeypatch.setattr(
        bridge, "_ensure_vllm_compatible", lambda: calls.append("version")
    )
    monkeypatch.setattr(
        bridge,
        "_ensure_hcu_platform_selected",
        lambda: calls.append("platform"),
    )
    monkeypatch.setattr(
        bridge,
        "configure_vllm_hip_runtime_path",
        lambda: calls.append("dtk"),
    )
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())

    assert bridge.hcu_runtime_plugin() is None
    assert bridge.hcu_runtime_plugin() is None

    assert calls == ["version", "platform", "dtk"]


def test_hcu_platform_delegates_to_official_rocm_and_defaults_to_triton(
    monkeypatch,
):
    class RocmPlatform:
        @classmethod
        def get_attn_backend_cls(
            cls, selected_backend, attn_selector_config, num_heads=None
        ):
            return selected_backend, attn_selector_config, num_heads

    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    platforms = ModuleType("vllm.platforms")
    platforms.__path__ = []
    rocm = ModuleType("vllm.platforms.rocm")
    rocm.RocmPlatform = RocmPlatform
    v1 = ModuleType("vllm.v1")
    v1.__path__ = []
    attention = ModuleType("vllm.v1.attention")
    attention.__path__ = []
    backends = ModuleType("vllm.v1.attention.backends")
    backends.__path__ = []
    registry = ModuleType("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = SimpleNamespace(TRITON_ATTN="TRITON_ATTN")
    for name, module in {
        "vllm": fake_vllm,
        "vllm.platforms": platforms,
        "vllm.platforms.rocm": rocm,
        "vllm.v1": v1,
        "vllm.v1.attention": attention,
        "vllm.v1.attention.backends": backends,
        "vllm.v1.attention.backends.registry": registry,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(
        sys.modules, "hcu_adapter.bridges.vllm_platform", raising=False
    )

    platform_module = importlib.import_module(
        "hcu_adapter.bridges.vllm_platform"
    )
    platform = platform_module.HcuPlatform
    selector = object()

    assert platform.device_name == "hcu"
    assert platform.get_attn_backend_cls(None, selector, 32) == (
        "TRITON_ATTN",
        selector,
        32,
    )
    assert platform.get_attn_backend_cls("EXPLICIT", selector, 16) == (
        "EXPLICIT",
        selector,
        16,
    )


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
