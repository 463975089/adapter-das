"""vLLM custom-op compatibility bridge used by the sglang host chain.

Run: python3 -m pytest -q tests/adapter/test_vllm_bridge.py
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

from hcu_adapter.bridges import vllm as bridge
from hcu_adapter.extensions import (
    ExtensionReport,
    ExtensionStatus,
    RequiredExtensionError,
)


@pytest.mark.parametrize(
    "version",
    (
        "0.32.0",
        "0.32.0rc1",
        "0.32.0.dev7",
        "0.32.0.post1",
        "0.32.0+rocm723",
        "0.32.1",
        "0.32.1rc1",
        "0.32.99+vendor.7",
        "0.33.0",
        "0.33.0rc1",
        "0.33.0.dev7",
        "0.33.0.post1",
        "0.33.0+dtk2610",
        "0.33.1",
        "0.33.99+vendor.7",
    ),
)
def test_vllm_release_gate_accepts_supported_patch_lines(version):
    assert bridge.is_supported_vllm_version(version) is True


@pytest.mark.parametrize(
    "version",
    (
        None,
        "not-a-version",
        "1!0.32.0",
        "1!0.33.0",
        "0.3.0",
        "0.31.99",
        "0.34.0rc1",
        "0.34.0",
    ),
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


@pytest.mark.parametrize("architecture", ("gfx928", "gfx92a", "gfx936", "gfx938"))
@pytest.mark.parametrize("suffix", ("", ":sramecc+:xnack-"))
def test_hcu_detection_accepts_supported_architectures(architecture, suffix):
    torch_module, _ = _fake_torch(architecture=architecture + suffix)

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


@pytest.mark.parametrize("architecture", ("gfx942", "gfx950", "gfx92a0"))
def test_hcu_detection_rejects_an_unsupported_architecture(architecture):
    torch_module, _ = _fake_torch(architecture=architecture)

    result = bridge.detect_hcu(torch_module)

    assert result.detected is False
    assert result.architecture == architecture
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


def test_process_actions_drop_extension_reports_after_fork_like_change(
    monkeypatch,
):
    pid = 101
    monkeypatch.setattr(bridge.os, "getpid", lambda: pid)
    actions = bridge.ProcessLocalActions()
    report = ExtensionReport(
        "das_hcu",
        "das-hcu",
        ExtensionStatus.ACTIVE,
        frozenset({"rms-norm"}),
        "",
    )
    actions.record_extension_reports((report,))

    assert actions.extension_reports() == (report,)
    pid = 202
    assert actions.extension_reports() == ()


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

    assert first == "hcu_adapter.bridges.vllm.HcuPlatform"
    assert repeated == first
    assert calls == ["version"]


def test_runtime_plugin_runs_guards_then_extensions_once_and_keeps_reports(
    monkeypatch,
):
    calls = []
    report = ExtensionReport(
        "das_hcu",
        "das-hcu",
        ExtensionStatus.ACTIVE,
        frozenset({"rms-norm"}),
        "",
    )
    monkeypatch.setattr(
        bridge,
        "detect_hcu",
        lambda: bridge.HcuDetection(True, "gfx938", ""),
    )
    monkeypatch.setattr(
        bridge,
        "_ensure_vllm_compatible",
        lambda: (calls.append("version"), "0.32.0+rocm723")[1],
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
    monkeypatch.setattr(bridge, "_require_vllm_hip_runtime_abi", lambda: calls.append("runtime-abi"))
    monkeypatch.setattr(
        bridge,
        "_installed_adapter_version",
        lambda: (calls.append("adapter-version"), "0.1.0")[1],
    )

    def activate(context, *, health_check_sink):
        calls.append(
            ("extensions", context.adapter_version, context.vllm_version)
        )
        return (report,)

    monkeypatch.setattr(bridge, "activate_extensions", activate)
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())

    assert bridge.hcu_runtime_plugin() is None
    assert bridge.hcu_runtime_plugin() is None

    assert calls == [
        "version",
        "platform",
        "dtk",
        "runtime-abi",
        "adapter-version",
        ("extensions", "0.1.0", "0.32.0+rocm723"),
    ]
    assert bridge.extension_reports() == (report,)


def test_runtime_plugin_records_required_reports_before_latching(monkeypatch):
    report = ExtensionReport(
        "required",
        "required",
        ExtensionStatus.FAILED,
        frozenset({"attention"}),
        "activation failed: RuntimeError: unavailable",
    )
    calls = []
    monkeypatch.setattr(
        bridge,
        "detect_hcu",
        lambda: bridge.HcuDetection(True, "gfx938", ""),
    )
    monkeypatch.setattr(bridge, "_ensure_vllm_compatible", lambda: "0.32.0")
    monkeypatch.setattr(bridge, "_ensure_hcu_platform_selected", lambda: None)
    monkeypatch.setattr(bridge, "configure_vllm_hip_runtime_path", lambda: None)
    monkeypatch.setattr(bridge, "_require_vllm_hip_runtime_abi", lambda: None)
    monkeypatch.setattr(bridge, "_installed_adapter_version", lambda: "0.1.0")

    def fail(_context, *, health_check_sink):
        calls.append("activate")
        raise RequiredExtensionError("required extension failed", (report,))

    monkeypatch.setattr(bridge, "activate_extensions", fail)
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())

    with pytest.raises(bridge.LatchedVllmRuntimeError, match="required"):
        bridge.hcu_runtime_plugin()
    with pytest.raises(bridge.LatchedVllmRuntimeError, match="required"):
        bridge.hcu_runtime_plugin()

    assert calls == ["activate"]
    assert bridge.extension_reports() == (report,)


def test_adapter_version_comes_from_the_shipping_distribution(monkeypatch):
    calls = []

    def version(name):
        calls.append(name)
        return "0.1.0"

    monkeypatch.setattr(bridge.metadata, "version", version)

    assert bridge._installed_adapter_version() == "0.1.0"
    assert calls == ["sgl-kernel"]


def test_hcu_platform_defaults_to_triton_and_routes_flash_to_plugin(
    monkeypatch,
):
    validation_calls = []

    class HcuFlashAttentionBackend:
        @classmethod
        def validate_configuration(cls, **kwargs):
            validation_calls.append(kwargs)
            return []

    class RocmPlatform:
        @classmethod
        def get_device_capability(cls):
            return "gfx936-capability"

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
    registry.AttentionBackendEnum = SimpleNamespace(
        FLASH_ATTN="FLASH_ATTN",
        TRITON_ATTN="TRITON_ATTN",
    )
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
    monkeypatch.delitem(sys.modules, "hcu_vllm_das", raising=False)
    monkeypatch.delitem(bridge.__dict__, "HcuPlatform", raising=False)
    real_import_module = bridge.import_module
    monkeypatch.setattr(
        bridge,
        "import_module",
        lambda name: (
            SimpleNamespace(HcuFlashAttentionBackend=HcuFlashAttentionBackend)
            if name == "hcu_vllm_das.attention.flash_attn"
            else real_import_module(name)
        ),
    )

    platform = bridge.HcuPlatform
    selector_values = {"dtype": "bf16", "block_size": 64, "use_dcp": False}
    selector = SimpleNamespace(_asdict=lambda: selector_values)

    assert platform.device_name == "hcu"
    assert platform.get_attn_backend_cls(None, selector, 32) == (
        "TRITON_ATTN",
        selector,
        32,
    )
    mla_selector = SimpleNamespace(
        use_mla=True,
        use_sparse=False,
        _asdict=lambda: {"use_mla": True, "use_sparse": False},
    )
    assert platform.get_attn_backend_cls(None, mla_selector, 24) == (
        None,
        mla_selector,
        24,
    )
    sparse_mla_selector = SimpleNamespace(
        use_mla=True,
        use_sparse=True,
        _asdict=lambda: {"use_mla": True, "use_sparse": True},
    )
    assert platform.get_attn_backend_cls(None, sparse_mla_selector, 20) == (
        None,
        sparse_mla_selector,
        20,
    )
    assert platform.get_attn_backend_cls("EXPLICIT", selector, 16) == (
        "EXPLICIT",
        selector,
        16,
    )
    actions = bridge.ProcessLocalActions()
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", actions)
    active = ExtensionReport("flash", "flash", ExtensionStatus.ACTIVE,
                             frozenset({"hcu-flash-attention"}), "")
    actions.record_extension_health(active, lambda: None)
    actions.record_extension_reports((active,))
    assert platform.get_attn_backend_cls("FLASH_ATTN", selector, 8) == (
        "FLASH_ATTN", selector, 8,
    )
    assert validation_calls == []
    assert "hcu_vllm_das" not in sys.modules


def test_hcu_platform_rejects_invalid_flash_plugin_configuration(monkeypatch):
    class HcuFlashAttentionBackend:
        @classmethod
        def validate_configuration(cls, **kwargs):
            return ["DCP not supported", "block_size not supported"]

    class RocmPlatform:
        @classmethod
        def get_device_capability(cls):
            return "gfx936-capability"

        @classmethod
        def get_attn_backend_cls(cls, selected, attn_selector_config, num_heads=None):
            # CPU-double official selector owns validation, not Adapter.
            invalid = HcuFlashAttentionBackend.validate_configuration(
                device_capability=cls.get_device_capability(), **attn_selector_config._asdict())
            raise ValueError(str(invalid))

    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    platforms = ModuleType("vllm.platforms")
    platforms.__path__ = []
    rocm = ModuleType("vllm.platforms.rocm")
    rocm.RocmPlatform = RocmPlatform
    registry = ModuleType("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = SimpleNamespace(
        FLASH_ATTN="FLASH_ATTN",
        TRITON_ATTN="TRITON_ATTN",
    )
    for name, module in {
        "vllm": fake_vllm,
        "vllm.platforms": platforms,
        "vllm.platforms.rocm": rocm,
        "vllm.v1.attention.backends.registry": registry,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(bridge.__dict__, "HcuPlatform", raising=False)
    real_import_module = bridge.import_module
    monkeypatch.setattr(
        bridge,
        "import_module",
        lambda name: (
            SimpleNamespace(HcuFlashAttentionBackend=HcuFlashAttentionBackend)
            if name == "hcu_vllm_das.attention.flash_attn"
            else real_import_module(name)
        ),
    )

    actions = bridge.ProcessLocalActions()
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", actions)
    active = ExtensionReport("flash", "flash", ExtensionStatus.ACTIVE,
                             frozenset({"hcu-flash-attention"}), "")
    actions.record_extension_health(active, lambda: None)
    actions.record_extension_reports((active,))
    with pytest.raises(ValueError, match="DCP not supported"):
        bridge.HcuPlatform.get_attn_backend_cls(
            "FLASH_ATTN",
            SimpleNamespace(_asdict=lambda: {"block_size": 128, "use_dcp": True}),
        )


def test_import_and_install_do_not_eagerly_import_vllm():
    repo_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["HCU_ADAPTER_HOST"] = "none"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from hcu_adapter.bridges import vllm as bridge; "
                "bridge.install(); "
                "assert not any(name == 'vllm' or name.startswith('vllm.') "
                "for name in sys.modules), sorted(name for name in sys.modules "
                "if name == 'vllm' or name.startswith('vllm.'))"
            ),
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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
