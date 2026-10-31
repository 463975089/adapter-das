"""Exercise the real runtime callback and extension host with injected discovery."""
from dataclasses import dataclass
import os

import pytest

from hcu_adapter.bridges import vllm as bridge
from hcu_adapter import extensions


@dataclass
class Point:
    name: str
    capability: str
    contexts: list

    def load(self):
        return lambda: extensions.ExtensionSpec(
            "declared-" + self.name, 1, 0, {self.capability}, self.contexts.append,
        )


@pytest.fixture
def startup(monkeypatch):
    monkeypatch.delenv("HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS", raising=False)
    monkeypatch.delenv("HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES", raising=False)
    monkeypatch.setattr(bridge, "detect_hcu", lambda: bridge.HcuDetection(True, "gfx936", ""))
    monkeypatch.setattr(bridge, "_ensure_hcu_platform_selected", lambda: None)
    monkeypatch.setattr(bridge, "configure_vllm_hip_runtime_path", lambda: None)
    monkeypatch.setattr(bridge, "_installed_adapter_version", lambda: "test")
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())
    checks = []
    monkeypatch.setattr(bridge, "_ensure_vllm_compatible",
                        lambda: (checks.append("version"), "0.32.0+test")[1])
    points = []
    monkeypatch.setattr(extensions, "_entry_points", lambda: points)
    return points, checks


def test_environment_requirements_reach_activation_and_run_once(startup, monkeypatch):
    points, checks = startup
    contexts = []
    points.append(Point("provider", "attention", contexts))
    monkeypatch.setenv("HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS", " provider,provider ")
    monkeypatch.setenv("HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES", " attention ")
    bridge.hcu_runtime_plugin()
    bridge.hcu_runtime_plugin()
    assert checks == ["version"]
    assert len(contexts) == 1
    assert contexts[0].requested_extensions == frozenset({"provider"})
    assert contexts[0].requested_capabilities == frozenset({"attention"})
    assert bridge.extension_reports()[0].status is extensions.ExtensionStatus.ACTIVE


@pytest.mark.parametrize("variable,name", [
    ("HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS", "missing"),
    ("HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS", "declared-provider"),
    ("HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES", "missing"),
])
def test_missing_required_entrypoint_or_capability_latches_with_reports(startup, monkeypatch, variable, name):
    points, checks = startup
    contexts = []
    points.append(Point("provider", "attention", contexts))
    monkeypatch.setenv(variable, name)
    for _ in range(2):
        with pytest.raises(bridge.LatchedVllmRuntimeError, match=name):
            bridge.hcu_runtime_plugin()
    assert len(contexts) == 1
    assert checks == ["version"]
    assert len(bridge.extension_reports()) == 1
    assert bridge.extension_reports()[0].entry_point_name == "provider"


def test_conflicting_required_capability_latches_without_either_activation(startup, monkeypatch):
    points, _ = startup
    contexts = []
    points.extend([Point("a", "attention", contexts), Point("b", "attention", contexts)])
    monkeypatch.setenv("HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES", "attention")
    with pytest.raises(bridge.LatchedVllmRuntimeError, match="attention"):
        bridge.hcu_runtime_plugin()
    assert contexts == []
    assert len(bridge.extension_reports()) == 2
    assert all(r.status is extensions.ExtensionStatus.FAILED for r in bridge.extension_reports())


@pytest.mark.parametrize("raw", (",", "attention,", "a,,b", "two words"))
def test_malformed_requirements_fail_before_runtime_checks(startup, monkeypatch, raw):
    points, checks = startup
    monkeypatch.setenv("HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES", raw)
    with pytest.raises(bridge.LatchedVllmRuntimeError, match="comma-separated"):
        bridge.hcu_runtime_plugin()
    assert checks == []
    assert bridge.extension_reports() == ()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires process fork")
def test_forked_runtime_rechecks_inherited_required_capability(startup, monkeypatch):
    points, _ = startup
    points.append(Point("provider", "attention", []))
    monkeypatch.setenv("HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES", "attention")
    bridge.hcu_runtime_plugin()
    pid = os.fork()
    if pid == 0:
        points.clear()
        try:
            bridge.hcu_runtime_plugin()
        except bridge.LatchedVllmRuntimeError as error:
            os._exit(0 if "attention" in str(error) and not bridge.extension_reports() else 2)
        except BaseException:
            os._exit(3)
        os._exit(4)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert bridge.extension_reports()[0].status is extensions.ExtensionStatus.ACTIVE
