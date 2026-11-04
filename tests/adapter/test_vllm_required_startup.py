# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

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
    # Required-extension policy has no device/runtime prerequisites. Native ABI
    # and FP8 registration are exercised by their dedicated contract tests.
    monkeypatch.setattr(bridge, "_require_vllm_hip_runtime_abi", lambda: None)
    from hcu_adapter.bridges import vllm_fp8
    monkeypatch.setattr(vllm_fp8, "install", lambda: False)
    # Required-policy CPU fixtures isolate operator activation. The real C4
    # publication, health and startup paths have their own contracts/probes.
    from hcu_adapter.bridges import vllm_sparse_indexer
    monkeypatch.setattr(vllm_sparse_indexer, "install", lambda: False)
    monkeypatch.setattr(vllm_sparse_indexer, "require_healthy", lambda: None)
    # Keep this CPU required-policy fixture independent of optional mHC
    # startup policy and its real publication/cache/health tests.
    from hcu_adapter.bridges import vllm_mhc
    monkeypatch.setattr(vllm_mhc, "install", lambda: False)
    monkeypatch.setattr(vllm_mhc, "require_healthy", lambda: None)
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


def _actual_general_plugin_loader(monkeypatch):
    from importlib.metadata import entry_points
    from vllm import envs, plugins

    assert not envs._is_envs_cache_enabled(), "startup probe requires uncached env"
    monkeypatch.setenv("VLLM_PLUGINS", "hcu_adapter")
    monkeypatch.setattr(plugins, "plugins_loaded", False)
    points = [point for point in entry_points(group="vllm.general_plugins")
              if point.name == "hcu_adapter"]
    assert len(points) == 1
    assert points[0].value == "hcu_adapter.bridges.vllm:hcu_runtime_plugin"
    assert points[0].load() is bridge.hcu_runtime_plugin
    return plugins.load_general_plugins


def _record_startup_evidence(case, mode, caplog):
    import hashlib
    import json
    from pathlib import Path
    import vllm.plugins

    destination = os.environ.get("HCU_REQUIRED_STARTUP_EVIDENCE")
    if not destination:
        return
    root = Path(destination)
    root.mkdir(exist_ok=True, parents=True)
    row = dict(case=case, mode=mode, adapter_origin=bridge.__file__,
               adapter_sha256=hashlib.sha256(Path(bridge.__file__).read_bytes()).hexdigest(),
               official_loader_origin=vllm.plugins.__file__,
               official_loader_sha256=hashlib.sha256(Path(vllm.plugins.__file__).read_bytes()).hexdigest(),
               summaries=[record.getMessage() for record in caplog.records
                          if record.getMessage().startswith("HCU adapter env:")],
               scope="real Adapter metadata/bootstrap/common/official general-plugin/runtime; CPU HCU admission/topology/JIT/extension-provider doubles; no GPU/model")
    (root / f"{case}-{mode}.json").write_text(json.dumps(row, indent=2) + "\n")


@pytest.mark.parametrize("mode", ["default", "explicit", "blank", "malformed-extension", "malformed-capability"])
def test_host_none_actual_bootstrap_and_runtime_resolve_sources_without_common_summary(startup, monkeypatch, caplog, mode):
    import logging
    from hcu_adapter import bootstrap, hcu_env

    points, checks = startup
    contexts, order = [], []
    points.append(Point("provider", "attention", contexts))
    extension, capability = "HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS", "HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES"
    expected = {extension: ("", "default"), capability: ("", "default")}
    if mode == "explicit":
        expected = {extension: ("provider", "explicit"), capability: ("attention", "explicit")}
    elif mode == "blank":
        expected = {extension: ("  ", "explicit"), capability: ("\t", "explicit")}
    elif mode == "malformed-extension":
        expected[extension] = ("provider,", "explicit")
    elif mode == "malformed-capability":
        expected[capability] = (",", "explicit")
    for variable, (value, source) in expected.items():
        if source == "explicit":
            monkeypatch.setenv(variable, value)
    monkeypatch.setenv("HCU_ADAPTER_HOST", "none")
    monkeypatch.setattr(bootstrap, "_installed_pid", None)
    monkeypatch.setattr(bootstrap, "_is_hip", lambda: True)  # CPU admission only.
    load_plugins = _actual_general_plugin_loader(monkeypatch)
    resolver = hcu_env.resolve
    def resolve(name):
        result = resolver(name)
        order.append(("resolve", name, *result))
        return result
    monkeypatch.setattr(hcu_env, "resolve", resolve)
    monkeypatch.setattr(bridge, "_ensure_vllm_compatible", lambda: (order.append(("version",)), "0.32.0+cpu")[1])
    monkeypatch.setattr(bridge, "configure_vllm_hip_runtime_path", lambda: order.append(("registration-boundary",)))
    with caplog.at_level(logging.WARNING, logger="hcu.adapter"):
        bootstrap.install()
        bootstrap.install()
        assert order == []
        for attempt in range(2):
            invoke = load_plugins if attempt == 0 else bridge.hcu_runtime_plugin
            if mode.startswith("malformed"):
                with pytest.raises(bridge.LatchedVllmRuntimeError, match="comma-separated"):
                    invoke()
            else:
                invoke()
        load_plugins()  # Official once flag is separate from host retry health.
    reached = [extension] if mode == "malformed-extension" else [extension, capability]
    assert order[:len(reached)] == [("resolve", key, *expected[key]) for key in reached]
    if mode.startswith("malformed"):
        assert len(order) == len(reached) and contexts == []
    else:
        assert order[len(reached):] == [("version",), ("registration-boundary",)]
        assert len(contexts) == 1
        assert contexts[0].requested_extensions == (frozenset({"provider"}) if mode == "explicit" else frozenset())
        assert contexts[0].requested_capabilities == (frozenset({"attention"}) if mode == "explicit" else frozenset())
    assert not any(r.getMessage().startswith("HCU adapter env:") for r in caplog.records)
    _record_startup_evidence("host-none", mode, caplog)


@pytest.mark.parametrize("mode", ["default", "explicit", "blank", "malformed"])
def test_real_vllm_bootstrap_common_logs_once_then_actual_runtime_preserves_policy(startup, monkeypatch, caplog, mode):
    import logging
    from hcu_adapter import bootstrap, hcu_env, hcu_config, nccl_topo_guard, tvmffi_warmup

    points, checks = startup
    contexts, order = [], []
    points.append(Point("provider", "attention", contexts))
    names = ("HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS", "HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES")
    # Only registered requirements participate; unrelated auto rules and the
    # topology/JIT providers must not touch hardware in this CPU contract.
    # The default all-empty line below concerns this controlled subset only;
    # other real registry entries (for example HOST) may still be logged.
    monkeypatch.setattr(hcu_env, "REGISTRY", {key: hcu_env.REGISTRY[key] for key in names})
    monkeypatch.setattr(hcu_config, "lookup", lambda name: None)
    monkeypatch.setattr(hcu_config, "summary_line", lambda: "")
    monkeypatch.setattr(nccl_topo_guard, "install", lambda: order.append("topology-double"))
    monkeypatch.setattr(tvmffi_warmup, "install", lambda: order.append("warmup-double"))
    monkeypatch.delenv("HCU_ADAPTER_DUMP", raising=False)
    monkeypatch.setenv("HCU_ADAPTER_HOST", "vllm")
    monkeypatch.setattr(bootstrap, "_installed_pid", None)
    monkeypatch.setattr(bootstrap, "_is_hip", lambda: True)
    monkeypatch.setattr(bootstrap, "_available_hosts", lambda: {"vllm"})
    monkeypatch.setattr(bootstrap, "_loaded_module_names", lambda: ())
    monkeypatch.setenv(bootstrap.ACTIVE_HOST_ENV, "none")
    if mode != "default":
        monkeypatch.setenv(names[0], "provider" if mode != "blank" else "  ")
        monkeypatch.setenv(names[1], "attention," if mode == "malformed" else "attention" if mode == "explicit" else "\t")
    load_plugins = _actual_general_plugin_loader(monkeypatch)
    class Order(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith("HCU adapter env:"):
                order.append("actual-summary")
    logger = logging.getLogger("hcu.adapter")
    observer = Order()
    logger.addHandler(observer)
    try:
        with caplog.at_level(logging.WARNING, logger="hcu.adapter"):
            bootstrap.install()
            bootstrap.install()
            for attempt in range(2):
                invoke = load_plugins if attempt == 0 else bridge.hcu_runtime_plugin
                if mode == "malformed":
                    with pytest.raises(bridge.LatchedVllmRuntimeError, match="comma-separated"):
                        invoke()
                else:
                    invoke()
            load_plugins()
    finally:
        logger.removeHandler(observer)
    summaries = [r.getMessage() for r in caplog.records if r.getMessage().startswith("HCU adapter env:")]
    assert len(summaries) == 1
    assert order == ["actual-summary", "topology-double", "warmup-double"]
    if mode == "default":
        assert summaries == ["HCU adapter env: all defaults (no explicit HCU_* vars)"]
    else:
        for name in names:
            assert name + "=" in summaries[0]
        assert summaries[0].count("(explicit)") == 2
    assert "host=vllm source=explicit" in caplog.text
    assert checks == ([] if mode == "malformed" else ["version"])
    assert len(contexts) == (0 if mode == "malformed" else 1)
    _record_startup_evidence("host-vllm", mode, caplog)
