# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Installed test provider exercising actual metadata, general plugins and spawn.

Only HCU admission/platform selection are CPU doubles. Discovery, factories,
host activation, failure propagation and config unpickle are real.
This PR17 fixture includes no PR20 environment-required policy.
"""

import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec


class UnrenderableError(RuntimeError):
    def __str__(self):
        raise KeyboardInterrupt("probe diagnostic formatting failed")


class UnrenderableCommitError(ExtensionCommitError):
    def __str__(self):
        raise ValueError("probe fatal diagnostic formatting failed")


def emit(event):
    import vllm.plugins

    with Path(os.environ["HCU_COMMIT_PROBE_LOG"]).open("a") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "event": event,
                                 "plugins_loaded": vllm.plugins.plugins_loaded}) + "\n")


def _cpu_guard():
    import torch

    assert not torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU discovery/unpickle probe queried/initialized GPU")

    for name in ("init", "_lazy_init", "current_device", "get_device_properties"):
        setattr(torch.cuda, name, forbidden)


def early_spec():
    return ExtensionSpec("early", 1, 30, {"probe-early"}, lambda _: emit("early"))


def owner_spec():
    if os.environ["HCU_COMMIT_PROBE_MODE"] == "factory-badstr":
        emit("factory-published")
        raise UnrenderableCommitError("probe factory publication failure")

    def activate(context):
        emit("prepare")
        mode = os.environ["HCU_COMMIT_PROBE_MODE"]
        if mode == "clean":
            raise ImportError("clean prepare failure; no provider state published")
        if mode == "clean-badstr":
            raise UnrenderableError()
        emit("published")
        try:
            error_type = UnrenderableCommitError if mode == "fatal-badstr" else ExtensionCommitError
            raise error_type("probe late commit failure")
        except ExtensionCommitError as error:
            if mode == "wrapped-badstr":
                raise UnrenderableError() from error
            if mode == "wrapped":
                raise ValueError("provider wrapper") from error
            raise

    return ExtensionSpec("owner", 1, 20, {"probe-owner"}, activate)


def later_spec():
    return ExtensionSpec("later", 1, 10, {"probe-later"}, lambda _: emit("later"))


def exercise(config=None):
    import torch
    from importlib.metadata import entry_points
    from hcu_adapter.bridges import vllm as bridge
    import hcu_adapter.extensions as public
    import vllm.plugins

    _cpu_guard()
    if config is not None:
        assert config.quant_config.get_name() == "cpu_commit_probe"
    # Refuse a dirty test host instead of discovering unrelated live providers.
    assert {p.name for p in entry_points(group="hcu_adapter.extensions")} == {
        "commit_probe_early", "commit_probe_owner", "commit_probe_later",
    }
    adapter_points = [p for p in entry_points(group="vllm.general_plugins")
                      if p.name == "hcu_adapter"]
    assert len(adapter_points) == 1
    assert adapter_points[0].value == "hcu_adapter.bridges.vllm:hcu_runtime_plugin"
    # Explicit CPU test doubles; no vendor/platform/GPU probe is executed.
    bridge.detect_hcu = lambda: bridge.HcuDetection(True, None, "CPU admission double")
    bridge._ensure_hcu_platform_selected = lambda: None
    mode = os.environ["HCU_COMMIT_PROBE_MODE"]
    if mode == "runtime-badstr":
        def fail_platform():
            emit("runtime-prepare")
            raise UnrenderableError()
        bridge._ensure_hcu_platform_selected = fail_platform
    emit("before-discovery")
    failure = None
    try:
        vllm.plugins.load_general_plugins()
    except bridge.LatchedVllmRuntimeError as error:
        failure = error
    reports = bridge.extension_reports()
    clean = mode in ("clean", "clean-badstr")
    expected = not clean
    assert (failure is not None) == expected
    expected_reports = {
        "commit_probe_early": "active", "commit_probe_owner": "failed",
    }
    if clean:
        expected_reports["commit_probe_later"] = "active"
    elif mode == "runtime-badstr":
        expected_reports = {}
    elif mode == "factory-badstr":
        expected_reports = {"commit_probe_owner": "failed"}
    assert {r.entry_point_name: r.status.value for r in reports} == expected_reports
    if mode.endswith("badstr") and mode != "runtime-badstr":
        assert "details unavailable" in next(r.reason for r in reports if r.entry_point_name == "commit_probe_owner")
    fatal = expected and mode != "runtime-badstr"
    if failure is not None:
        assert public.is_fatal_extension_error(failure) == fatal
        for operation in (bridge.require_runtime_healthy, bridge.hcu_runtime_plugin):
            try:
                operation()
            except bridge.LatchedVllmRuntimeError:
                pass
            else:
                raise AssertionError("caught boot failure became healthy")
        # Official loader's once flag is not a runtime-health certificate.
        vllm.plugins.load_general_plugins()
        try:
            bridge.require_runtime_healthy()
        except bridge.LatchedVllmRuntimeError:
            pass
        else:
            raise AssertionError("official retry erased Adapter failure")
    else:
        bridge.require_runtime_healthy()
        bridge.hcu_runtime_plugin()
    assert bridge.extension_reports() == reports
    assert not torch.cuda.is_initialized()
    rows = [json.loads(x) for x in Path(os.environ["HCU_COMMIT_PROBE_LOG"]).read_text().splitlines()]
    rows = [x for x in rows if x["pid"] == os.getpid()]
    events = [x["event"] for x in rows]
    expected_events = ["before-discovery", "early", "prepare"] + (
        ["later"] if clean else ["published"])
    if mode == "runtime-badstr":
        expected_events = ["before-discovery", "runtime-prepare"]
    elif mode == "factory-badstr":
        expected_events = ["before-discovery", "factory-published"]
    assert events == (["payload-unpickle"] if config is not None else []) + expected_events
    if config is not None:
        assert rows[0]["plugins_loaded"] is False  # Actual state restoration before discovery.
    return {"pid": os.getpid(), "events": events, "failed": expected,
            "fatal": fatal, "config_roundtrip": config is not None,
            "adapter_origin": bridge.__file__,
            "adapter_sha256": hashlib.sha256(Path(bridge.__file__).read_bytes()).hexdigest(),
            "gpu_initialized": torch.cuda.is_initialized(),
            "scope": "real metadata/general-plugin/spawn; CPU admission doubles; no engine/model"}


def spawn_child(config, writer):
    writer.send(exercise(config))
    writer.close()


def run(spawn):
    _cpu_guard()
    if not spawn:
        result = exercise()
    else:
        from vllm.config import DeviceConfig, VllmConfig
        from hcu_commit_fatal_payload import ProbeQuantizationConfig

        config = VllmConfig(device_config=DeviceConfig("cpu"))
        config.quant_config = ProbeQuantizationConfig()
        context = mp.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        child = context.Process(target=spawn_child, args=(config, writer))
        child.start()
        writer.close()
        try:
            assert reader.poll(90), "CPU spawn child failed to report"
            result = reader.recv()
            child.join(10)
            assert child.exitcode == 0
            result.update(start_method=context.get_start_method(), child_exit=child.exitcode)
        finally:
            if child.is_alive():
                child.terminate()
                child.join(10)
            reader.close()
    print("COMMIT_PROBE=" + json.dumps(result), flush=True)
