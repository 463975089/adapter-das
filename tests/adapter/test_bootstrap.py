"""Multi-host bootstrap selection contracts.

Run: python3 -m pytest -q tests/adapter/test_bootstrap.py
"""

from __future__ import annotations

import pytest

import hcu_adapter.bootstrap as bootstrap
from hcu_adapter.bootstrap import (
    ACTIVE_HOST_ENV,
    detect_host,
    propagate_host,
)


def _detect(
    *,
    argv=(),
    loaded=(),
    available=(),
    public="auto",
    inherited=None,
):
    environment = {"HCU_ADAPTER_HOST": public}
    if inherited is not None:
        environment[ACTIVE_HOST_ENV] = inherited
    return detect_host(
        argv=argv,
        loaded_modules=loaded,
        available=available,
        environment=environment,
    )


@pytest.mark.parametrize("host", ["sglang", "vllm", "deepinfer"])
def test_explicit_host_overrides_inherited_selection(host):
    other = "vllm" if host == "sglang" else "sglang"

    assert _detect(public=host, inherited=other) == host


def test_explicit_none_disables_an_inherited_host():
    assert _detect(public="none", inherited="sglang") is None


def test_invalid_explicit_host_is_rejected():
    with pytest.raises(ValueError, match="HCU_ADAPTER_HOST"):
        _detect(public="invalid-host")


@pytest.mark.parametrize("host", ["sglang", "vllm", "deepinfer"])
def test_inherited_automatic_host_survives_generic_child_argv(host):
    assert _detect(
        argv=("python", "-c", "from multiprocessing.spawn import spawn_main"),
        available=("sglang", "vllm"),
        inherited=host,
    ) == host


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("python", "-m", "sglang.launch_server"), "sglang"),
        (("python", "-m", "vllm.entrypoints.openai.api_server"), "vllm"),
        (("python", "-m", "deepinfer.entrypoints.openai.api_server"), "deepinfer"),
        (("sglang", "serve", "/model"), "sglang"),
        (("/usr/local/bin/vllm", "serve", "/model"), "vllm"),
        (("/usr/local/bin/deepinfer", "serve", "/model"), "deepinfer"),
    ],
)
def test_launch_intent_selects_exactly_one_host(argv, expected):
    assert _detect(argv=argv, available=("sglang", "vllm")) == expected


def test_non_launcher_argument_does_not_override_host_selection():
    assert _detect(
        argv=("python", "serve.py", "--served-model-name", "vllm.demo"),
        available=("sglang", "vllm"),
    ) == "sglang"


@pytest.mark.parametrize("module_name", ["sglang", "sglang.srt.server"])
def test_loaded_sglang_module_selects_sglang(module_name):
    assert _detect(loaded=(module_name,), available=("sglang", "vllm")) == "sglang"


@pytest.mark.parametrize("module_name", ["vllm", "vllm.engine.arg_utils"])
def test_loaded_vllm_module_selects_vllm(module_name):
    assert _detect(loaded=(module_name,), available=("sglang", "vllm")) == "vllm"


@pytest.mark.parametrize("module_name", ["deepinfer", "deepinfer.engine.llm_engine"])
def test_loaded_deepinfer_module_selects_deepinfer(module_name):
    assert _detect(
        loaded=(module_name,), available=("sglang", "vllm", "deepinfer")
    ) == "deepinfer"


@pytest.mark.parametrize("host", ["sglang", "vllm", "deepinfer"])
def test_single_available_host_is_selected(host):
    assert _detect(available=(host,)) == host


def test_both_available_hosts_keep_legacy_sglang_default():
    assert _detect(available=("sglang", "vllm")) == "sglang"


def test_no_available_host_is_a_noop():
    assert _detect(available=()) is None


@pytest.mark.parametrize("host", ["sglang", "vllm", "none"])
def test_selected_host_is_propagated_without_overwriting_public_choice(host):
    environment = {"HCU_ADAPTER_HOST": "auto"}

    propagate_host(host, environment)

    assert environment == {
        "HCU_ADAPTER_HOST": "auto",
        ACTIVE_HOST_ENV: host,
    }


class _Bridge:
    def __init__(self, event, events, failure=None):
        self.event = event
        self.events = events
        self.failure = failure

    def install(self):
        self.events.append(self.event)
        if self.failure is not None:
            raise self.failure


def _arm_install(monkeypatch, *, hip=True, available=("sglang",), common_failure=None):
    events = []
    modules = {
        "hcu_adapter.bridges.common": _Bridge(
            "common", events, failure=common_failure
        ),
        "hcu_adapter.bridges.sglang": _Bridge("sglang", events),
        "hcu_adapter.bridges.vllm": _Bridge("vllm", events),
        "hcu_adapter.bridges.deepinfer": _Bridge("deepinfer", events),
    }
    monkeypatch.setattr(bootstrap, "_is_hip", lambda: hip)
    monkeypatch.setattr(bootstrap, "_available_hosts", lambda: set(available))
    monkeypatch.setattr(bootstrap, "_loaded_module_names", lambda: ())
    monkeypatch.setattr(
        bootstrap.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(bootstrap, "_installed_pid", None)
    monkeypatch.delenv(ACTIVE_HOST_ENV, raising=False)
    monkeypatch.setenv("HCU_ADAPTER_HOST", "auto")
    return events


@pytest.mark.parametrize("host", ["sglang", "vllm"])
def test_install_runs_common_before_the_selected_host(monkeypatch, caplog, host):
    events = _arm_install(monkeypatch, available=(host,))

    bootstrap.install()

    assert events == ["common", host]
    assert f"host={host} source=availability" in caplog.text


def test_deepinfer_installs_without_the_early_common_bridge(monkeypatch, caplog):
    events = _arm_install(monkeypatch, available=("deepinfer",))

    bootstrap.install()

    assert events == ["deepinfer"]
    assert "host=deepinfer source=availability" in caplog.text


def test_install_is_process_idempotent(monkeypatch):
    events = _arm_install(monkeypatch)

    bootstrap.install()
    bootstrap.install()

    assert events == ["common", "sglang"]


def test_common_failure_does_not_block_the_selected_host(monkeypatch):
    events = _arm_install(monkeypatch, common_failure=RuntimeError("guard failed"))

    bootstrap.install()

    assert events == ["common", "sglang"]


def test_install_is_a_noop_on_non_hip(monkeypatch):
    events = _arm_install(monkeypatch, hip=False)

    bootstrap.install()

    assert events == []
    assert ACTIVE_HOST_ENV not in bootstrap.os.environ


def test_install_is_a_noop_when_no_host_is_available(monkeypatch):
    events = _arm_install(monkeypatch, available=())

    bootstrap.install()

    assert events == []
    assert ACTIVE_HOST_ENV not in bootstrap.os.environ
