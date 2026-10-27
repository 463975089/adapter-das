"""Multi-host bootstrap selection contracts.

Run: python3 -m pytest -q tests/adapter/test_bootstrap.py
"""

from __future__ import annotations

import pytest

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


@pytest.mark.parametrize("host", ["sglang", "vllm"])
def test_explicit_host_overrides_inherited_selection(host):
    other = "vllm" if host == "sglang" else "sglang"

    assert _detect(public=host, inherited=other) == host


def test_explicit_none_disables_an_inherited_host():
    assert _detect(public="none", inherited="sglang") is None


def test_invalid_explicit_host_is_rejected():
    with pytest.raises(ValueError, match="HCU_ADAPTER_HOST"):
        _detect(public="deepinfer")


@pytest.mark.parametrize("host", ["sglang", "vllm"])
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
        (("sglang", "serve", "/model"), "sglang"),
        (("/usr/local/bin/vllm", "serve", "/model"), "vllm"),
    ],
)
def test_launch_intent_selects_exactly_one_host(argv, expected):
    assert _detect(argv=argv, available=("sglang", "vllm")) == expected


@pytest.mark.parametrize("module_name", ["sglang", "sglang.srt.server"])
def test_loaded_sglang_module_selects_sglang(module_name):
    assert _detect(loaded=(module_name,), available=("sglang", "vllm")) == "sglang"


@pytest.mark.parametrize("module_name", ["vllm", "vllm.engine.arg_utils"])
def test_loaded_vllm_module_selects_vllm(module_name):
    assert _detect(loaded=(module_name,), available=("sglang", "vllm")) == "vllm"


@pytest.mark.parametrize("host", ["sglang", "vllm"])
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
