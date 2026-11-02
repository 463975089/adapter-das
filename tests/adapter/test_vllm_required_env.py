"""CPU required-startup env governance; no model, device or provider execution.

Run with HCU_ADAPTER_HOST=none and empty GPU visibility:
python -m pytest -q tests/adapter/test_vllm_required_env.py
"""
import logging
from pathlib import Path

import pytest

from hcu_adapter import hcu_env, hcu_config, registry
from hcu_adapter.bridges import vllm as bridge

NAMES = ("HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS", "HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES")


@pytest.mark.parametrize("name", NAMES)
def test_required_startup_variable_is_registered_documented_and_explicit_only(name):
    spec = hcu_env.REGISTRY[name]
    assert spec["env"] == name and spec["default"] == ""
    assert spec["about"] and "auto" not in spec
    manual = Path(__file__).resolve().parents[2] / "docs/env-vars.md"
    assert f"`{name}`" in manual.read_text()


@pytest.mark.parametrize("name", NAMES)
def test_requirement_parser_consumes_the_canonical_resolver(monkeypatch, name):
    calls = []
    monkeypatch.delenv(name, raising=False)
    def resolve(variable):
        calls.append(variable)
        return " provider-a, provider-b, provider-a ", "explicit"
    monkeypatch.setattr(hcu_env, "resolve", resolve)
    assert bridge._required_names(name) == frozenset({"provider-a", "provider-b"})
    assert calls == [name]


@pytest.mark.parametrize("name", NAMES)
def test_registered_requirements_report_explicit_source_without_auto_probe(monkeypatch, caplog, name):
    subset = {key: hcu_env.REGISTRY[key] for key in NAMES}
    monkeypatch.setattr(hcu_env, "REGISTRY", subset)
    monkeypatch.setattr(hcu_config, "lookup", lambda key: None)
    monkeypatch.setattr(hcu_config, "summary_line", lambda: "")
    monkeypatch.setattr(registry, "log_summary", lambda logger: None)
    monkeypatch.delenv("HCU_ADAPTER_DUMP", raising=False)
    for key in NAMES:
        monkeypatch.delenv(key, raising=False)
    def forbidden():
        raise AssertionError("startup requirements must not probe hardware")
    monkeypatch.setattr(hcu_env, "arch_name", forbidden)
    assert hcu_env.resolve(name) == ("", "default")
    monkeypatch.setenv(name, "owner-a")
    assert hcu_env.resolve(name) == ("owner-a", "explicit")
    with caplog.at_level(logging.WARNING, logger="hcu.adapter"):
        hcu_env.log_summary()
    assert f"{name}=owner-a(explicit)" in caplog.text
    assert bridge._required_names(name) == frozenset({"owner-a"})
