"""CLI evidence driver import is inert; no official CLI, model or GPU is run.

Run: HCU_ADAPTER_HOST=none python -m pytest -q tests/adapter/test_required_cli_probe_import.py
"""
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import pytest


def test_required_cli_driver_import_does_not_run_cli_write_or_exit(monkeypatch, tmp_path):
    driver = Path(__file__).parent / "tools/run_required_capability_cli_probe.py"
    output = tmp_path / "result.json"
    events = tmp_path / "events.jsonl"
    monkeypatch.setattr(sys, "argv", [str(driver), str(output)])
    monkeypatch.setenv("HCU_REQUIRED_PROBE_EVENTS", str(events))
    calls = []
    monkeypatch.setattr(runpy, "run_module", lambda *a, **kw: calls.append((a, kw)))
    spec = importlib.util.spec_from_file_location("_required_cli_import_contract", driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert calls == [] and not output.exists() and not events.exists()
    assert sys.argv == [str(driver), str(output)]
    assert callable(module.main)


@pytest.mark.parametrize("outcome,exit_code", [("return", 0), ("exit0", 0), ("exit2", 2), ("error", 1)])
def test_explicit_driver_main_retains_cli_exit_and_evidence(monkeypatch, tmp_path, outcome, exit_code):
    from hcu_adapter.bridges import vllm as bridge

    driver = Path(__file__).parent / "tools/run_required_capability_cli_probe.py"
    output, events = tmp_path / "result.json", tmp_path / "events.jsonl"
    monkeypatch.setattr(sys, "argv", [str(driver), str(output)])
    monkeypatch.setenv("HCU_REQUIRED_PROBE_EVENTS", str(events))
    monkeypatch.setenv("HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS", "owner-a")
    monkeypatch.setenv("HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES", "cap-a")
    monkeypatch.setattr(bridge, "extension_reports", lambda: ())
    calls = []

    def cli(name, *, run_name):
        calls.append((name, run_name, list(sys.argv)))
        events.write_text('{"owner":"observed"}\n')
        if outcome.startswith("exit"):
            raise SystemExit(int(outcome[-1]))
        if outcome == "error":
            raise ValueError("CPU CLI fixture failure")

    monkeypatch.setattr(runpy, "run_module", cli)
    spec = importlib.util.spec_from_file_location("_required_cli_main_contract", driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == exit_code
    assert calls == [("vllm.entrypoints.openai.api_server", "__main__", ["vllm.entrypoints.openai.api_server", "--help"])]
    row = json.loads(output.read_text())
    assert row["exit_code"] == exit_code and row["activation_events"] == [{"owner": "observed"}]
    assert row["required_extensions"] == "owner-a" and row["required_capabilities"] == "cap-a"
    assert (row["error"] is not None) is (outcome == "error")
