"""CLI evidence driver import is inert; no official CLI, model or GPU is run.

Run: HCU_ADAPTER_HOST=none python -m pytest -q tests/adapter/test_required_cli_probe_import.py
"""
import importlib.util
from pathlib import Path
import runpy
import sys


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
