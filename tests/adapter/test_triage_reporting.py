# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Triage must retain pytest failures instead of extracting only pass counts."""

from pathlib import Path
import runpy
import subprocess
from unittest.mock import Mock, patch

import pytest


@pytest.mark.parametrize("stdout,stderr,returncode,expected", [
    ("FAILED tests/kernels/test_example.py::test_bad\n"
     "=== 3 failed, 184 passed, 2 skipped in 1.2s ===\n", "", 1,
     "3 failed, 184 passed, 2 skipped in 1.2s"),
    ("38 passed in 1.2s\n", "", 0, "38 passed in 1.2s"),
    ("", "ImportError: missing provider\n", 2, "ImportError: missing provider"),
])
def test_pytest_counts_and_failure_output(stdout, stderr, returncode, expected, capsys):
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/run_triage.py"))
    function = module["_pytest"]
    completed = subprocess.CompletedProcess([], returncode, stdout, stderr)
    with patch.object(function.__globals__["subprocess"], "run", return_value=completed):
        function("ops", "example", ["tests/kernels", "-q"])
    result = function.__globals__["RESULTS"][-1]
    assert result[2:] == ("FAIL" if returncode else "PASS", expected)
    output = capsys.readouterr().out
    if returncode:
        assert stdout in output
        assert stderr in output


@pytest.mark.parametrize("summary,status", [
    ("53 skipped in 4.44s", "SKIP"),
    ("254 passed, 2 skipped, 2 xfailed, 2 warnings in 24.75s", "INCOMPLETE"),
    ("2 passed, 1 skipped, 1 deselected in 51.80s", "INCOMPLETE"),
    ("2 xfailed in 1s", "INCOMPLETE"),
    ("1 xpassed in 1s", "INCOMPLETE"),
    ("38 passed, 5 deselected in 1s", "PASS"),
    ("no summary available", "INCOMPLETE"),
])
def test_skips_and_expected_failures_do_not_count_as_pass(summary, status, capsys):
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/run_triage.py"))
    function = module["_pytest"]
    completed = subprocess.CompletedProcess([], 0, summary + "\n", "")
    with patch.object(function.__globals__["subprocess"], "run", return_value=completed):
        function("ops", "example", ["tests/kernels", "-q"])
    assert function.__globals__["RESULTS"][-1][2] == status
    result = module["verdict"]()
    output = capsys.readouterr().out
    assert result == (0 if status == "PASS" else 2)
    assert ("ALL GREEN" in output) == (status == "PASS")


@pytest.mark.parametrize("records,expected", [
    ([], 2),
    ([("env", "torch", "INFO", "version")], 2),
    ([("ops", "probe", "SKIP", "no device")], 2),
    ([("ops", "probe", "PASS", "")], 0),
    ([("ops", "a", "SKIP", ""), ("ops", "b", "FAIL", "broken")], 1),
])
def test_verdict_requires_actual_passes_and_failure_has_priority(records, expected):
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/run_triage.py"))
    module["RESULTS"].extend(records)
    assert module["verdict"]() == expected


@pytest.mark.parametrize("checkpoint", [None, "/explicit-model"])
def test_ops_rope_uses_original_contract_and_keeps_explicit_checkpoint(checkpoint, monkeypatch):
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/run_triage.py"))
    if checkpoint is None:
        monkeypatch.delenv("HCU_CKPT", raising=False)
    else:
        monkeypatch.setenv("HCU_CKPT", checkpoint)
    function = module["stage_ops"]
    pytest_group, probes = Mock(), Mock()
    with patch.dict(function.__globals__, _pytest=pytest_group, run_probe=probes):
        function(False)
    assert pytest_group.call_args_list[0].args == (
        "ops", "rope-parity", ["tests/kernels/test_original_rope.py", "-q"])
    checkpoint_calls = [call for call in probes.call_args_list
                        if call.args[1] == "rope-checkpoint-parity"]
    assert len(checkpoint_calls) == int(checkpoint is not None)
    if checkpoint_calls:
        assert checkpoint_calls[0].args[2] == "_rope_probe.py"
