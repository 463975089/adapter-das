# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Triage must retain pytest failures instead of extracting only pass counts."""

from pathlib import Path
import runpy
import subprocess
from unittest.mock import patch

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
