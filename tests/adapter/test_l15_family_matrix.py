"""CPU-only validation contracts for tools/l15_family_matrix.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "l15_family_matrix", ROOT / "tools" / "l15_family_matrix.py"
)
assert SPEC and SPEC.loader
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class L15FamilyMatrixTests(unittest.TestCase):
    def test_repository_matrix_is_valid_and_complete(self):
        document = json.loads(
            (ROOT / "tests" / "manifests" / "l15_family_matrix_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(matrix.validate(document), [])
        self.assertEqual(
            {family["id"] for family in document["families"]}, matrix.FAMILIES
        )
        self.assertGreaterEqual(
            sum(len(family["cases"]) for family in document["families"]), 20
        )

    def test_rejects_arbitrary_command_and_path_escape(self):
        document = json.loads(
            (ROOT / "tests" / "manifests" / "l15_family_matrix_v1.json").read_text(
                encoding="utf-8"
            )
        )
        bad = json.loads(json.dumps(document))
        bad["families"][0]["cases"][0] = {
            "id": "unsafe",
            "kind": "shell",
            "path": "../../escape",
        }
        errors = matrix.validate(bad)
        self.assertTrue(errors)
        self.assertIn("unsupported", errors[0])

    def test_requires_exact_upstream_nodeid(self):
        document = json.loads(
            (ROOT / "tests" / "manifests" / "l15_family_matrix_v1.json").read_text(
                encoding="utf-8"
            )
        )
        bad = json.loads(json.dumps(document))
        bad["families"][0]["cases"][0]["nodeid"] = (
            "test/registered/attention/unittests/dense/test_triton.py"
        )
        errors = matrix.validate(bad)
        self.assertTrue(errors)
        self.assertIn("exact nodeid", errors[0])


if __name__ == "__main__":
    unittest.main()


def repository_matrix():
    return json.loads(
        (ROOT / "tests/manifests/l15_family_matrix_v1.json").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("value", [None, "", 123])
def test_requires_manifest_id(value):
    document = repository_matrix()
    document["manifest_id"] = value
    assert "manifest_id" in matrix.validate(document)[0]


@pytest.mark.parametrize("timeout", [None, True, False, 0, -1, 1.5, "10"])
def test_requires_positive_integer_timeout(timeout):
    document = repository_matrix()
    document["defaults"]["timeout"] = timeout
    assert "defaults.timeout" in matrix.validate(document)[0]


@pytest.mark.parametrize("target", ["family", "case", "nodeid", "duplicate-family"])
def test_rejects_malformed_nested_records(target):
    document = repository_matrix()
    if target == "family":
        document["families"].append("not an object")
    elif target == "case":
        document["families"][0]["cases"].append("not an object")
    elif target == "nodeid":
        document["families"][0]["cases"][0]["nodeid"] = 17
    else:
        document["families"].append(document["families"][0])
    assert matrix.validate(document)


def test_device_count_rejects_boolean():
    document = repository_matrix()
    document["defaults"]["device_count"] = True
    assert "device_count" in matrix.validate(document)[0]


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_artifact_failure_cleans_temporary_and_preserves_previous_file(tmp_path, failure):
    document = repository_matrix()
    document["families"][0]["id"] = "dense"
    document["families"][0]["cases"] = [
        {"id": "fixture", "kind": "adapter-pytest", "nodeid": "tests/worker.py::test_ok"}
    ]
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/worker.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "tests/manifests").mkdir()
    (tmp_path / document["upstream_manifest"]).write_text("{}")
    output = tmp_path / "result.json"
    output.write_text("previous artifact\n", encoding="utf-8")
    result = matrix.subprocess.CompletedProcess([], 0, "", "")
    target, method = (matrix.json, "dump") if failure == "write" else (matrix.os, "replace")
    with patch.object(matrix, "preflight", return_value={"upstream": {"commit": "fixture"}}):
        with patch.object(matrix, "audit_source", return_value={"passed": True}), \
                patch.object(matrix.subprocess, "run", return_value=result):
            with patch.object(target, method, side_effect=OSError("disk failure")):
                with pytest.raises(OSError, match="disk failure"):
                    matrix.run(document, tmp_path, tmp_path, {"dense"}, output)
    assert output.read_text(encoding="utf-8") == "previous artifact\n"
    assert list(tmp_path.glob(".result.json.*")) == []


@pytest.mark.parametrize("outcome", ["skipped", "xfail", "xpass", "error", "failed"])
def test_strict_gate_rejects_nonpassing_nodes_even_with_zero_exit(outcome):
    from tools.l15_runtime import check_outcomes

    report = {"artifact": "l15-pytest-outcomes", "exitstatus": 0,
              "tests": [{"nodeid": "tests/test_x.py::test_x", "outcome": outcome,
                         "phases": {"call": {"outcome": "passed"}}}]}
    assert not check_outcomes(report, "tests/test_x.py::test_x")


@pytest.mark.parametrize("change", ["missing", "wrong-node", "setup-only", "collection-error"])
def test_strict_gate_requires_actual_selected_call(change):
    from tools.l15_runtime import check_outcomes

    report = {"artifact": "l15-pytest-outcomes", "exitstatus": 0,
              "tests": [{"nodeid": "tests/test_x.py::test_x", "outcome": "passed",
                         "phases": {"call": {"outcome": "passed"}}}]}
    assert check_outcomes(report, "tests/test_x.py::test_x")
    if change == "missing":
        report["tests"] = []
    elif change == "wrong-node":
        report["tests"][0]["nodeid"] = "tests/test_other.py::test_x"
    elif change == "setup-only":
        report["tests"][0]["phases"] = {"setup": {"outcome": "passed"}}
    else:
        report["collection_errors"] = ["broken import"]
    assert not check_outcomes(report, "tests/test_x.py::test_x")


@pytest.mark.parametrize("mark,outcome", [
    ("skip(reason='no HCU')", "skipped"), ("xfail(reason='')", "xpass"),
])
def test_real_pytest_skip_is_not_a_green_case(tmp_path, mark, outcome):
    import os
    import subprocess
    import sys
    from tools.l15_runtime import check_outcomes

    (tmp_path / "test_required.py").write_text(
        "import pytest\n"
        f"@pytest.mark.{mark}\n"
        "def test_gpu(): pass\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "outcomes.json"
    env = dict(os.environ, PYTHONPATH=str(ROOT), HCU_ADAPTER_HOST="none",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", L15_PYTEST_REPORT=str(report_path))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "tools.l15_pytest_report",
         "test_required.py::test_gpu"], env=env, cwd=tmp_path,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["tests"][0]["outcome"] == outcome
    assert not check_outcomes(report, "test_required.py::test_gpu")


def test_passing_parent_cannot_erase_a_skipped_subtest():
    from types import SimpleNamespace
    from tools import l15_pytest_report as plugin
    from tools.l15_runtime import check_outcomes

    plugin.pytest_sessionstart(None)
    node = "tests/test_x.py::test_x"
    plugin.pytest_runtest_logreport(SimpleNamespace(
        nodeid=node, when="call", outcome="skipped", failed=False,
        skipped=True, passed=False, longrepr="unsupported shape", context=object(),
    ))
    plugin.pytest_runtest_logreport(SimpleNamespace(
        nodeid=node, when="call", outcome="passed", failed=False,
        skipped=False, passed=True,
    ))
    document = {"artifact": "l15-pytest-outcomes", "exitstatus": 0,
                "tests": list(plugin._RESULTS.values())}
    assert not check_outcomes(document, node)
