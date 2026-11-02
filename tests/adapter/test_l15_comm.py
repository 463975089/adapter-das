"""CPU-only contract tests for tools/l15_comm.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "l15_comm.py"
SPEC = importlib.util.spec_from_file_location("l15_comm", MODULE_PATH)
assert SPEC and SPEC.loader
comm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comm)


def fixture_manifest():
    return {
        "schema_version": 1,
        "manifest_id": "fixture-comm",
        "layer": "L1.5-comm",
        "defaults": {
            "device_count": 2,
            "classification": "candidate-observe",
            "external_network": False,
            "needs_model_weights": False,
        },
        "cases": [
            {
                "id": "gather",
                "kind": "torchrun-script",
                "path": "tests/gather.py",
                "primitive": "all_gather",
                "backend": "nccl",
                "timeout": 10,
                "owner": "comm",
            },
            {
                "id": "ar",
                "kind": "pytest",
                "path": "tests/test_ar.py",
                "nodeid": "tests/test_ar.py::test_ar",
                "primitive": "all_reduce",
                "backend": "aiter",
                "timeout": 10,
                "owner": "comm",
            },
        ],
    }


class CommManifestTests(unittest.TestCase):
    def test_repository_manifest_is_valid(self):
        document = json.loads(
            (ROOT / "tests" / "manifests" / "l15_comm_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(comm.validate_manifest(document), [])

    def test_requires_exactly_two_devices(self):
        document = fixture_manifest()
        document["defaults"]["device_count"] = 1
        self.assertTrue(comm.validate_manifest(document))

    def test_rejects_unsafe_path(self):
        document = fixture_manifest()
        document["cases"][0]["path"] = "../worker.py"
        self.assertTrue(comm.validate_manifest(document))

    def test_builds_fixed_torchrun_argv(self):
        command = comm._case_command(fixture_manifest()["cases"][0], "python")
        self.assertEqual(
            command,
            [
                "python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                "tests/gather.py",
            ],
        )

    def test_unknown_case_fails_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(comm.CommManifestError, "unknown"):
                comm.run_manifest(
                    fixture_manifest(),
                    Path(temporary),
                    Path(temporary) / "out",
                    "observe",
                    {"typo"},
                )

    def test_observe_reports_failure_without_gating(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tests").mkdir()
            (root / "tests" / "gather.py").write_text("pass\n", encoding="utf-8")
            manifest = fixture_manifest()
            manifest["cases"] = [manifest["cases"][0]]
            original = comm.subprocess.run
            comm.subprocess.run = lambda *args, **kwargs: comm.subprocess.CompletedProcess(
                args[0], 1, "", "failure"
            )
            try:
                artifact, passed = comm.run_manifest(
                    manifest, root, root / "out", "observe"
                )
            finally:
                comm.subprocess.run = original
            self.assertTrue(passed)
            self.assertEqual(artifact["summary"], {"failed": 1})
            self.assertEqual(artifact["results"][0]["device_count"], 2)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("timeout", [True, False, 0, -1, 1.5, None])
def test_rejects_non_positive_integer_timeout(timeout):
    manifest = fixture_manifest()
    manifest["cases"][0]["timeout"] = timeout
    assert ".timeout: positive integer required" in comm.validate_manifest(manifest)[0]


@pytest.mark.parametrize("mode", ["observe", "blocking"])
@pytest.mark.parametrize(
    "returncode,stdout,outcome",
    [
        (1, "1 failed, 1 skipped in 0.1s", "failed"),
        (2, "1 skipped, 1 error in 0.1s", "failed"),
        (0, "1 skipped in 0.1s", "skipped"),
        (5, "no tests ran in 0.1s", "skipped"),
        (0, "1 passed in 0.1s", "passed"),
    ],
)
def test_pytest_outcome_preserves_failure_with_skipped_summary(
    tmp_path, mode, returncode, stdout, outcome
):
    manifest = fixture_manifest()
    manifest["cases"] = [manifest["cases"][1]]
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_ar.py").write_text("pass\n", encoding="utf-8")
    result = comm.subprocess.CompletedProcess([], returncode, stdout, "")
    with patch.object(comm.subprocess, "run", return_value=result):
        artifact, passed = comm.run_manifest(manifest, tmp_path, tmp_path / "out", mode)
    assert artifact["summary"] == {outcome: 1}
    assert artifact["hard_failures"] == (["ar"] if outcome == "failed" else [])
    assert passed is (mode == "observe" or outcome != "failed")


def test_torchrun_exit_five_is_failure(tmp_path):
    manifest = fixture_manifest()
    manifest["cases"] = [manifest["cases"][0]]
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/gather.py").write_text("pass\n", encoding="utf-8")
    result = comm.subprocess.CompletedProcess([], 5, "worker skipped a step", "")
    with patch.object(comm.subprocess, "run", return_value=result):
        artifact, passed = comm.run_manifest(manifest, tmp_path, tmp_path / "out", "blocking")
    assert not passed
    assert artifact["summary"] == {"failed": 1}
