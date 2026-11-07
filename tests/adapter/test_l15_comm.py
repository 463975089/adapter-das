# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

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


def ct_case():
    return dict(
        id="ct", kind="ct-int8-deepep", path=comm.CT_PROBE,
        primitive="dispatch+int8-experts+combine", backend="deepep",
        recv_hook=True, swiglu_limit=.75, timeout=600, owner="moe",
    )


def rank_report(rank):
    return dict(
        rank=rank, communication_passed=True, model_acceptance=False,
        community_dispatcher=True, recv_hook=True, swiglu_limit=.75,
        community_commit="fixture-commit",
        checks=[dict(phase=phase, max_abs=0)
                for phase in ["eager", "graph-0", "graph-1", "graph-2", "graph-3"]],
    )


def write_ranks(path, reports):
    path.mkdir(parents=True, exist_ok=True)
    for rank, report in enumerate(reports):
        (path / f"rank{rank}.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.mark.parametrize(
    "key,value",
    [("path", "tools/other.py"), ("recv_hook", 1),
     ("swiglu_limit", True), ("swiglu_limit", 0),
     ("swiglu_limit", float("nan")), ("swiglu_limit", float("inf"))],
)
def test_ct_rejects_unvalidated_execution_options(key, value):
    manifest = fixture_manifest()
    case = ct_case()
    case[key] = value
    manifest["cases"] = [case]
    assert comm.validate_manifest(manifest)


def test_ct_requires_tree_before_any_case_executes(tmp_path):
    manifest = fixture_manifest()
    manifest["cases"].append(ct_case())
    with patch.object(comm.subprocess, "run") as run:
        with pytest.raises(comm.CommManifestError, match="--up-tree"):
            comm.run_manifest(manifest, tmp_path, tmp_path / "out", "blocking")
    run.assert_not_called()


def test_ct_command_uses_fixed_module_and_separate_arguments(tmp_path):
    tree, report = tmp_path / "community tree", tmp_path / "rank reports"
    argv = comm._case_command(ct_case(), "python", up_tree=tree, rank_out=report)
    assert argv[-10:] == [
        "-m", "tools.ct_w4a8_deepep_probe", "--up-tree", str(tree),
        "--out", str(report), "--community-dispatcher",
        "--recv-hook", "--swiglu-limit", "0.75",
    ]


def test_ct_evidence_requires_both_ranks(tmp_path):
    write_ranks(tmp_path, [rank_report(0)])
    with pytest.raises(comm.CommManifestError, match="rank 1"):
        comm._ct_evidence(ct_case(), tmp_path, "fixture-commit")


@pytest.mark.parametrize(
    "key,value",
    [("rank", 0), ("rank", True), ("community_commit", "other-commit"),
     ("communication_passed", False), ("model_acceptance", True),
     ("community_dispatcher", False), ("recv_hook", False),
     ("swiglu_limit", None)],
)
def test_ct_rejects_mismatched_rank_evidence(tmp_path, key, value):
    reports = [rank_report(0), rank_report(1)]
    reports[1][key] = value
    write_ranks(tmp_path, reports)
    with pytest.raises(comm.CommManifestError, match="rank 1"):
        comm._ct_evidence(ct_case(), tmp_path, "fixture-commit")


@pytest.mark.parametrize("bad", ["missing", "duplicate", "error", "nan", "boolean"])
def test_ct_rejects_incomplete_or_inexact_phases(tmp_path, bad):
    reports = [rank_report(0), rank_report(1)]
    checks = reports[1]["checks"]
    if bad == "missing":
        checks.pop()
    elif bad == "duplicate":
        checks[-1]["phase"] = "graph-2"
    else:
        checks[-1]["max_abs"] = {"error": 1e-8, "nan": float("nan"), "boolean": False}[bad]
    write_ranks(tmp_path, reports)
    with pytest.raises(comm.CommManifestError, match="rank 1"):
        comm._ct_evidence(ct_case(), tmp_path, "fixture-commit")


def test_ct_requires_all_ten_exact_checks(tmp_path):
    write_ranks(tmp_path, [rank_report(0), rank_report(1)])
    evidence = comm._ct_evidence(ct_case(), tmp_path, "fixture-commit")
    assert evidence == {"rank_phase_checks": 10, "model_acceptance": False}


@pytest.mark.parametrize("changed_source", [False, True])
def test_ct_zero_exit_requires_reports_and_pristine_source(tmp_path, changed_source):
    from tools import l15_runtime

    (tmp_path / "tools").mkdir()
    (tmp_path / comm.CT_PROBE).write_text("pass\n", encoding="utf-8")
    manifest = fixture_manifest()
    manifest["cases"] = [ct_case()]
    before = dict(passed=True, commit="fixture-commit", changes=[], added_sources=[])
    after = dict(before, passed=not changed_source)

    def fake_job(argv, *, env, log, **kwargs):
        assert env["HCU_ADAPTER_HOST"] == "sglang"
        assert env["HCU_W4A8_MOE"] == "int8"
        assert env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] == "8"
        if changed_source:
            write_ranks(tmp_path / "out/ct/ranks", [rank_report(0), rank_report(1)])
        log.write_text("launcher exited zero\n", encoding="utf-8")
        return dict(returncode=0, timed_out=False)

    with patch.object(l15_runtime, "audit_source", side_effect=[before, after]), \
         patch.object(l15_runtime, "run_job", side_effect=fake_job):
        artifact, passed = comm.run_manifest(
            manifest, tmp_path, tmp_path / "out", "blocking",
            up_tree=tmp_path / "upstream")
    assert not passed
    assert artifact["summary"] == {"failed": 1}
    assert artifact["results"][0]["returncode"] == 0
    assert artifact["results"][0]["evidence_error"]
