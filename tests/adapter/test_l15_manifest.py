"""CPU-only tests for tools/l15_manifest.py (stdlib unittest)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "l15_manifest.py"
SPEC = importlib.util.spec_from_file_location("l15_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
l15 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(l15)


@pytest.mark.parametrize("value", [True, False])
def test_manifest_device_count_rejects_boolean(value):
    document = minimal_manifest()
    document["defaults"]["device_count"] = value
    assert "device_count" in l15.validate_manifest(document)[0]


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize(
    "kind,field",
    [
        ("l15-collect", "returncode"),
        ("l15-classification", "promoted_contract_green_count"),
        ("l15-classification", "unobserved_count"),
    ],
)
def test_artifact_integer_fields_reject_boolean(kind, field, value):
    document = {
        "schema_version": 1,
        "artifact": kind,
        "manifest_id": "fixture",
        "upstream": {"project": "sglang", "version": "fixture", "marker_hash": "fixture"},
        "nodeids": [],
        "missing_paths": [],
        "collection_errors": [],
        "returncode": 0,
        "timed_out": False,
        "outcome_summary": {},
        "classification_summary": {},
        "promoted_contract_green_count": 0,
        "unobserved_count": 0,
    }
    assert l15.validate_artifact(document) == []
    document[field] = value
    assert field in l15.validate_artifact(document)[0]


@pytest.mark.parametrize("use_defaults", [False, True])
def test_compaction_preserves_effective_classification_and_route_evidence(use_defaults):
    document = minimal_manifest()
    if use_defaults:
        document["defaults"].update(
            classification="contract-green", route_evidence="optional"
        )
    else:
        document["defaults"].pop("classification")
        document["defaults"].pop("route_evidence")
    source = {
        "id": "fixture",
        "domain": "kernel",
        "logical_ops": ["op"],
        "owner": "kernel",
        "paths": ["test/registered"],
    }
    document["selectors"] = [source]
    common = {
        "source_id": source["id"],
        "domain": source["domain"],
        "logical_ops": source["logical_ops"],
        "owner": source["owner"],
    }
    tests = [
        dict(common, nodeid="test/registered/test_x.py::test_default"),
        dict(common, nodeid="test/registered/test_x.py::test_core", classification="core-green", route_evidence="required"),
        dict(common, nodeid="test/registered/test_x.py::test_override", classification="candidate-observe", route_evidence="not_collected"),
    ]
    document["tests"] = tests
    assert l15.validate_manifest(document) == []
    compact = l15._compact_tests(document, tests)
    assert l15.validate_manifest(compact) == []
    expanded = {test["nodeid"]: test for test in l15._expanded_tests(compact)}
    for original in tests:
        restored = expanded[original["nodeid"]]
        assert l15._classification(restored, compact["defaults"]) == l15._classification(original, document["defaults"])
        assert l15._route_evidence(restored, compact["defaults"]) == l15._route_evidence(original, document["defaults"])


def minimal_manifest(**overrides):
    document = {
        "schema_version": 1,
        "manifest_id": "fixture-v1",
        "upstream": {
            "project": "sglang",
            "version": "v0.fixture",
            "binding_files": {
                "python/binding.txt": "e80b71cd14d3cbd65f4173abcbfcf01a545dbca32a72d575108b553a648cc96f"
            },
            "tree_markers": ["python/sglang", "test/registered"],
        },
        "defaults": {
            "device_count": 1,
            "needs_server": False,
            "needs_model_weights": False,
            "external_network": False,
            "route_evidence": "required",
            "classification": "candidate-observe",
        },
        "selectors": [],
        "candidates": [],
        "tests": [],
    }
    document.update(overrides)
    return document


def make_upstream(root: Path) -> Path:
    tree = root / "upstream"
    (tree / "python" / "sglang").mkdir(parents=True)
    (tree / "test" / "registered").mkdir(parents=True)
    (tree / "python" / "binding.txt").write_bytes(b"fixture\n")
    return tree


class L15ManifestTests(unittest.TestCase):
    def test_repository_manifest_is_valid(self):
        document = json.loads(
            (ROOT / "tests" / "manifests" / "l15_v0522.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(l15.validate_manifest(document), [])

    def test_validation_rejects_unsafe_candidate_path(self):
        candidate = {
            "id": "escape",
            "batch": "batch-1",
            "domain": "kernel",
            "path": "test/registered/../../secret.py",
            "logical_ops": ["op"],
            "expected_backends": ["triton"],
            "owner": "kernel",
        }
        errors = l15.validate_manifest(minimal_manifest(candidates=[candidate]))
        self.assertTrue(errors)
        self.assertIn("'..'", errors[0])

    def test_validation_requires_diagnostic_fingerprint_and_conditions(self):
        candidate = {
            "id": "known-fail",
            "batch": "batch-1",
            "domain": "kernel",
            "path": "test/registered/test_x.py",
            "logical_ops": ["op"],
            "expected_backends": ["triton"],
            "classification": "diagnostic-known-fail",
            "route_evidence": "not_collected",
            "owner": "kernel",
        }
        errors = l15.validate_manifest(minimal_manifest(candidates=[candidate]))
        self.assertTrue(errors)
        self.assertIn("fingerprint", errors[0])

    def test_validation_rejects_core_without_route_evidence(self):
        test = {
            "nodeid": "test/registered/test_x.py::test_x[param]",
            "domain": "kernel",
            "logical_ops": ["op"],
            "classification": "core-green",
            "route_evidence": "optional",
            "owner": "kernel",
        }
        errors = l15.validate_manifest(minimal_manifest(tests=[test]))
        self.assertTrue(errors)
        self.assertIn("core-green", errors[0])

    def test_inventory_is_stable_and_marks_missing_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            tree = make_upstream(Path(temporary))
            existing = tree / "test" / "registered" / "test_x.py"
            existing.write_text("def test_x(): pass\n", encoding="utf-8")
            candidates = [
                {
                    "id": "x",
                    "batch": "batch-1",
                    "domain": "kernel",
                    "path": "test/registered/test_x.py",
                    "logical_ops": ["op"],
                    "expected_backends": ["triton"],
                    "owner": "kernel",
                },
                {
                    "id": "missing",
                    "batch": "batch-1",
                    "domain": "kernel",
                    "path": "test/registered/test_missing.py",
                    "logical_ops": ["op"],
                    "expected_backends": ["triton"],
                    "owner": "kernel",
                },
            ]
            manifest = minimal_manifest(candidates=candidates)
            first = l15.build_inventory(manifest, tree)
            second = l15.build_inventory(manifest, tree)
            self.assertEqual(first, second)
            self.assertEqual(first["summary"], {"total": 2, "ready": 1, "missing": 1})

    def test_inventory_rejects_wrong_upstream_binding_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            tree = make_upstream(Path(temporary))
            (tree / "python" / "binding.txt").write_text(
                "wrong tree\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(l15.ManifestError, "binding mismatch"):
                l15.build_inventory(minimal_manifest(), tree)

    def test_inventory_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            tree = make_upstream(base)
            outside = base / "outside.py"
            outside.write_text("def test_escape(): pass\n", encoding="utf-8")
            link = tree / "test" / "registered" / "escape.py"
            try:
                link.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            candidate = {
                "id": "escape",
                "batch": "batch-1",
                "domain": "kernel",
                "path": "test/registered/escape.py",
                "logical_ops": ["op"],
                "expected_backends": ["triton"],
                "owner": "kernel",
            }
            with self.assertRaisesRegex(l15.ManifestError, "escapes tree"):
                l15.build_inventory(minimal_manifest(candidates=[candidate]), tree)

    def test_build_candidate_baseline_maps_selector_and_candidate(self):
        manifest = minimal_manifest(
            selectors=[
                {
                    "id": "attention",
                    "domain": "attention-core",
                    "paths": ["test/registered/attention"],
                    "owner": "attention",
                }
            ],
            candidates=[
                {
                    "id": "kernel-x",
                    "batch": "batch-1",
                    "domain": "kernel",
                    "path": "test/registered/test_x.py",
                    "logical_ops": ["op_x"],
                    "expected_backends": ["triton"],
                    "owner": "kernel",
                }
            ],
        )
        collected = {
            "artifact": "l15-collect",
            "manifest_id": "fixture-v1",
            "nodeids": [
                "test/registered/attention/test_a.py::test_a[p]",
                "test/registered/test_x.py::test_x",
            ],
        }
        baseline = l15.build_candidate_baseline(manifest, collected)
        expanded = l15._expanded_tests(baseline)
        self.assertEqual(len(expanded), 2)
        self.assertEqual(len(baseline["test_groups"]), 2)
        by_nodeid = {item["nodeid"]: item for item in expanded}
        self.assertEqual(
            by_nodeid["test/registered/test_x.py::test_x"]["source_id"],
            "kernel-x",
        )
        self.assertEqual(
            by_nodeid["test/registered/attention/test_a.py::test_a[p]"]["domain"],
            "attention-core",
        )
        self.assertTrue(
            all(item["classification"] == "candidate-observe" for item in expanded)
        )

    def test_classify_outcomes_promotes_contract_and_platform_skip(self):
        tests = [
            {
                "nodeid": "test/registered/test_x.py::test_pass",
                "source_id": "x",
                "domain": "kernel",
                "logical_ops": ["op"],
                "classification": "candidate-observe",
                "route_evidence": "not_collected",
                "owner": "kernel",
            },
            {
                "nodeid": "test/registered/test_x.py::test_skip",
                "source_id": "x",
                "domain": "kernel",
                "logical_ops": ["op"],
                "classification": "candidate-observe",
                "route_evidence": "not_collected",
                "owner": "kernel",
            },
            {
                "nodeid": "test/registered/test_x.py::test_pending",
                "source_id": "x",
                "domain": "kernel",
                "logical_ops": ["op"],
                "classification": "candidate-observe",
                "route_evidence": "not_collected",
                "owner": "kernel",
            },
        ]
        outcomes = {
            "artifact": "l15-pytest-outcomes",
            "tests": [
                {"nodeid": tests[0]["nodeid"], "outcome": "passed", "phases": {}},
                {
                    "nodeid": tests[1]["nodeid"],
                    "outcome": "skipped",
                    "phases": {"setup": {"reason": "unsupported platform"}},
                },
            ],
        }
        source = {
            "id": "x",
            "batch": "batch-1",
            "domain": "kernel",
            "path": "test/registered/test_x.py",
            "logical_ops": ["op"],
            "expected_backends": ["triton"],
            "owner": "kernel",
        }
        classified, report = l15.classify_outcomes(
            minimal_manifest(tests=tests, candidates=[source]), [outcomes]
        )
        by_nodeid = {item["nodeid"]: item for item in l15._expanded_tests(classified)}
        self.assertEqual(by_nodeid[tests[0]["nodeid"]]["classification"], "contract-green")
        self.assertEqual(by_nodeid[tests[1]["nodeid"]]["classification"], "platform-skip")
        self.assertEqual(by_nodeid[tests[2]["nodeid"]]["classification"], "candidate-observe")
        self.assertEqual(
            report["outcome_summary"],
            {"passed": 1, "skipped": 1, "unobserved": 1},
        )
        self.assertEqual(
            report["classification_summary"],
            {"candidate-observe": 1, "contract-green": 1, "platform-skip": 1},
        )

    def test_compact_groups_expand_with_sparse_override(self):
        nodeids = [
            "test/registered/test_x.py::test_a",
            "test/registered/test_x.py::test_b[param]",
        ]
        manifest = minimal_manifest(
            tests=None,
            test_groups=[
                {
                    "source_id": "x",
                    "domain": "kernel",
                    "logical_ops": ["op"],
                    "owner": "kernel",
                    "classification": "contract-green",
                    "route_evidence": "not_collected",
                    "nodeids": nodeids,
                }
            ],
            test_overrides={
                nodeids[1]: {
                    "classification": "platform-skip",
                    "platform_guard": "pytest-runtime-skip",
                    "skip_reason": "unsupported platform",
                }
            },
        )
        manifest.pop("tests")
        self.assertEqual(l15.validate_manifest(manifest), [])
        expanded = l15._expanded_tests(manifest)
        self.assertEqual([item["nodeid"] for item in expanded], nodeids)
        self.assertEqual(expanded[0]["classification"], "contract-green")
        self.assertEqual(expanded[1]["classification"], "platform-skip")

    def test_compact_groups_reject_duplicate_and_unknown_override(self):
        nodeid = "test/registered/test_x.py::test_a"
        group = {
            "source_id": "x",
            "domain": "kernel",
            "logical_ops": ["op"],
            "owner": "kernel",
            "classification": "candidate-observe",
            "route_evidence": "not_collected",
            "nodeids": [nodeid],
        }
        duplicate = minimal_manifest(test_groups=[group, group])
        duplicate.pop("tests")
        self.assertIn("duplicate nodeid", l15.validate_manifest(duplicate)[0])
        unknown = minimal_manifest(
            test_groups=[group],
            test_overrides={"test/registered/test_x.py::missing": {}},
        )
        unknown.pop("tests")
        self.assertIn("unknown nodeids", l15.validate_manifest(unknown)[0])

    def test_inventory_artifact_validation(self):
        artifact = {
            "schema_version": 1,
            "artifact": "l15-inventory",
            "manifest_id": "fixture-v1",
            "upstream": {
                "project": "sglang",
                "version": "v0.fixture",
                "git_head": None,
                "marker_hash": "abc",
            },
            "entries": [],
            "summary": {"total": 0, "ready": 0, "missing": 0},
        }
        self.assertEqual(l15.validate_artifact(artifact), [])

    def test_selection_groups_keep_selector_filters_local(self):
        with tempfile.TemporaryDirectory() as temporary:
            tree = make_upstream(Path(temporary))
            selector_dir = tree / "test" / "registered" / "attention"
            selector_dir.mkdir()
            candidate_file = tree / "test" / "registered" / "candidate.py"
            candidate_file.write_text("def test_x(): pass\n", encoding="utf-8")
            manifest = minimal_manifest(
                selectors=[
                    {
                        "id": "attention",
                        "domain": "attention",
                        "paths": ["test/registered/attention"],
                        "exclude_keywords": ["tilelang"],
                        "owner": "attention",
                    }
                ],
                candidates=[
                    {
                        "id": "candidate",
                        "batch": "batch-1",
                        "domain": "kernel",
                        "path": "test/registered/candidate.py",
                        "logical_ops": ["op"],
                        "expected_backends": ["triton"],
                        "owner": "kernel",
                    }
                ],
            )
            groups, missing = l15._selection_groups(
                manifest, tree, {"current", "batch-1"}
            )
            self.assertEqual(missing, [])
            self.assertEqual(groups[0]["exclude_keywords"], ["tilelang"])
            self.assertEqual(groups[1]["exclude_keywords"], [])
            self.assertEqual(groups[1]["id"], "candidate")
            self.assertEqual(groups[1]["paths"], ["test/registered/candidate.py"])

    def test_collect_merges_group_results_and_preserves_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            tree = make_upstream(Path(temporary))
            first = tree / "test" / "registered" / "a.py"
            second = tree / "test" / "registered" / "b.py"
            first.write_text("def test_a(): pass\n", encoding="utf-8")
            second.write_text("def test_b(): pass\n", encoding="utf-8")
            manifest = minimal_manifest(
                selectors=[
                    {
                        "id": "current",
                        "domain": "attention",
                        "paths": ["test/registered/a.py"],
                        "exclude_keywords": ["tilelang"],
                        "owner": "attention",
                    }
                ],
                candidates=[
                    {
                        "id": "candidate",
                        "batch": "batch-1",
                        "domain": "kernel",
                        "path": "test/registered/b.py",
                        "logical_ops": ["op"],
                        "expected_backends": ["triton"],
                        "owner": "kernel",
                    }
                ],
            )
            original = l15.subprocess.run
            outputs = iter(
                [
                    "test/registered/a.py::test_a\n1 test collected",
                    "test/registered/b.py::test_b\n1 test collected",
                ]
            )

            def fake_run(command, **kwargs):
                if command[:2] == ["git", "-C"]:
                    return l15.subprocess.CompletedProcess(command, 1, "", "")
                return l15.subprocess.CompletedProcess(command, 0, next(outputs), "")

            l15.subprocess.run = fake_run
            try:
                artifact = l15.collect_manifest(
                    manifest, tree, ROOT, {"current", "batch-1"}, 10
                )
            finally:
                l15.subprocess.run = original
            self.assertEqual(artifact["returncode"], 0)
            self.assertEqual(
                artifact["nodeids"],
                [
                    "test/registered/a.py::test_a",
                    "test/registered/b.py::test_b",
                ],
            )
            self.assertEqual(artifact["upstream"]["project"], "sglang")
            self.assertEqual(len(artifact["commands"]), 2)

    def test_parse_nodeids_keeps_parameterized_and_unicode_ids(self):
        output = "\n".join(
            [
                "test/registered/test_x.py::test_x[a::b]",
                "test/registered/test_x.py::test_中文[param]",
                "registered/test_y.py::test_y[with space]",
                "2 tests collected in 0.01s",
            ]
        )
        self.assertEqual(
            l15._parse_nodeids(output),
            [
                "test/registered/test_x.py::test_x[a::b]",
                "test/registered/test_x.py::test_中文[param]",
                "test/registered/test_y.py::test_y[with space]",
            ],
        )

    def test_verify_detects_added_missing_swap_with_same_count(self):
        tests = [
            {
                "nodeid": "test/registered/test_x.py::test_old",
                "domain": "kernel",
                "logical_ops": ["op"],
                "classification": "core-green",
                "owner": "kernel",
            }
        ]
        manifest = minimal_manifest(tests=tests)
        collected = {
            "artifact": "l15-collect",
            "manifest_id": "fixture-v1",
            "upstream": {"project": "sglang", "version": "v0.fixture"},
            "returncode": 0,
            "timed_out": False,
            "nodeids": ["test/registered/test_x.py::test_new"],
            "collection_errors": [],
            "missing_paths": [],
        }
        report, passed = l15.verify_collection(manifest, collected, "blocking")
        self.assertFalse(passed)
        self.assertEqual(report["added"], ["test/registered/test_x.py::test_new"])
        self.assertEqual(
            report["missing_core"], ["test/registered/test_x.py::test_old"]
        )

    def test_observe_reports_unclassified_nodes_without_failing(self):
        manifest = minimal_manifest()
        collected = {
            "artifact": "l15-collect",
            "manifest_id": "fixture-v1",
            "upstream": {"project": "sglang", "version": "v0.fixture"},
            "returncode": 0,
            "timed_out": False,
            "nodeids": ["test/registered/test_x.py::test_new"],
            "collection_errors": [],
            "missing_paths": [],
        }
        report, passed = l15.verify_collection(manifest, collected, "observe")
        self.assertTrue(passed)
        self.assertTrue(report["warnings"])
        self.assertFalse(report["baseline_present"])
        self.assertEqual(
            report["added"], ["test/registered/test_x.py::test_new"]
        )

    def test_blocking_requires_exact_nodeid_baseline(self):
        collected = {
            "artifact": "l15-collect",
            "manifest_id": "fixture-v1",
            "upstream": {"project": "sglang", "version": "v0.fixture"},
            "returncode": 0,
            "timed_out": False,
            "nodeids": ["test/registered/test_x.py::test_new"],
            "collection_errors": [],
            "missing_paths": [],
        }
        report, passed = l15.verify_collection(
            minimal_manifest(), collected, "blocking"
        )
        self.assertFalse(passed)
        self.assertIn(
            "blocking mode requires a non-empty exact nodeid baseline",
            report["hard_failures"],
        )

    def test_selection_rejects_unknown_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            tree = make_upstream(Path(temporary))
            with self.assertRaisesRegex(l15.ManifestError, "unknown L1.5 batches"):
                l15._selection_groups(minimal_manifest(), tree, {"current", "batch-11"})

    def test_observe_matches_known_diagnostic_collection_error(self):
        diagnostic = {
            "id": "known-fail",
            "batch": "batch-1",
            "domain": "kernel",
            "path": "test/registered/test_x.py",
            "logical_ops": ["op"],
            "expected_backends": ["triton"],
            "classification": "diagnostic-known-fail",
            "route_evidence": "not_collected",
            "owner": "kernel",
            "fingerprint": "ImportError: expected ABI",
            "review_condition": "ABI is available",
            "promotion_condition": "stable green",
        }
        manifest = minimal_manifest(candidates=[diagnostic])
        collected = {
            "artifact": "l15-collect",
            "manifest_id": "fixture-v1",
            "upstream": {"project": "sglang", "version": "v0.fixture"},
            "returncode": 2,
            "timed_out": False,
            "nodeids": [],
            "collection_errors": ["ImportError: expected ABI from extension"],
            "group_results": [
                {
                    "id": "known-fail",
                    "collection_errors": ["ImportError: expected ABI from extension"],
                }
            ],
            "missing_paths": [],
        }
        report, passed = l15.verify_collection(manifest, collected, "observe")
        self.assertTrue(passed)
        self.assertEqual(report["matched_diagnostics"][0]["id"], "known-fail")
        self.assertEqual(report["unmatched_collection_errors"], [])

    def test_observe_still_fails_collection_infrastructure_errors(self):
        collected = {
            "artifact": "l15-collect",
            "manifest_id": "fixture-v1",
            "upstream": {"project": "sglang", "version": "v0.fixture"},
            "returncode": 2,
            "timed_out": False,
            "nodeids": [],
            "collection_errors": [],
            "missing_paths": [],
        }
        report, passed = l15.verify_collection(
            minimal_manifest(), collected, "observe"
        )
        self.assertFalse(passed)
        self.assertIn("pytest collect returned 2", report["hard_failures"])

    def test_verify_rejects_wrong_upstream_binding(self):
        collected = {
            "artifact": "l15-collect",
            "manifest_id": "fixture-v1",
            "upstream": {"project": "sglang", "version": "wrong"},
            "returncode": 0,
            "timed_out": False,
            "nodeids": [],
            "collection_errors": [],
            "missing_paths": [],
        }
        with self.assertRaisesRegex(l15.ManifestError, "upstream binding"):
            l15.verify_collection(minimal_manifest(), collected, "observe")

    def test_run_splits_duplicate_basenames_and_merges_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            tree = make_upstream(Path(temporary))
            for path in (
                "test/registered/a/test_same.py",
                "test/registered/b/test_same.py",
                "test/registered/c/test_other.py",
            ):
                target = tree / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("def test_x(): pass\n", encoding="utf-8")
            nodeids = [
                "test/registered/a/test_same.py::test_a",
                "test/registered/b/test_same.py::test_b",
                "test/registered/c/test_other.py::test_c",
            ]
            tests = [
                {
                    "nodeid": nodeid,
                    "source_id": "fixture",
                    "domain": "kernel",
                    "logical_ops": ["op"],
                    "classification": "contract-green",
                    "route_evidence": "not_collected",
                    "owner": "kernel",
                }
                for nodeid in nodeids
            ]
            calls = []
            original = l15.subprocess.run

            def fake_run(command, **kwargs):
                if command[:2] == ["git", "-C"]:
                    return l15.subprocess.CompletedProcess(command, 1, "", "")
                calls.append((command, kwargs))
                report = Path(kwargs["env"]["L15_PYTEST_REPORT"])
                selected = command[command.index("tools.l15_pytest_report") + 1 :]
                report.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "artifact": "l15-pytest-outcomes",
                            "exitstatus": 0,
                            "tests": [
                                {"nodeid": nodeid, "outcome": "passed", "phases": {}}
                                for nodeid in selected
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return l15.subprocess.CompletedProcess(
                    command, 0, f"stdout-{len(calls)}", f"stderr-{len(calls)}"
                )

            l15.subprocess.run = fake_run
            outcome_path = Path(temporary) / "outcomes.json"
            try:
                artifact, passed = l15.run_manifest(
                    minimal_manifest(tests=tests),
                    tree,
                    ROOT,
                    {"contract-green"},
                    None,
                    17,
                    outcome_path,
                )
            finally:
                l15.subprocess.run = original
            self.assertTrue(passed)
            self.assertEqual(len(calls), 2)
            self.assertEqual(
                calls[0][0][-2:],
                [nodeids[0], nodeids[2]],
            )
            self.assertEqual(calls[1][0][-1:], [nodeids[1]])
            self.assertTrue(all(call[1]["timeout"] == 17 for call in calls))
            self.assertTrue(all(call[1]["check"] is False for call in calls))
            self.assertEqual(artifact["timeout_seconds_per_group"], 17)
            self.assertEqual(artifact["returncode"], 0)
            self.assertEqual(artifact["outcome_summary"], {"passed": 3})
            self.assertIn("stdout-1", artifact["stdout"])
            self.assertIn("stdout-2", artifact["stdout"])
            merged = json.loads(outcome_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [record["nodeid"] for record in merged["tests"]], sorted(nodeids)
            )
            self.assertFalse(list(Path(temporary).glob(".outcomes.json.group-*.json")))

    def test_run_aggregates_group_failure_and_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            tree = make_upstream(Path(temporary))
            nodeids = [
                "test/registered/a/test_same.py::test_a",
                "test/registered/b/test_same.py::test_b",
            ]
            tests = [
                {
                    "nodeid": nodeid,
                    "domain": "kernel",
                    "logical_ops": ["op"],
                    "classification": "contract-green",
                    "route_evidence": "not_collected",
                    "owner": "kernel",
                }
                for nodeid in nodeids
            ]
            original = l15.subprocess.run
            count = 0

            def fake_run(command, **kwargs):
                nonlocal count
                if command[:2] == ["git", "-C"]:
                    return l15.subprocess.CompletedProcess(command, 1, "", "")
                count += 1
                if count == 2:
                    raise l15.subprocess.TimeoutExpired(
                        command, kwargs["timeout"], output="partial", stderr="slow"
                    )
                report = Path(kwargs["env"]["L15_PYTEST_REPORT"])
                report.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "artifact": "l15-pytest-outcomes",
                            "exitstatus": 1,
                            "tests": [
                                {"nodeid": nodeids[0], "outcome": "failed", "phases": {}}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return l15.subprocess.CompletedProcess(command, 1, "failed", "trace")

            l15.subprocess.run = fake_run
            outcome_path = Path(temporary) / "outcomes.json"
            try:
                artifact, passed = l15.run_manifest(
                    minimal_manifest(tests=tests), tree, ROOT,
                    {"contract-green"}, None, 5, outcome_path
                )
            finally:
                l15.subprocess.run = original
            self.assertFalse(passed)
            self.assertEqual(artifact["returncode"], 124)
            self.assertTrue(artifact["timed_out"])
            self.assertEqual([g["returncode"] for g in artifact["process_groups"]], [1, 124])
            self.assertEqual(artifact["outcome_summary"], {"failed": 1})
            self.assertIn("partial", artifact["stdout"])
            self.assertIn("slow", artifact["stderr"])

    def test_run_refuses_manifest_without_exact_nodeids(self):
        with tempfile.TemporaryDirectory() as temporary:
            tree = make_upstream(Path(temporary))
            with self.assertRaisesRegex(l15.ManifestError, "no exact nodeids"):
                l15.run_manifest(
                    minimal_manifest(),
                    tree,
                    ROOT,
                    {"core-green"},
                    None,
                    1,
                    Path(temporary) / "outcomes.json",
                )

    def test_atomic_json_write_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested" / "artifact.json"
            l15._write_json(target, {"z": 1, "a": [2]})
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")), {"a": [2], "z": 1}
            )
            self.assertTrue(target.read_bytes().endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
