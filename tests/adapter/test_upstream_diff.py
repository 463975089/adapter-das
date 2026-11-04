# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""A clean baseline and deliberate upstream breakages must be distinguished."""

import subprocess

from ci.upstream_diff import compare, snapshot


BASELINE = {
    "python/sglang/example.py": "def compute(x, scale=1):\n    return x * scale\n# needle\n",
    "test/registered/test_example.py": "class TestCase:\n    def test_run(self):\n        assert True\n",
}
TARGET = "sglang.example.compute"
NODE = "test/registered/test_example.py::TestCase::test_run"


def diff(candidate):
    return compare(BASELINE, candidate, [TARGET], [("example", "# needle")], [NODE])


def test_identical_baseline_has_no_false_drift():
    report = diff(dict(BASELINE))
    assert report["passed"] and not report["baseline_gaps"]
    assert len(report["checked"]) == 3


def test_signature_body_and_missing_symbol_are_visible():
    for content, kind in (
        ("def compute(x):\n    return x\n# needle\n", "signature"),
        ("def compute(x, scale=1):\n    return x + scale\n# needle\n", "source"),
        ("def renamed(x):\n    return x\n# needle\n", "missing-symbol"),
    ):
        candidate = dict(BASELINE, **{"python/sglang/example.py": content})
        report = diff(candidate)
        assert not report["passed"]
        assert kind in {change["kind"] for change in report["changes"]}


def test_needle_and_nodeid_removal_are_visible():
    candidate = dict(BASELINE)
    candidate["python/sglang/example.py"] = candidate["python/sglang/example.py"].replace("# needle", "")
    candidate["test/registered/test_example.py"] = "class Renamed:\n    pass\n"
    report = diff(candidate)
    assert not report["passed"]
    assert {change["kind"] for change in report["changes"]} == {"needle", "missing-nodeid"}


def test_unknown_baseline_contract_is_reported_as_a_gap():
    report = compare(BASELINE, BASELINE, ["sglang.example.unknown"], [], [])
    assert report["passed"] and report["baseline_gaps"]
    assert not report["checked"]


def test_needle_copy_in_new_file_requires_review():
    candidate = dict(BASELINE)
    candidate["python/sglang/copied.py"] = "# needle\n"
    report = diff(candidate)
    assert not report["passed"]
    change, = report["changes"]
    assert change["kind"] == "needle"
    assert change["before_count"] == 0 and change["after_count"] == 1


def test_snapshots_use_distinct_commits_and_ignore_dirty_worktree(tmp_path):
    def git(*argv):
        return subprocess.check_output(["git", "-C", str(tmp_path), *argv], text=True).strip()
    git("init", "-q")
    git("config", "user.email", "1263739855@qq.com")
    git("config", "user.name", "Adapter Test")
    path = tmp_path / "python/sglang/example.py"
    path.parent.mkdir(parents=True)
    path.write_text(BASELINE["python/sglang/example.py"])
    git("add", ".")
    git("commit", "-qm", "baseline")
    first = git("rev-parse", "HEAD")
    path.write_text("def compute(x):\n    return x\n")
    git("add", ".")
    git("commit", "-qm", "candidate")
    second = git("rev-parse", "HEAD")
    path.write_text("dirty worktree must not replace either revision\n")
    old_commit, old = snapshot(tmp_path, first)
    new_commit, new = snapshot(tmp_path, second)
    assert old_commit != new_commit
    assert old[path.relative_to(tmp_path).as_posix()] != new[path.relative_to(tmp_path).as_posix()]
    report = compare(old, new, [TARGET], [], [])
    assert not report["passed"] and report["changes"][0]["kind"] == "signature"
