# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Byte integrity must detect modifications hidden from ordinary git status."""

import os
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools.upstream_source import audit


@pytest.fixture
def tree(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args])
    git("init", "-q")
    git("config", "user.name", "Adapter Test")
    git("config", "user.email", "adapter-tests@example.com")
    (tmp_path / "common.py").write_bytes(b"def original(): return 1\n")
    (tmp_path / ".gitignore").write_bytes(b"ignored/\n")
    git("add", ".")
    git("commit", "-qm", "community source")
    return tmp_path, git


def test_clean_source_and_index_only_changes(tree):
    path, git = tree
    result = audit(path)
    assert result["passed"] and result["tracked_files"] == 2
    # An index does not define the accepted upstream source.
    (path / "common.py").write_bytes(b"staged change\n")
    git("add", ".")
    (path / "common.py").write_bytes(b"def original(): return 1\n")
    assert audit(path)["passed"]


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_index_flags_cannot_hide_source_rewrites(tree, flag):
    path, git = tree
    git("update-index", flag, "common.py")
    original = (path / "common.py").stat()
    (path / "common.py").write_bytes(b"def original(): return 2\n")  # same byte count
    os.utime(path / "common.py", ns=(original.st_atime_ns, original.st_mtime_ns))
    result = audit(path)
    assert not result["passed"]
    assert result["changes"][0]["path"] == "common.py"


@pytest.mark.parametrize("staged", [False, True])
def test_ignored_added_python_sources_are_not_hidden(tree, staged):
    path, git = tree
    ignored = path / "ignored"
    ignored.mkdir()
    (ignored / "override.py").write_bytes(b"extra import surface\n")
    (ignored / "bytecode.pyc").write_bytes(b"not a source file\n")
    if staged:
        git("add", "-f", "ignored/override.py")
    result = audit(path)
    assert not result["passed"] and result["added_sources"] == ["ignored/override.py"]


def test_deleted_source_fails(tree):
    path, _ = tree
    (path / "common.py").unlink()
    result = audit(path)
    assert not result["passed"] and result["changes"][0]["path"] == "common.py"


def test_dirty_source_preflight_refuses_before_binding_or_gpu_probe(tree):
    from tools.l15_runtime import preflight, UpstreamSourceError

    path, _ = tree
    (path / "common.py").write_bytes(b"dirty community source\n")
    with pytest.raises(UpstreamSourceError) as caught:
        preflight(path, path, path / "nonexistent-manifest.json")
    assert not caught.value.report["passed"]
    assert caught.value.report["changes"][0]["path"] == "common.py"


@pytest.mark.parametrize("fault", ["dirty-source", "missing-repository"])
def test_model_preflight_failure_overwrites_old_green_report(tree, monkeypatch, fault):
    from tools import l15_model_smoke as smoke

    path, _ = tree
    output = path / "old-model-report.json"
    output.write_text('{"passed": true}\n')
    if fault == "dirty-source":
        (path / "common.py").write_bytes(b"dirty source\n")
    else:
        path = path / "missing-repository"
    monkeypatch.setattr("sys.argv", [
        "l15_model_smoke", "--up-tree", str(path), "--families", "dense",
        "--out", str(output),
    ])
    assert smoke.main() == 2
    report = json.loads(output.read_text())
    assert not report["passed"] and not report["complete"] and report["jobs"] == []
    if fault == "dirty-source":
        assert not report["source_integrity"]["passed"]


def test_green_model_worker_cannot_hide_source_mutation(tree, tmp_path, monkeypatch):
    from tools import l15_model_smoke as smoke

    path, _ = tree
    before = audit(path)
    args = SimpleNamespace(
        up_tree=path, out=tmp_path / "reports/models.json", families="dense",
        modes="eager,graph", timeout=30, binding_manifest=path / "binding.json",
    )
    monkeypatch.setattr(smoke, "preflight",
                        lambda *a, **kw: {"upstream": {"commit": before["commit"]}})
    calls = []

    def job(argv, **kwargs):
        calls.append(argv)
        report = Path(argv[argv.index("--out") + 1])
        report.write_text(json.dumps({
            "artifact": "l15-model-worker", "passed": True,
            "family": "dense", "mode": "eager",
        }))
        (path / "common.py").write_bytes(b"adapter wrote upstream source\n")
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(smoke, "run_job", job)
    assert not smoke.run(args)
    result = json.loads(args.out.read_text())
    assert not result["passed"] and not result["complete"]
    assert len(calls) == 1  # graph mode never launches on damaged source
    assert result["jobs"][0]["evidence"]["passed"]  # preserve worker evidence
    assert not result["jobs"][0]["source_integrity_after"]["passed"]


def test_green_component_cannot_hide_source_mutation(tree, tmp_path, monkeypatch):
    from tools import l15_family_matrix as matrix

    path, _ = tree
    before = audit(path)
    adapter = tmp_path.with_name(tmp_path.name + "-adapter")
    (adapter / "tests").mkdir(parents=True)
    (adapter / "tests/worker.py").write_text("def test_ok(): pass\n")
    (adapter / "tests/binding.json").write_text("{}")
    output = adapter / "result.json"
    node = "tests/worker.py::test_ok"
    document = {
        "manifest_id": "fixture", "upstream_manifest": "tests/binding.json",
        "defaults": {"timeout": 30}, "families": [
            {"id": "dense", "cases": [
                {"id": "fixture", "kind": "adapter-pytest", "nodeid": node},
            ]},
        ],
    }
    monkeypatch.setattr(matrix, "preflight",
                        lambda *a: {"upstream": {"commit": before["commit"]}})
    actual_run = subprocess.run

    def run(argv, **kwargs):
        if argv[0] == "git":
            return actual_run(argv, **kwargs)
        Path(kwargs["env"]["L15_PYTEST_REPORT"]).write_text(json.dumps({
            "artifact": "l15-pytest-outcomes", "exitstatus": 0,
            "collection_errors": [], "tests": [{
                "nodeid": node, "outcome": "passed",
                "phases": {"call": {"outcome": "passed"}},
            }],
        }))
        (path / "common.py").write_bytes(b"source changed during component\n")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(matrix.subprocess, "run", run)
    assert not matrix.run(document, path, adapter, {"dense"}, output)
    result = json.loads(output.read_text())
    assert result["results"][0]["passed"]  # numerical outcome is retained
    assert not result["passed"] and not result["source_integrity_after"]["passed"]


@pytest.mark.skipif(os.name != "posix", reason="real POSIX symlink semantics")
def test_symlinks_are_compared_without_reading_their_targets(tree, tmp_path):
    path, git = tree
    (path / "alias.py").symlink_to("common.py")
    git("add", "alias.py")
    git("commit", "-qm", "community symlink")
    assert audit(path)["passed"]
    (path / "alias.py").unlink()
    (path / "alias.py").symlink_to("/outside/missing.py")
    result = audit(path)
    assert not result["passed"] and result["changes"][0]["path"] == "alias.py"
