# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Upgrade orchestration must not accept stale or unaudited evidence."""

import json
from types import SimpleNamespace

import pytest

from tools import l15_upgrade_check as upgrade


@pytest.fixture
def args(tmp_path):
    return SimpleNamespace(
        up_tree=tmp_path, baseline_ref="baseline", candidate_ref="candidate",
        out=tmp_path / "upgrade.json", checkpoint=tmp_path / "checkpoint",
        audit_only=False, timeout=30,
    )


def setup(monkeypatch, *, static_pass=True):
    monkeypatch.setattr(upgrade, "audit", lambda *a: {
        "artifact": "l15-upstream-diff",
        "passed": static_pass, "baseline": "old", "candidate": "new",
        "same_revision": False,
        "changes": [] if static_pass else [{"kind": "signature", "target": "example"}],
    })
    monkeypatch.setattr(upgrade, "git_stamp", lambda *a: {"commit": "new"})


def test_static_drift_blocks_gpu_jobs(monkeypatch, args):
    setup(monkeypatch, static_pass=False)
    monkeypatch.setattr(upgrade, "run_job", lambda *a, **k: pytest.fail("unexpected GPU job"))
    assert not upgrade.run(args)
    assert not json.loads(args.out.read_text())["ready"]


def test_audit_only_never_claims_runtime_readiness(monkeypatch, args):
    setup(monkeypatch)
    args.audit_only = True
    monkeypatch.setattr(upgrade, "run_job", lambda *a, **k: pytest.fail("unexpected GPU job"))
    assert upgrade.run(args)
    assert not json.loads(args.out.read_text())["ready"]


def test_stale_green_report_is_removed_before_job(monkeypatch, args):
    setup(monkeypatch)
    stale = args.out.parent / "upgrade-stages/components.json"
    stale.parent.mkdir()
    stale.write_text(json.dumps({"artifact": "l15-family-matrix-run", "passed": True}))
    calls = []

    def job(argv, **kwargs):
        calls.append(argv)
        assert not stale.exists()
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(upgrade, "run_job", job)
    assert not upgrade.run(args)
    assert len(calls) == 1  # failed component evidence prevents model startup
    assert not json.loads(args.out.read_text())["ready"]


def test_report_for_other_upstream_commit_is_rejected(monkeypatch, args):
    setup(monkeypatch)

    def job(argv, **kwargs):
        report = args.out.parent / "upgrade-stages/components.json"
        report.write_text(json.dumps({
            "artifact": "l15-family-matrix-run", "passed": True,
            "runtime": {"upstream": {"commit": "wrong"}},
        }))
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(upgrade, "run_job", job)
    assert not upgrade.run(args)
    assert not json.loads(args.out.read_text())["ready"]


@pytest.mark.parametrize("candidate,accepted", [("other", False), ("new", True)])
def test_reviewed_drift_must_match_exact_changes(monkeypatch, args, candidate, accepted):
    setup(monkeypatch, static_pass=False)
    args.audit_only = True
    args.reviewed_drift = args.out.parent / "reviewed.json"
    args.reviewed_drift.write_text(json.dumps({
        "artifact": "l15-upstream-diff", "baseline": "old", "candidate": candidate,
        "changes": [{"kind": "signature", "target": "example"}],
    }))
    monkeypatch.setattr(upgrade, "run_job", lambda *a, **k: pytest.fail("unexpected GPU job"))
    if accepted:
        assert upgrade.run(args)
        assert not json.loads(args.out.read_text())["ready"]
    else:
        with pytest.raises(ValueError, match="does not match"):
            upgrade.run(args)
