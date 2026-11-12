# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the L1.6 manifest and runner."""

import json
from pathlib import Path

import pytest

from tools import l16_model_readiness as readiness


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests/manifests/l16_model_readiness_v1.json"


def test_default_l16_manifest_is_valid_and_has_unique_cases():
    document = readiness.load_manifest()
    assert document["layer"] == "L1.6"
    assert len(document["cases"]) == 6
    assert len({case["id"] for case in document["cases"]}) == 6
    assert {case["family"] for case in document["cases"]} == {
        "dense", "moe", "mla", "mamba", "hybrid", "checkpoint",
    }


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(layer="L1.5"), "layer"),
        (lambda d: d["cases"].append(dict(d["cases"][0])), "duplicate case"),
        (lambda d: d["cases"][0].update(kind="checkpoint"), "checkpoint"),
        (lambda d: d["defaults"].update(modes=["eager", "eager"]), "modes"),
    ],
)
def test_manifest_rejects_invalid_structure(mutate, message):
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutate(document)
    with pytest.raises(readiness.ReadinessManifestError, match=message):
        readiness.validate_manifest(document)


def test_case_selection_is_deterministic_and_rejects_unknown():
    document = readiness.load_manifest()
    selected = readiness._selected_cases(document, "dummy-mamba,dummy-dense")
    assert [case["id"] for case in selected] == ["dummy-dense", "dummy-mamba"]
    with pytest.raises(readiness.ReadinessManifestError, match="unknown"):
        readiness._selected_cases(document, "missing")


@pytest.mark.parametrize("fault", ["clean", "stale", "timeout", "wrong-family",
                                  "wrong-source", "wrong-host", "missing-job",
                                  "source-changed", "exception", "malformed-source", "worker-config"])
def test_runner_requires_fresh_bound_evidence_and_preserves_failure(tmp_path, monkeypatch, fault):
    args = readiness.build_parser().parse_args([
        "run", "--up-tree", str(tmp_path), "--cases", "dummy-dense",
        "--out", str(tmp_path / "result.json"),
    ])
    document = readiness.load_manifest()
    defaults = document["defaults"]
    work = tmp_path / "result-jobs"
    work.mkdir()
    report_path = work / "dummy-dense.json"
    report_path.write_text('{"passed":true,"complete":true}')
    audit_count = []

    def audit(*a):
        audit_count.append(a)
        return {"passed": fault != "source-changed" or len(audit_count) == 1,
                "commit": "bound-source"}

    def job(argv, **kwargs):
        assert not report_path.exists(), "old report must be removed before launching"
        assert kwargs["env"]["HCU_ADAPTER_HOST"] == "none"
        if fault == "exception":
            raise OSError("launcher failed")
        if fault != "stale":
            evidence = {
                "artifact": "l15-model-smoke", "passed": True, "complete": True,
                "families": ["dense"], "modes": defaults["modes"],
                "generation_profile": "smoke", "dtype": "bfloat16",
                "tp_size": 1, "mem_fraction_static": .25,
                "runtime": {"host": "adapter", "upstream": {"commit": "bound-source"},
                            "source_integrity": {"passed": True}},
                "jobs": [{"passed": True, "returncode": 0, "timed_out": False,
                          "family": "dense", "mode": mode,
                          "source_integrity_after": {"passed": True},
                          "evidence": {"artifact": "l15-model-worker", "passed": True,
                                       "family": "dense", "mode": mode,
                                       "host": "adapter", "weight_evidence": "dummy",
                                       "dtype": "bfloat16", "tp_size": 1,
                                       "generation_profile": "smoke", "mem_fraction_static": .25}}
                         for mode in defaults["modes"]],
            }
            if fault == "malformed-source":
                evidence["runtime"]["upstream"] = None
            if fault == "worker-config":
                evidence["jobs"][0]["evidence"]["tp_size"] = True
            if fault == "wrong-family":
                evidence["families"] = ["moe"]
            if fault == "wrong-source":
                evidence["runtime"]["upstream"]["commit"] = "other"
            if fault == "wrong-host":
                evidence["runtime"]["host"] = "adapter+plugin"
            if fault == "missing-job":
                evidence["jobs"].pop()
            report_path.write_text(json.dumps(evidence))
        return {"returncode": 124 if fault == "timeout" else 0,
                "timed_out": fault == "timeout"}

    monkeypatch.setattr(readiness, "audit_source", audit)
    monkeypatch.setattr(readiness, "run_job", job)
    assert readiness.run(args) is (fault == "clean")
    report = json.loads(args.out.read_text())
    assert report["passed"] is (fault == "clean")
    assert len(report["jobs"]) == 1


def test_missing_checkpoint_refused_before_any_job(tmp_path, monkeypatch):
    args = readiness.build_parser().parse_args(["run", "--up-tree", str(tmp_path)])
    monkeypatch.setattr(readiness, "run_job", lambda *a, **k: pytest.fail("job started"))
    monkeypatch.setattr(readiness, "audit_source", lambda *a: pytest.fail("probe started"))
    with pytest.raises(readiness.ReadinessManifestError, match="checkpoint"):
        readiness.run(args)
