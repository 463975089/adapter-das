# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""L1.6 model launch readiness manifest and runner.

L1.6 is the bridge between component-level L1.5 evidence and full L2 model
validation.  It proves that a selected dummy architecture or real checkpoint
can load the community Engine, generate fresh requests in eager and graph
mode, and preserve the source/checkpoint bindings.

The runner deliberately delegates actual Engine execution to
``tools.l15_model_smoke``.  This keeps one implementation of the runtime
contracts and makes the L1.6 artifact a manifest-level aggregation rather
than a second, subtly different model runner.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys

from tools.l15_runtime import audit_source, run_job, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests/manifests/l16_model_readiness_v1.json"
ALLOWED_KINDS = {"dummy", "checkpoint"}
ALLOWED_FAMILIES = {"dense", "moe", "mla", "mamba", "hybrid", "checkpoint"}
ALLOWED_MODES = {"eager", "graph"}
ALLOWED_HOSTS = {"adapter", "adapter+plugin"}


class ReadinessManifestError(ValueError):
    """Invalid or internally inconsistent L1.6 manifest."""


def load_manifest(path=DEFAULT_MANIFEST):
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadinessManifestError(f"cannot read manifest {path}: {exc}") from exc
    validate_manifest(document)
    return document


def validate_manifest(document):
    if not isinstance(document, dict):
        raise ReadinessManifestError("manifest must be an object")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ReadinessManifestError("schema_version must be 1")
    if document.get("layer") != "L1.6":
        raise ReadinessManifestError("layer must be L1.6")
    if not isinstance(document.get("manifest_id"), str) or not document["manifest_id"]:
        raise ReadinessManifestError("manifest_id must be non-empty")
    defaults = document.get("defaults")
    if not isinstance(defaults, dict):
        raise ReadinessManifestError("defaults must be an object")
    if type(defaults.get("device_count")) is not int or defaults["device_count"] not in (1, 2, 4, 8):
        raise ReadinessManifestError("defaults.device_count must be a positive integer")
    modes = defaults.get("modes")
    if (not isinstance(modes, list) or not modes
            or any(not isinstance(mode, str) for mode in modes)
            or len(set(modes)) != len(modes)
            or set(modes) - ALLOWED_MODES):
        raise ReadinessManifestError("defaults.modes must contain eager/graph without duplicates")
    if defaults.get("profile") != "smoke":
        raise ReadinessManifestError("L1.6 defaults.profile must be smoke")
    if defaults.get("dtype") not in ("bfloat16", "float32"):
        raise ReadinessManifestError("defaults.dtype must be bfloat16 or float32")
    fraction = defaults.get("mem_fraction_static")
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ReadinessManifestError("defaults.mem_fraction_static must be between zero and one")
    if defaults.get("host") not in ALLOWED_HOSTS:
        raise ReadinessManifestError("defaults.host is not supported")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ReadinessManifestError("cases must be a non-empty list")
    ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ReadinessManifestError("each case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", case_id):
            raise ReadinessManifestError("case.id must be non-empty")
        if case_id in ids:
            raise ReadinessManifestError(f"duplicate case id: {case_id}")
        ids.add(case_id)
        kind = case.get("kind")
        family = case.get("family")
        if kind not in ALLOWED_KINDS:
            raise ReadinessManifestError(f"{case_id}: unsupported kind {kind!r}")
        if family not in ALLOWED_FAMILIES:
            raise ReadinessManifestError(f"{case_id}: unsupported family {family!r}")
        if kind == "dummy" and family == "checkpoint":
            raise ReadinessManifestError(f"{case_id}: dummy case cannot use checkpoint family")
        if kind == "checkpoint":
            if family != "checkpoint" or case.get("requires_checkpoint") is not True:
                raise ReadinessManifestError(
                    f"{case_id}: checkpoint case must require the checkpoint family")
        elif case.get("requires_checkpoint", False):
            raise ReadinessManifestError(f"{case_id}: dummy case cannot require checkpoint")
    return True


def _selected_cases(document, requested):
    cases = document["cases"]
    if not requested:
        return cases
    parts = requested.split(",")
    if len(set(parts)) != len(parts):
        raise ReadinessManifestError("duplicate selected cases")
    wanted = set(parts)
    known = {case["id"] for case in cases}
    unknown = sorted(wanted - known)
    if unknown:
        raise ReadinessManifestError(f"unknown L1.6 case(s): {', '.join(unknown)}")
    return [case for case in cases if case["id"] in wanted]


def _run_case(case, defaults, args, work):
    if case["kind"] == "checkpoint" and args.checkpoint is None:
        raise ReadinessManifestError(
            f"{case['id']} requires --checkpoint with config.json and safetensors")
    output = work / f"{case['id']}.json"
    output.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "tools.l15_model_smoke",
        "--up-tree",
        str(args.up_tree.resolve()),
        "--families",
        case["family"],
        "--modes",
        ",".join(defaults["modes"]),
        "--profile",
        defaults["profile"],
        "--dtype",
        defaults["dtype"],
        "--tp-size",
        str(defaults["device_count"]),
        "--mem-fraction-static",
        str(defaults["mem_fraction_static"]),
        "--host",
        defaults["host"],
        "--binding-manifest",
        str(args.binding_manifest.resolve()),
        "--out",
        str(output),
        "--timeout",
        str(args.timeout),
    ]
    if case["kind"] == "checkpoint":
        command.extend(["--checkpoint", str(args.checkpoint.resolve())])
    result = run_job(
        command, cwd=ROOT, env=dict(args.env),
        timeout=args.timeout * len(defaults["modes"]) + 120,
        log=work / f"{case['id']}.log",
    )
    try:
        evidence = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        evidence = {}
    if not isinstance(evidence, dict):
        evidence = {}
    result.update(id=case["id"], kind=case["kind"], family=case["family"], evidence=evidence)
    result["passed"] = (
        result["returncode"] == 0 and not result["timed_out"]
        and valid_evidence(evidence, case, defaults, args)
    )
    return result


def valid_evidence(evidence, case, defaults, args):
    """Malformed child reports fail closed without losing earlier case results."""
    try:
        return _valid_evidence(evidence, case, defaults, args)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _valid_evidence(evidence, case, defaults, args):
    """Bind a successful child report to this exact request and source."""
    if (evidence.get("artifact") != "l15-model-smoke"
            or evidence.get("passed") is not True or evidence.get("complete") is not True
            or evidence.get("families") != [case["family"]]
            or evidence.get("modes") != defaults["modes"]
            or evidence.get("generation_profile") != defaults["profile"]
            or evidence.get("dtype") != defaults["dtype"]
            or type(evidence.get("tp_size")) is not int
            or evidence["tp_size"] != defaults["device_count"]
            or type(evidence.get("mem_fraction_static")) not in (int, float)
            or evidence["mem_fraction_static"] != defaults["mem_fraction_static"]):
        return False
    runtime = evidence.get("runtime", {})
    if (not isinstance(runtime, dict) or runtime.get("host") != defaults["host"]
            or runtime.get("upstream", {}).get("commit") != args.expected_commit
            or runtime.get("source_integrity", {}).get("passed") is not True):
        return False
    jobs = evidence.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(defaults["modes"]):
        return False
    for job, mode in zip(jobs, defaults["modes"]):
        if (not isinstance(job, dict) or job.get("passed") is not True
                or job.get("returncode") != 0 or job.get("timed_out") is not False
                or job.get("family") != case["family"] or job.get("mode") != mode
                or job.get("source_integrity_after", {}).get("passed") is not True):
            return False
        worker = job.get("evidence", {})
        if (not isinstance(worker, dict) or worker.get("artifact") != "l15-model-worker"
                or worker.get("passed") is not True
                or worker.get("family") != case["family"] or worker.get("mode") != mode
                or worker.get("host") != defaults["host"]
                or worker.get("dtype") != defaults["dtype"]
                or worker.get("generation_profile") != defaults["profile"]
                or type(worker.get("tp_size")) is not int
                or worker["tp_size"] != defaults["device_count"]
                or type(worker.get("mem_fraction_static")) not in (int, float)
                or worker["mem_fraction_static"] != defaults["mem_fraction_static"]
                or worker.get("weight_evidence") != (
                    "real-checkpoint" if case["kind"] == "checkpoint" else "dummy")):
            return False
        if case["kind"] == "checkpoint":
            if (worker.get("model") != str(args.checkpoint.resolve())
                    or not worker.get("checkpoint_files_sha256")
                    or job.get("checkpoint_integrity_after", {}).get("passed") is not True):
                return False
    return True



def run(args):
    document = load_manifest(args.manifest)
    selected = _selected_cases(document, args.cases)
    defaults = dict(document["defaults"])
    if args.device_count is not None:
        if type(args.device_count) is not int or args.device_count not in (1, 2, 4, 8):
            raise ReadinessManifestError("device-count must be a positive integer")
        defaults["device_count"] = args.device_count
    if args.host is not None:
        if args.host not in ALLOWED_HOSTS:
            raise ReadinessManifestError(f"unsupported host: {args.host}")
        defaults["host"] = args.host
    if args.up_tree is None or not args.up_tree.is_dir():
        raise ReadinessManifestError("--up-tree must point to an existing SGLang tree")
    if args.binding_manifest is None or not args.binding_manifest.is_file():
        raise ReadinessManifestError("--binding-manifest is required")
    # Validate all selected inputs before the first expensive dummy/Engine job.
    if any(case["kind"] == "checkpoint" for case in selected):
        if (args.checkpoint is None or not (args.checkpoint / "config.json").is_file()
                or not any(args.checkpoint.glob("*.safetensors"))):
            raise ReadinessManifestError("checkpoint selection requires local config and safetensors")
    integrity = audit_source(args.up_tree.resolve())
    if not integrity["passed"]:
        raise ReadinessManifestError("community source is not pristine")
    args.expected_commit = integrity["commit"]
    args.env = dict(os.environ)
    args.env["HCU_ADAPTER_HOST"] = "none"
    work = args.out.resolve().parent / (args.out.stem + "-jobs")
    work.mkdir(parents=True, exist_ok=True)
    jobs = []
    write_json(args.out, {"artifact": "l16-model-readiness", "passed": False,
                          "complete": False, "jobs": []})
    for case in selected:
        try:
            job = _run_case(case, defaults, args, work)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            job = {"id": case["id"], "passed": False, "error": str(exc)}
        integrity = audit_source(args.up_tree.resolve(), args.expected_commit)
        job["source_integrity_after"] = integrity
        job["passed"] = job["passed"] and integrity["passed"]
        jobs.append(job)
        write_json(args.out, {
            "artifact": "l16-model-readiness",
            "manifest_id": document["manifest_id"],
            "manifest": str(Path(args.manifest).resolve()),
            "selected_cases": [item["id"] for item in selected],
            "defaults": defaults,
            "runtime": {"upstream": {"commit": args.expected_commit}},
            "jobs": jobs,
            "complete": len(jobs) == len(selected),
            "passed": len(jobs) == len(selected) and all(item["passed"] for item in jobs),
        })
        print(f"L1.6: {case['id']} {'PASS' if job['passed'] else 'FAIL'}", flush=True)
        if not integrity["passed"]:
            return False
    return all(item["passed"] for item in jobs)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cases", help="comma-separated case ids; default is all cases")
    parser.add_argument("--up-tree", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--binding-manifest", type=Path,
                        default=ROOT / "tests/manifests/l15_v0522.json")
    parser.add_argument("--device-count", type=int)
    parser.add_argument("--host", choices=tuple(sorted(ALLOWED_HOSTS)))
    parser.add_argument("--out", type=Path,
                        default=Path("results/l16-readiness/model-readiness.json"))
    parser.add_argument("--timeout", type=int, default=900)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        document = load_manifest(args.manifest)
        if args.command == "validate":
            print(f"L1.6 manifest: PASS ({len(document['cases'])} cases)")
            return 0
        if args.timeout <= 0:
            raise ReadinessManifestError("timeout must be positive")
        return 0 if run(args) else 1
    except (OSError, ReadinessManifestError, subprocess.SubprocessError) as exc:
        if args.command == "run":
            write_json(args.out, {
                "artifact": "l16-model-readiness",
                "passed": False,
                "complete": False,
                "error": str(exc),
            })
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
