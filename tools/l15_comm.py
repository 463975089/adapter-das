#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run the adapter-owned L1.5-comm two-rank communication manifest.

The runner accepts no arbitrary command fragments.  Every argv comes from a
validated repository-relative case in tests/manifests/l15_comm_v1.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
KINDS = {"pytest", "torchrun-script", "ct-int8-deepep"}
CT_PROBE = "tools/ct_w4a8_deepep_probe.py"
CLASSIFICATIONS = {
    "candidate-observe",
    "core-green",
    "diagnostic-known-fail",
    "platform-skip",
}


class CommManifestError(ValueError):
    pass


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _safe_path(value: object, where: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CommManifestError(f"{where}: non-empty POSIX relative path required")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value.startswith("tests/"):
        raise CommManifestError(f"{where}: unsafe repository test path {value!r}")
    return value


def validate_manifest(document: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        if document.get("schema_version") != SCHEMA_VERSION:
            raise CommManifestError(f"schema_version must be {SCHEMA_VERSION}")
        if document.get("layer") != "L1.5-comm":
            raise CommManifestError("layer must be 'L1.5-comm'")
        if not isinstance(document.get("manifest_id"), str):
            raise CommManifestError("manifest_id string required")
        defaults = document.get("defaults")
        if not isinstance(defaults, dict) or defaults.get("device_count") != 2:
            raise CommManifestError("defaults.device_count must be exactly 2")
        cases = document.get("cases")
        if not isinstance(cases, list) or not cases:
            raise CommManifestError("at least one communication case is required")
        ids: set[str] = set()
        for index, case in enumerate(cases):
            where = f"cases[{index}]"
            if not isinstance(case, dict):
                raise CommManifestError(f"{where}: object required")
            case_id = case.get("id")
            if not isinstance(case_id, str) or not case_id or case_id in ids:
                raise CommManifestError(f"{where}.id: unique non-empty string required")
            ids.add(case_id)
            kind = case.get("kind")
            if kind not in KINDS:
                raise CommManifestError(f"{where}.kind: unsupported {kind!r}")
            if kind == "ct-int8-deepep":
                path = case.get("path")
                if path != CT_PROBE:
                    raise CommManifestError(f"{where}.path: CT probe path must be {CT_PROBE}")
                if type(case.get("recv_hook")) is not bool:
                    raise CommManifestError(f"{where}.recv_hook: boolean required")
                limit = case.get("swiglu_limit")
                if limit is not None and (
                    type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0
                ):
                    raise CommManifestError(f"{where}.swiglu_limit: positive finite number required")
            else:
                path = _safe_path(case.get("path"), f"{where}.path")
            if kind == "pytest":
                nodeid = case.get("nodeid")
                if not isinstance(nodeid, str) or not nodeid.startswith(path + "::"):
                    raise CommManifestError(f"{where}.nodeid: exact nodeid under path required")
            for key in ("primitive", "backend", "owner"):
                if not isinstance(case.get(key), str) or not case[key]:
                    raise CommManifestError(f"{where}.{key}: non-empty string required")
            timeout = case.get("timeout")
            if type(timeout) is not int or timeout <= 0:
                raise CommManifestError(f"{where}.timeout: positive integer required")
            classification = case.get("classification", defaults.get("classification"))
            if classification not in CLASSIFICATIONS:
                raise CommManifestError(f"{where}.classification: unsupported {classification!r}")
    except CommManifestError as exc:
        errors.append(str(exc))
    return errors


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CommManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise CommManifestError("manifest root must be an object")
    errors = validate_manifest(document)
    if errors:
        raise CommManifestError("; ".join(errors))
    return document


def _contained(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    target = (resolved_root / relative).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError as exc:
        raise CommManifestError(f"test path escapes repository: {relative}") from exc
    if not target.is_file():
        raise CommManifestError(f"test path does not exist: {relative}")
    return target


def _case_command(
    case: Mapping[str, Any], python: str,
    *, up_tree: Path | None = None, rank_out: Path | None = None,
) -> list[str]:
    if case["kind"] == "pytest":
        return [python, "-m", "pytest", "-q", "--tb=line", case["nodeid"]]
    command = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
    ]
    if case["kind"] == "ct-int8-deepep":
        if up_tree is None or rank_out is None:
            raise CommManifestError("CT communication requires --up-tree and a rank report directory")
        command += [
            "-m", "tools.ct_w4a8_deepep_probe",
            "--up-tree", str(up_tree), "--out", str(rank_out),
            "--community-dispatcher",
        ]
        if case["recv_hook"]:
            command.append("--recv-hook")
        if case.get("swiglu_limit") is not None:
            command += ["--swiglu-limit", str(case["swiglu_limit"])]
    else:
        command.append(case["path"])
    return command


def _ct_evidence(case, rank_out, commit):
    """Require both ranks and each distinct phase, not just a launcher exit."""
    phases = ["eager"] + [f"graph-{phase}" for phase in range(4)]
    for rank in range(2):
        try:
            report = json.loads((rank_out / f"rank{rank}.json").read_text(encoding="utf-8"))
            checks = report["checks"]
            valid = (
                type(report["rank"]) is int and report["rank"] == rank
                and report["communication_passed"] is True
                and report["model_acceptance"] is False
                and report["community_dispatcher"] is True
                and report["recv_hook"] is case["recv_hook"]
                and report["swiglu_limit"] == case.get("swiglu_limit")
                and report["community_commit"] == commit
                and [row["phase"] for row in checks] == phases
                and all(type(row["max_abs"]) in (int, float) and row["max_abs"] == 0
                        for row in checks)
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CommManifestError(f"missing or invalid CT rank {rank} evidence") from exc
        if not valid:
            raise CommManifestError(f"incomplete or mismatched CT rank {rank} evidence")
    return {"rank_phase_checks": 10, "model_acceptance": False}


def _run_ct_case(case, root, out_dir, up_tree, command):
    try:
        from tools.l15_runtime import audit_source, run_job, test_env
    except ModuleNotFoundError:
        from l15_runtime import audit_source, run_job, test_env

    before = audit_source(up_tree)
    _write_json(out_dir / f"{case['id']}-source-before.json", before)
    if not before["passed"]:
        raise CommManifestError("CT communication requires pristine community source")
    rank_out = out_dir / case["id"] / "ranks"
    if rank_out.exists():
        raise CommManifestError("CT rank evidence directory already exists; use a fresh output directory")
    env = test_env(up_tree, root)
    env.update(HCU_W4A8_MOE="int8", SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="8")
    log = out_dir / f"{case['id']}.log"
    job = run_job(command, cwd=root, env=env, timeout=case["timeout"], log=log)
    after = audit_source(up_tree, before["commit"])
    _write_json(out_dir / f"{case['id']}-source-after.json", after)
    evidence = {"source_before": before, "source_after": after}
    error = None
    try:
        if not after["passed"]:
            raise CommManifestError("community source changed during CT communication")
        if job["returncode"] == 0:
            evidence.update(_ct_evidence(case, rank_out, before["commit"]))
    except CommManifestError as exc:
        error = str(exc)
    return job, log.read_text(encoding="utf-8", errors="replace"), evidence, error


def run_manifest(
    manifest: Mapping[str, Any],
    root: Path,
    out_dir: Path,
    mode: str,
    case_ids: set[str] | None = None,
    up_tree: Path | None = None,
) -> tuple[dict[str, Any], bool]:
    available = {case["id"] for case in manifest["cases"]}
    if case_ids:
        unknown = sorted(case_ids - available)
        if unknown:
            raise CommManifestError(f"unknown communication cases: {', '.join(unknown)}")
    selected = [case for case in manifest["cases"] if not case_ids or case["id"] in case_ids]
    if any(case["kind"] == "ct-int8-deepep" for case in selected) and up_tree is None:
        raise CommManifestError("CT communication requires --up-tree")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    results: list[dict[str, Any]] = []
    for index, case in enumerate(selected):
        _contained(root, case["path"])
        command = _case_command(
            case, sys.executable, up_tree=up_tree,
            rank_out=out_dir / case["id"] / "ranks",
        )
        case_env = env.copy()
        case_env["MASTER_PORT"] = str(29820 + index)
        evidence, evidence_error = {}, None
        try:
            if case["kind"] == "ct-int8-deepep":
                job, stdout, evidence, evidence_error = _run_ct_case(
                    case, root, out_dir, up_tree, command)
                returncode, timed_out, stderr = job["returncode"], job["timed_out"], ""
            else:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    env=case_env,
                    capture_output=True,
                    text=True,
                    timeout=case["timeout"],
                    check=False,
                )
                returncode = completed.returncode
                timed_out = False
                stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            timed_out = True
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        outcome = "passed" if returncode == 0 and evidence_error is None else "failed"
        classification = case.get("classification", manifest["defaults"]["classification"])
        if case["kind"] == "pytest" and (
            returncode == 5 or (returncode == 0 and " skipped" in stdout[-500:])
        ):
            outcome = "skipped"
        result = {
            "id": case["id"],
            "kind": case["kind"],
            "primitive": case["primitive"],
            "backend": case["backend"],
            "device_count": 2,
            "classification": classification,
            "command": command,
            "returncode": returncode,
            "timed_out": timed_out,
            "outcome": outcome,
            "stdout_tail": stdout[-8000:],
            "stderr_tail": stderr[-4000:],
        }
        if case["kind"] == "ct-int8-deepep":
            result.update(evidence=evidence, evidence_error=evidence_error)
        results.append(result)
        _write_json(out_dir / f"{case['id']}.json", result)
    hard_failures = [
        result["id"]
        for result in results
        if result["timed_out"]
        or (
            result["outcome"] == "failed"
            and result["classification"] not in {"diagnostic-known-fail"}
        )
        or (
            mode == "blocking"
            and result["outcome"] != "passed"
            and result["classification"] == "core-green"
        )
    ]
    passed = not hard_failures if mode == "blocking" else not any(
        result["timed_out"] for result in results
    )
    summary: dict[str, int] = {}
    for result in results:
        summary[result["outcome"]] = summary.get(result["outcome"], 0) + 1
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "l15-comm-run",
        "manifest_id": manifest["manifest_id"],
        "mode": mode,
        "device_count": 2,
        "passed": passed,
        "summary": dict(sorted(summary.items())),
        "hard_failures": hard_failures,
        "results": results,
    }
    _write_json(out_dir / "summary.json", artifact)
    return artifact, passed


def _csv_set(value: str) -> set[str]:
    result = {item.strip() for item in value.split(",") if item.strip()}
    if not result:
        raise argparse.ArgumentTypeError("at least one case id is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--root", required=True, type=Path)
    run.add_argument("--out-dir", required=True, type=Path)
    run.add_argument("--mode", choices=("observe", "blocking"), default="observe")
    run.add_argument("--cases", type=_csv_set)
    run.add_argument("--up-tree", type=Path, help="Pristine community tree required by CT DeepEP cases")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            manifest = load_manifest(args.manifest.resolve())
            print(f"L1.5-comm manifest OK: {manifest['manifest_id']} ({len(manifest['cases'])} cases)")
            return 0
        manifest = load_manifest(args.manifest.resolve())
        artifact, passed = run_manifest(
            manifest,
            args.root.resolve(),
            args.out_dir.resolve(),
            args.mode,
            args.cases,
            args.up_tree.resolve() if args.up_tree is not None else None,
        )
        print(
            f"L1.5-comm ({args.mode}): {'PASS' if passed else 'FAIL'} "
            f"{artifact['summary']} -> {args.out_dir.resolve()}"
        )
        return 0 if passed else 1
    except CommManifestError as exc:
        print(f"L1.5-comm error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
