#!/usr/bin/env python3
"""Validate and run the no-weight L1.5 Dense/MLA/MoE/Quant/Mamba matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from typing import Any, Mapping

SCHEMA_VERSION = 1
FAMILIES = {"dense", "mla", "moe", "quantization", "mamba"}
KINDS = {"upstream-pytest", "adapter-pytest", "adapter-script"}


class MatrixError(ValueError):
    pass


def _safe_path(value: object, prefix: str, where: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise MatrixError(f"{where}: non-empty POSIX path required")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value.startswith(prefix):
        raise MatrixError(f"{where}: unsafe path {value!r}")
    return value


def validate(document: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        if document.get("schema_version") != SCHEMA_VERSION:
            raise MatrixError(f"schema_version must be {SCHEMA_VERSION}")
        if document.get("layer") != "L1.5-family-matrix":
            raise MatrixError("layer must be 'L1.5-family-matrix'")
        if not isinstance(document.get("manifest_id"), str) or not document["manifest_id"]:
            raise MatrixError("manifest_id: non-empty string required")
        defaults = document.get("defaults")
        if (
            not isinstance(defaults, dict)
            or type(defaults.get("device_count")) is not int
            or defaults["device_count"] != 1
        ):
            raise MatrixError("defaults.device_count must be exactly 1")
        timeout = defaults.get("timeout")
        if type(timeout) is not int or timeout <= 0:
            raise MatrixError("defaults.timeout: positive integer required")
        for key in ("needs_server", "needs_model_weights", "external_network"):
            if defaults.get(key) is not False:
                raise MatrixError(f"defaults.{key} must be false")
        families = document.get("families")
        if not isinstance(families, list):
            raise MatrixError("families: list required")
        for fi, family in enumerate(families):
            if not isinstance(family, dict):
                raise MatrixError(f"families[{fi}]: object required")
            if not isinstance(family.get("id"), str):
                raise MatrixError(f"families[{fi}].id: string required")
        if len(families) != len(FAMILIES) or {f["id"] for f in families} != FAMILIES:
            raise MatrixError("families must contain exactly dense, mla, moe, quantization, mamba")
        case_ids: set[str] = set()
        for fi, family in enumerate(families):
            where = f"families[{fi}]"
            cases = family.get("cases")
            if not isinstance(family.get("intent"), str) or not family["intent"]:
                raise MatrixError(f"{where}.intent: non-empty string required")
            if not isinstance(cases, list) or not cases:
                raise MatrixError(f"{where}.cases: non-empty list required")
            for ci, case in enumerate(cases):
                cwhere = f"{where}.cases[{ci}]"
                if not isinstance(case, dict):
                    raise MatrixError(f"{cwhere}: object required")
                case_id = case.get("id")
                if not isinstance(case_id, str) or not case_id or case_id in case_ids:
                    raise MatrixError(f"{cwhere}.id: unique non-empty string required")
                case_ids.add(case_id)
                kind = case.get("kind")
                if kind not in KINDS:
                    raise MatrixError(f"{cwhere}.kind: unsupported {kind!r}")
                if kind == "upstream-pytest":
                    nodeid = _safe_path(case.get("nodeid", ""), "test/", f"{cwhere}.nodeid")
                    if "::" not in nodeid:
                        raise MatrixError(f"{cwhere}.nodeid: exact nodeid required")
                elif kind == "adapter-pytest":
                    nodeid = _safe_path(case.get("nodeid", ""), "tests/", f"{cwhere}.nodeid")
                    if "::" not in nodeid:
                        raise MatrixError(f"{cwhere}.nodeid: exact nodeid required")
                else:
                    _safe_path(case.get("path", ""), "tests/", f"{cwhere}.path")
    except MatrixError as exc:
        errors.append(str(exc))
    return errors


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MatrixError("manifest root must be an object")
    errors = validate(value)
    if errors:
        raise MatrixError("; ".join(errors))
    return value


def _contained(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise MatrixError(f"path escapes root: {relative}") from exc
    if not target.is_file():
        raise MatrixError(f"missing test path: {relative}")
    return target


def _env(up_tree: Path, adapter_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    parts = [str(adapter_root), str(up_tree / "python")]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run(document: Mapping[str, Any], up_tree: Path, adapter_root: Path, selected: set[str], out: Path) -> bool:
    env = _env(up_tree, adapter_root)
    timeout = int(document["defaults"]["timeout"])
    results: list[dict[str, Any]] = []
    all_passed = True
    for family in document["families"]:
        if family["id"] not in selected:
            continue
        for case in family["cases"]:
            kind = case["kind"]
            if kind == "upstream-pytest":
                nodeid = case["nodeid"]
                _contained(up_tree, nodeid.split("::", 1)[0])
                argv = [sys.executable, "-m", "pytest", "-q", "--tb=line", nodeid]
                cwd = up_tree
            elif kind == "adapter-pytest":
                nodeid = case["nodeid"]
                _contained(adapter_root, nodeid.split("::", 1)[0])
                argv = [sys.executable, "-m", "pytest", "-q", "--tb=line", nodeid]
                cwd = adapter_root
            else:
                script = case["path"]
                _contained(adapter_root, script)
                argv = [sys.executable, script]
                cwd = adapter_root
            try:
                completed = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False)
                returncode, timed_out = completed.returncode, False
                stdout, stderr = completed.stdout, completed.stderr
            except subprocess.TimeoutExpired as exc:
                returncode, timed_out = 124, True
                stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            passed = returncode == 0 and not timed_out
            all_passed = all_passed and passed
            results.append({"family": family["id"], "case_id": case["id"], "kind": kind, "argv": argv, "returncode": returncode, "timed_out": timed_out, "passed": passed, "stdout": stdout, "stderr": stderr})
            print(f"L1.5 matrix: {family['id']}/{case['id']} {'PASS' if passed else 'FAIL'}")
    artifact = {"schema_version": 1, "artifact": "l15-family-matrix-run", "manifest_id": document["manifest_id"], "families": sorted(selected), "passed": all_passed, "results": results}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{out.name}.", dir=str(out.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(artifact, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(tmp, out)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument("--manifest", type=Path, default=Path("tests/manifests/l15_family_matrix_v1.json"))
    parser.add_argument("--up-tree", type=Path)
    parser.add_argument("--adapter-root", type=Path, default=Path.cwd())
    parser.add_argument("--families", default=",".join(sorted(FAMILIES)))
    parser.add_argument("--out", type=Path, default=Path("results/l15-family-matrix/run.json"))
    args = parser.parse_args()
    try:
        document = load(args.manifest.resolve())
        if args.command == "validate":
            count = sum(len(f["cases"]) for f in document["families"])
            print(f"L1.5 family matrix OK: {len(document['families'])} families, {count} cases")
            return 0
        if args.up_tree is None:
            raise MatrixError("run requires --up-tree")
        selected = {item for item in args.families.split(",") if item}
        unknown = selected - FAMILIES
        if unknown:
            raise MatrixError(f"unknown families: {sorted(unknown)}")
        return 0 if run(document, args.up_tree.resolve(), args.adapter_root.resolve(), selected, args.out.resolve()) else 1
    except (OSError, json.JSONDecodeError, MatrixError) as exc:
        print(f"L1.5 family matrix ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
