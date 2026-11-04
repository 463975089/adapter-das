#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Versioned L1.5 upstream-test manifest tooling.

The tool never edits the upstream tree.  It validates adapter-owned JSON,
checks candidate paths, asks pytest for exact nodeids, detects collection drift,
and can execute selected manifest classes in observe or blocking mode.

Examples:
  python tools/l15_manifest.py validate tests/manifests/l15_v0522.json
  python tools/l15_manifest.py inventory --up-tree "$UP_TREE" \
      --manifest tests/manifests/l15_v0522.json --out inventory.json
  python tools/l15_manifest.py collect --up-tree "$UP_TREE" \
      --manifest tests/manifests/l15_v0522.json --out-dir results/l15
  python tools/l15_manifest.py verify --manifest tests/manifests/l15_v0522.json \
      --collect results/l15/collect.json --mode observe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
CLASSIFICATIONS = {
    "candidate-observe",
    "core-green",
    "contract-green",
    "diagnostic-known-fail",
    "platform-skip",
    "unclassified",
}
ROUTE_EVIDENCE = {"required", "optional", "not_collected"}
RUN_CLASSES = CLASSIFICATIONS - {"platform-skip"}
NODEID_RE = re.compile(r"^(?P<nodeid>(?:test/)?registered/.+::.+)$")
_UPSTREAM_ABS_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|/)(?:[^\s'\"(),:]+[\\/])+"
    r"(?P<relative>(?:python[\\/]sglang|test[\\/]registered)[\\/]"
    r"[^\s'\"(),:]+)"
)


def _public_diagnostic(value: object) -> str:
    """Remove validation-host prefixes while preserving upstream-relative paths."""
    text = str(value)
    return _UPSTREAM_ABS_PATH_RE.sub(
        lambda match: match.group("relative").replace("\\", "/"), text
    )[:500]


class ManifestError(ValueError):
    """A manifest or collected artifact violates the L1.5 contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"JSON root must be an object: {path}")
    return value


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


def _require(mapping: Mapping[str, Any], key: str, expected: type, where: str) -> Any:
    if key not in mapping:
        raise ManifestError(f"{where}: missing required key {key!r}")
    value = mapping[key]
    if not isinstance(value, expected):
        raise ManifestError(
            f"{where}.{key}: expected {expected.__name__}, got {type(value).__name__}"
        )
    return value


def _safe_relative_path(value: str, where: str, *, require_test: bool = True) -> str:
    if not value or "\\" in value:
        raise ManifestError(f"{where}: path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ManifestError(f"{where}: absolute paths and '..' are forbidden: {value!r}")
    if require_test and not value.startswith("test/"):
        raise ManifestError(f"{where}: upstream test path must start with 'test/': {value!r}")
    return value.rstrip("/")


def _string_list(value: Any, where: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ManifestError(f"{where}: expected {'non-empty ' if nonempty else ''}list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ManifestError(f"{where}: entries must be non-empty strings")
    if len(value) != len(set(value)):
        raise ManifestError(f"{where}: duplicate entries are forbidden")
    return list(value)


def _classification(record: Mapping[str, Any], defaults: Mapping[str, Any]) -> str:
    value = record.get("classification", defaults.get("classification", "candidate-observe"))
    if value not in CLASSIFICATIONS:
        raise ManifestError(f"invalid classification {value!r}")
    return str(value)


def _route_evidence(record: Mapping[str, Any], defaults: Mapping[str, Any]) -> str:
    value = record.get("route_evidence", defaults.get("route_evidence", "required"))
    if value not in ROUTE_EVIDENCE:
        raise ManifestError(f"invalid route_evidence {value!r}")
    return str(value)


def _expanded_tests(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return exact-test records from expanded v1 or compact grouped storage."""
    tests = document.get("tests")
    if isinstance(tests, list) and tests:
        return [dict(test) for test in tests if isinstance(test, dict)]
    groups = document.get("test_groups", [])
    overrides = document.get("test_overrides", {})
    if not isinstance(groups, list) or not isinstance(overrides, dict):
        return []
    expanded: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        common = {key: value for key, value in group.items() if key != "nodeids"}
        for nodeid in group.get("nodeids", []):
            record = dict(common)
            record["nodeid"] = nodeid
            override = overrides.get(nodeid, {})
            if isinstance(override, dict):
                record.update(override)
            expanded.append(record)
    return expanded


def _compact_tests(
    document: Mapping[str, Any], tests: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Store repeated source metadata once and only exceptional fields per nodeid."""
    result = json.loads(json.dumps(document))
    result.pop("tests", None)
    result.pop("test_groups", None)
    result.pop("test_overrides", None)
    sources = {
        source["id"]: source
        for source in [*result.get("selectors", []), *result.get("candidates", [])]
    }
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    defaults = result.get("defaults", {})
    for test in tests:
        effective = dict(test)
        effective["classification"] = _classification(test, defaults)
        effective["route_evidence"] = _route_evidence(test, defaults)
        grouped.setdefault(str(test["source_id"]), []).append(effective)
    groups: list[dict[str, Any]] = []
    overrides: dict[str, dict[str, Any]] = {}
    inherited = {"source_id", "domain", "logical_ops", "owner"}
    for source_id in sorted(grouped):
        records = sorted(grouped[source_id], key=lambda item: str(item["nodeid"]))
        source = sources[source_id]
        class_counts: dict[tuple[str, str], int] = {}
        for record in records:
            key = (
                record["classification"],
                record["route_evidence"],
            )
            class_counts[key] = class_counts.get(key, 0) + 1
        dominant_class, dominant_evidence = max(
            class_counts, key=lambda key: (class_counts[key], key)
        )
        group = {
            "source_id": source_id,
            "domain": source["domain"],
            "logical_ops": list(source.get("logical_ops") or [source["domain"]]),
            "owner": source["owner"],
            "classification": dominant_class,
            "route_evidence": dominant_evidence,
            "nodeids": [str(record["nodeid"]) for record in records],
        }
        groups.append(group)
        group_defaults = {key: value for key, value in group.items() if key != "nodeids"}
        for record in records:
            override = {
                key: value
                for key, value in record.items()
                if key not in inherited
                and key != "nodeid"
                and group_defaults.get(key) != value
            }
            if override:
                overrides[str(record["nodeid"])] = override
    result["test_groups"] = groups
    if overrides:
        result["test_overrides"] = dict(sorted(overrides.items()))
    return result


def _validate_record_contract(record: Mapping[str, Any], defaults: Mapping[str, Any], where: str) -> None:
    classification = _classification(record, defaults)
    evidence = _route_evidence(record, defaults)
    owner = record.get("owner")
    if not isinstance(owner, str) or not owner:
        raise ManifestError(f"{where}.owner: non-empty owner is required")
    if classification == "core-green" and evidence != "required":
        raise ManifestError(f"{where}: core-green requires route_evidence='required'")
    if classification == "diagnostic-known-fail":
        for key in ("fingerprint", "review_condition", "promotion_condition"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ManifestError(f"{where}.{key}: required for diagnostic-known-fail")
    if classification == "platform-skip":
        for key in ("platform_guard", "skip_reason"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ManifestError(f"{where}.{key}: required for platform-skip")


def validate_manifest(document: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ManifestError(
                f"schema_version must be {SCHEMA_VERSION}, got {document.get('schema_version')!r}"
            )
        _require(document, "manifest_id", str, "manifest")
        upstream = _require(document, "upstream", dict, "manifest")
        _require(upstream, "project", str, "manifest.upstream")
        _require(upstream, "version", str, "manifest.upstream")
        binding_files = _require(
            upstream, "binding_files", dict, "manifest.upstream"
        )
        if not binding_files:
            raise ManifestError(
                "manifest.upstream.binding_files: at least one binding is required"
            )
        for path, digest in binding_files.items():
            _safe_relative_path(
                path,
                f"manifest.upstream.binding_files[{path!r}]",
                require_test=False,
            )
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ManifestError(
                    f"manifest.upstream.binding_files[{path!r}]: lowercase SHA-256 required"
                )
        markers = _string_list(
            _require(upstream, "tree_markers", list, "manifest.upstream"),
            "manifest.upstream.tree_markers",
        )
        for index, marker in enumerate(markers):
            if PurePosixPath(marker).is_absolute() or ".." in PurePosixPath(marker).parts:
                raise ManifestError(
                    f"manifest.upstream.tree_markers[{index}]: unsafe relative path"
                )

        defaults = _require(document, "defaults", dict, "manifest")
        for boolean_key in (
            "needs_server",
            "needs_model_weights",
            "external_network",
        ):
            if not isinstance(defaults.get(boolean_key), bool):
                raise ManifestError(f"manifest.defaults.{boolean_key}: boolean required")
        if type(defaults.get("device_count")) is not int or defaults["device_count"] < 0:
            raise ManifestError("manifest.defaults.device_count: non-negative integer required")
        _classification(defaults, {})
        _route_evidence(defaults, {})

        seen_ids: set[str] = set()
        seen_candidate_paths: set[str] = set()
        selectors = _require(document, "selectors", list, "manifest")
        candidates = _require(document, "candidates", list, "manifest")
        raw_tests = document.get("tests")
        raw_groups = document.get("test_groups")
        if raw_tests is not None and raw_groups is not None:
            raise ManifestError("manifest: use either tests or test_groups, not both")
        if raw_tests is None and raw_groups is None:
            raise ManifestError("manifest: missing exact tests/test_groups")
        if raw_tests is not None and not isinstance(raw_tests, list):
            raise ManifestError("manifest.tests: list required")
        if raw_groups is not None:
            if not isinstance(raw_groups, list):
                raise ManifestError("manifest.test_groups: list required")
            for index, group in enumerate(raw_groups):
                where = f"manifest.test_groups[{index}]"
                if not isinstance(group, dict):
                    raise ManifestError(f"{where}: object required")
                _require(group, "source_id", str, where)
                _require(group, "domain", str, where)
                _require(group, "owner", str, where)
                _string_list(_require(group, "logical_ops", list, where), f"{where}.logical_ops")
                _string_list(_require(group, "nodeids", list, where), f"{where}.nodeids")
        tests = _expanded_tests(document)
        overrides = document.get("test_overrides", {})
        if not isinstance(overrides, dict):
            raise ManifestError("manifest.test_overrides: object required")

        for index, selector in enumerate(selectors):
            where = f"manifest.selectors[{index}]"
            if not isinstance(selector, dict):
                raise ManifestError(f"{where}: object required")
            record_id = _require(selector, "id", str, where)
            if record_id in seen_ids:
                raise ManifestError(f"{where}.id: duplicate id {record_id!r}")
            seen_ids.add(record_id)
            _require(selector, "domain", str, where)
            paths = _string_list(_require(selector, "paths", list, where), f"{where}.paths")
            for path_index, path in enumerate(paths):
                _safe_relative_path(path, f"{where}.paths[{path_index}]")
            if "exclude_keywords" in selector:
                _string_list(selector["exclude_keywords"], f"{where}.exclude_keywords", nonempty=False)
            _validate_record_contract(selector, defaults, where)

        for index, candidate in enumerate(candidates):
            where = f"manifest.candidates[{index}]"
            if not isinstance(candidate, dict):
                raise ManifestError(f"{where}: object required")
            record_id = _require(candidate, "id", str, where)
            if record_id in seen_ids:
                raise ManifestError(f"{where}.id: duplicate id {record_id!r}")
            seen_ids.add(record_id)
            _require(candidate, "batch", str, where)
            _require(candidate, "domain", str, where)
            path = _safe_relative_path(_require(candidate, "path", str, where), f"{where}.path")
            if path in seen_candidate_paths:
                raise ManifestError(f"{where}.path: duplicate candidate path {path!r}")
            seen_candidate_paths.add(path)
            _string_list(
                _require(candidate, "logical_ops", list, where), f"{where}.logical_ops"
            )
            _string_list(
                _require(candidate, "expected_backends", list, where),
                f"{where}.expected_backends",
            )
            _validate_record_contract(candidate, defaults, where)

        seen_nodeids: set[str] = set()
        for index, test in enumerate(tests):
            where = f"manifest.tests[{index}]"
            if not isinstance(test, dict):
                raise ManifestError(f"{where}: object required")
            nodeid = _require(test, "nodeid", str, where)
            path_part = nodeid.split("::", 1)[0]
            _safe_relative_path(path_part, f"{where}.nodeid")
            if "::" not in nodeid:
                raise ManifestError(f"{where}.nodeid: exact pytest nodeid required")
            if nodeid in seen_nodeids:
                raise ManifestError(f"{where}.nodeid: duplicate nodeid {nodeid!r}")
            seen_nodeids.add(nodeid)
            _require(test, "domain", str, where)
            _string_list(_require(test, "logical_ops", list, where), f"{where}.logical_ops")
            _validate_record_contract(test, defaults, where)
        unknown_overrides = sorted(set(overrides) - seen_nodeids)
        if unknown_overrides:
            raise ManifestError(
                "manifest.test_overrides: unknown nodeids: "
                + ", ".join(unknown_overrides[:3])
            )
    except ManifestError as exc:
        errors.append(str(exc))
    return errors


def validate_artifact(document: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ManifestError(
                f"artifact schema_version must be {SCHEMA_VERSION}, "
                f"got {document.get('schema_version')!r}"
            )
        artifact = _require(document, "artifact", str, "artifact")
        _require(document, "manifest_id", str, "artifact")
        upstream = _require(document, "upstream", dict, "artifact")
        _require(upstream, "project", str, "artifact.upstream")
        _require(upstream, "version", str, "artifact.upstream")
        _require(upstream, "marker_hash", str, "artifact.upstream")
        if artifact == "l15-inventory":
            _require(document, "entries", list, "artifact")
            _require(document, "summary", dict, "artifact")
        elif artifact == "l15-collect":
            _string_list(document.get("nodeids"), "artifact.nodeids", nonempty=False)
            _require(document, "missing_paths", list, "artifact")
            _require(document, "collection_errors", list, "artifact")
            if type(document.get("returncode")) is not int:
                raise ManifestError("artifact.returncode: integer required")
            if not isinstance(document.get("timed_out"), bool):
                raise ManifestError("artifact.timed_out: boolean required")
        elif artifact in {"l15-verify", "l15-run"}:
            if not isinstance(document.get("passed"), bool):
                raise ManifestError("artifact.passed: boolean required")
        elif artifact == "l15-classification":
            _require(document, "outcome_summary", dict, "artifact")
            _require(document, "classification_summary", dict, "artifact")
            if type(document.get("promoted_contract_green_count")) is not int:
                raise ManifestError(
                    "artifact.promoted_contract_green_count: integer required"
                )
            if type(document.get("unobserved_count")) is not int:
                raise ManifestError("artifact.unobserved_count: integer required")
        else:
            raise ManifestError(f"unsupported artifact kind {artifact!r}")
    except ManifestError as exc:
        errors.append(str(exc))
    return errors


def load_manifest(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    errors = validate_manifest(document)
    if errors:
        raise ManifestError("; ".join(errors))
    return document


def _contained_upstream_path(up_tree: Path, relative: str) -> Path:
    root = up_tree.resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ManifestError(
            f"upstream path escapes tree through symlink: {relative!r}"
        ) from exc
    return target


def _tree_fingerprint(up_tree: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    upstream = manifest["upstream"]
    markers = upstream["tree_markers"]
    missing = [
        marker
        for marker in markers
        if not _contained_upstream_path(up_tree, marker).exists()
    ]
    if missing:
        raise ManifestError(
            f"upstream tree {up_tree} is missing markers: {', '.join(missing)}"
        )
    binding_hashes: dict[str, str] = {}
    for path, expected in upstream["binding_files"].items():
        target = _contained_upstream_path(up_tree, path)
        if not target.is_file():
            raise ManifestError(f"upstream binding file is missing: {path}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise ManifestError(
                f"upstream binding mismatch for {path}: expected {expected}, got {actual}"
            )
        binding_hashes[path] = actual
    git_head = None
    try:
        result = subprocess.run(
            ["git", "-C", str(up_tree), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            git_head = result.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    marker_hash = hashlib.sha256()
    for marker in sorted(markers):
        target = _contained_upstream_path(up_tree, marker)
        marker_hash.update(marker.encode("utf-8"))
        marker_hash.update(b"\0")
        marker_hash.update((b"d" if target.is_dir() else b"f"))
    return {
        "project": upstream["project"],
        "version": upstream["version"],
        "git_head": git_head,
        "binding_files": binding_hashes,
        "marker_hash": marker_hash.hexdigest(),
    }


def _effective(record: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "classification",
        "device_count",
        "needs_server",
        "needs_model_weights",
        "external_network",
        "route_evidence",
    )
    result = dict(record)
    for key in keys:
        if key not in result and key in defaults:
            result[key] = defaults[key]
    return result


def build_inventory(manifest: Mapping[str, Any], up_tree: Path) -> dict[str, Any]:
    fingerprint = _tree_fingerprint(up_tree, manifest)
    defaults = manifest["defaults"]
    entries: list[dict[str, Any]] = []
    for selector in manifest["selectors"]:
        effective = _effective(selector, defaults)
        path_states = [
            {
                "path": path,
                "exists": _contained_upstream_path(up_tree, path).exists(),
            }
            for path in selector["paths"]
        ]
        effective["kind"] = "selector"
        effective["path_states"] = path_states
        effective["ready_for_collect"] = any(item["exists"] for item in path_states)
        entries.append(effective)
    for candidate in manifest["candidates"]:
        effective = _effective(candidate, defaults)
        effective["kind"] = "candidate"
        effective["exists"] = _contained_upstream_path(
            up_tree, candidate["path"]
        ).is_file()
        effective["ready_for_collect"] = effective["exists"]
        entries.append(effective)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "l15-inventory",
        "manifest_id": manifest["manifest_id"],
        "upstream": fingerprint,
        "entries": entries,
        "summary": {
            "total": len(entries),
            "ready": sum(bool(entry["ready_for_collect"]) for entry in entries),
            "missing": sum(not bool(entry["ready_for_collect"]) for entry in entries),
        },
    }


def _validate_batches(
    manifest: Mapping[str, Any], batches: set[str] | None
) -> None:
    if batches is None:
        return
    allowed = {"current"} | {
        candidate["batch"] for candidate in manifest["candidates"]
    }
    unknown = sorted(batches - allowed)
    if unknown:
        raise ManifestError(
            f"unknown L1.5 batches: {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}"
        )


def _selection_groups(
    manifest: Mapping[str, Any], up_tree: Path, batches: set[str] | None
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    _validate_batches(manifest, batches)
    groups: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for selector in manifest["selectors"]:
        if batches and "current" not in batches:
            continue
        paths: list[str] = []
        for path in selector["paths"]:
            if _contained_upstream_path(up_tree, path).exists():
                paths.append(path)
            else:
                missing.append({"id": selector["id"], "path": path})
        if paths:
            groups.append(
                {
                    "id": selector["id"],
                    "paths": paths,
                    "exclude_keywords": list(selector.get("exclude_keywords", [])),
                }
            )
    for candidate in manifest["candidates"]:
        if batches and candidate["batch"] not in batches:
            continue
        path = candidate["path"]
        if _contained_upstream_path(up_tree, path).is_file():
            groups.append(
                {
                    "id": candidate["id"],
                    "paths": [path],
                    "exclude_keywords": [],
                }
            )
        else:
            missing.append({"id": candidate["id"], "path": path})
    return groups, missing


def _parse_nodeids(output: str) -> list[str]:
    nodeids: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        match = NODEID_RE.match(line)
        if match:
            nodeid = match.group("nodeid")
            if nodeid.startswith("registered/"):
                nodeid = "test/" + nodeid
            nodeids.append(nodeid)
    return sorted(set(nodeids))


def _fingerprint_errors(output: str, limit: int = 20) -> list[str]:
    fingerprints: list[str] = []
    for raw in output.splitlines():
        line = re.sub(r"\s+", " ", raw.strip())
        if not line:
            continue
        lowered = line.lower()
        if (
            "error collecting" in lowered
            or lowered.startswith(("importerror:", "modulenotfounderror:", "syntaxerror:"))
            or "no tests collected" in lowered
        ):
            if line not in fingerprints:
                fingerprints.append(line[:500])
        if len(fingerprints) >= limit:
            break
    return fingerprints


def _pytest_env(up_tree: Path, adapter_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    parts = [str(up_tree / "python"), str(adapter_root)]
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def collect_manifest(
    manifest: Mapping[str, Any],
    up_tree: Path,
    adapter_root: Path,
    batches: set[str] | None,
    timeout: int,
) -> dict[str, Any]:
    upstream_fingerprint = _tree_fingerprint(up_tree, manifest)
    groups, missing = _selection_groups(manifest, up_tree, batches)
    if not groups:
        raise ManifestError("no selected test paths exist in the upstream tree")
    commands: list[list[str]] = []
    all_nodeids: set[str] = set()
    all_errors: list[str] = []
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    group_results: list[dict[str, Any]] = []
    returncodes: list[int] = []
    timed_out = False
    for group in groups:
        command = [sys.executable, "-m", "pytest", "--collect-only", "-q", *group["paths"]]
        keywords = group["exclude_keywords"]
        if keywords:
            command.extend(["-k", " and ".join(f"not {word}" for word in keywords)])
        commands.append(command)
        try:
            completed = subprocess.run(
                command,
                cwd=up_tree,
                env=_pytest_env(up_tree, adapter_root),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            returncode = 124
        combined = stdout + ("\n" if stdout and stderr else "") + stderr
        group_nodeids = _parse_nodeids(combined)
        group_errors = _fingerprint_errors(combined)
        all_nodeids.update(group_nodeids)
        for error_fingerprint in group_errors:
            if error_fingerprint not in all_errors:
                all_errors.append(error_fingerprint)
        group_results.append(
            {
                "id": group["id"],
                "paths": list(group["paths"]),
                "returncode": returncode,
                "timed_out": returncode == 124,
                "nodeid_count": len(group_nodeids),
                "collection_errors": group_errors,
            }
        )
        stdout_parts.append(f"=== {group['id']} ===\n{stdout}")
        stderr_parts.append(f"=== {group['id']} ===\n{stderr}")
        returncodes.append(returncode)
    effective_returncode = 0 if all(code == 0 for code in returncodes) else next(
        code for code in returncodes if code != 0
    )
    paths = [path for group in groups for path in group["paths"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "l15-collect",
        "manifest_id": manifest["manifest_id"],
        "upstream": upstream_fingerprint,
        "selection": {
            "batches": sorted(batches) if batches else ["all"],
            "paths": paths,
            "groups": groups,
        },
        "missing_paths": missing,
        "commands": commands,
        "group_results": group_results,
        "returncodes": returncodes,
        "returncode": effective_returncode,
        "timed_out": timed_out,
        "nodeids": sorted(all_nodeids),
        "collection_errors": all_errors,
        "summary": {
            "collected": len(all_nodeids),
            "missing_paths": len(missing),
            "collection_errors": len(all_errors),
        },
        "stdout": "\n".join(stdout_parts),
        "stderr": "\n".join(stderr_parts),
    }


def verify_collection(
    manifest: Mapping[str, Any], collected: Mapping[str, Any], mode: str
) -> tuple[dict[str, Any], bool]:
    if collected.get("artifact") != "l15-collect":
        raise ManifestError("collect artifact must have artifact='l15-collect'")
    if collected.get("manifest_id") != manifest["manifest_id"]:
        raise ManifestError("collect artifact manifest_id does not match manifest")
    collected_upstream = collected.get("upstream")
    if not isinstance(collected_upstream, dict):
        raise ManifestError("collect artifact has no upstream binding")
    expected_upstream = manifest["upstream"]
    if (
        collected_upstream.get("project") != expected_upstream["project"]
        or collected_upstream.get("version") != expected_upstream["version"]
    ):
        raise ManifestError("collect artifact upstream binding does not match manifest")
    actual = set(_string_list(collected.get("nodeids"), "collect.nodeids", nonempty=False))
    expected_records = _expanded_tests(manifest)
    expected = {record["nodeid"] for record in expected_records}
    expected_class = {record["nodeid"]: _classification(record, manifest["defaults"]) for record in expected_records}
    added = sorted(actual - expected)
    missing = sorted(expected - actual)
    missing_core = sorted(nodeid for nodeid in missing if expected_class[nodeid] == "core-green")
    missing_contract = sorted(
        nodeid for nodeid in missing if expected_class[nodeid] == "contract-green"
    )
    collection_errors = list(collected.get("collection_errors") or [])
    group_results = list(collected.get("group_results") or [])
    diagnostics = {
        candidate["id"]: candidate
        for candidate in manifest["candidates"]
        if _classification(candidate, manifest["defaults"])
        == "diagnostic-known-fail"
    }
    matched_diagnostics: list[dict[str, Any]] = []
    unmatched_collection_errors: list[str] = []
    groups_by_id = {
        group.get("id"): group for group in group_results if isinstance(group, dict)
    }
    for diagnostic_id, candidate in diagnostics.items():
        group = groups_by_id.get(diagnostic_id)
        if group is None:
            continue
        fingerprints = list(group.get("collection_errors") or [])
        expected_fingerprint = candidate["fingerprint"].lower()
        matches = [
            value
            for value in fingerprints
            if expected_fingerprint in value.lower()
        ]
        if matches:
            matched_diagnostics.append(
                {"id": diagnostic_id, "fingerprints": matches}
            )
    matched_values = {
        value
        for item in matched_diagnostics
        for value in item["fingerprints"]
    }
    unmatched_collection_errors = [
        value for value in collection_errors if value not in matched_values
    ]
    path_missing = list(collected.get("missing_paths") or [])
    returncode = collected.get("returncode")
    baseline_present = bool(expected)
    hard_failures: list[str] = []
    warnings: list[str] = []
    diagnostic_only_failure = (
        returncode != 0
        and bool(matched_diagnostics)
        and not unmatched_collection_errors
        and not bool(collected.get("timed_out"))
    )
    if returncode != 0 and not diagnostic_only_failure:
        hard_failures.append(f"pytest collect returned {returncode}")
    if collected.get("timed_out"):
        hard_failures.append("pytest collect timed out")
    if unmatched_collection_errors:
        hard_failures.append("unmatched collection errors were observed")
    if missing_core:
        hard_failures.append("core-green nodeids disappeared")
    if mode == "blocking" and missing_contract:
        hard_failures.append("contract-green nodeids disappeared")
    if path_missing:
        warnings.append("manifest paths are missing in the bound upstream tree")
    if not baseline_present:
        warnings.append("manifest has no exact nodeid baseline; drift comparison is observe-only")
        if mode == "blocking":
            hard_failures.append("blocking mode requires a non-empty exact nodeid baseline")
    elif added:
        warnings.append("new nodeids are not classified")
    infrastructure_failed = (
        (returncode != 0 and not diagnostic_only_failure)
        or bool(collected.get("timed_out"))
        or bool(unmatched_collection_errors)
    )
    passed = not hard_failures if mode == "blocking" else not infrastructure_failed
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "l15-verify",
        "manifest_id": manifest["manifest_id"],
        "upstream": dict(collected_upstream),
        "mode": mode,
        "baseline_present": baseline_present,
        "passed": passed,
        "added": added,
        "missing": missing,
        "missing_core": missing_core,
        "missing_contract": missing_contract,
        "collection_errors": collection_errors,
        "matched_diagnostics": matched_diagnostics,
        "unmatched_collection_errors": unmatched_collection_errors,
        "missing_paths": path_missing,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "summary": {
            "expected": len(expected),
            "actual": len(actual),
            "added": len(added),
            "missing": len(missing),
        },
    }
    return report, passed


def build_candidate_baseline(
    manifest: Mapping[str, Any], collected: Mapping[str, Any]
) -> dict[str, Any]:
    if collected.get("artifact") != "l15-collect":
        raise ManifestError("baseline input must be an l15-collect artifact")
    if collected.get("manifest_id") != manifest["manifest_id"]:
        raise ManifestError("collect artifact manifest_id does not match manifest")
    candidates = {candidate["path"]: candidate for candidate in manifest["candidates"]}
    selectors = list(manifest["selectors"])
    tests: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for nodeid in _string_list(
        collected.get("nodeids"), "collect.nodeids", nonempty=False
    ):
        path = nodeid.split("::", 1)[0]
        source: Mapping[str, Any] | None = candidates.get(path)
        if source is None:
            matches = [
                selector
                for selector in selectors
                if any(
                    path == prefix or path.startswith(prefix.rstrip("/") + "/")
                    for prefix in selector["paths"]
                )
            ]
            if len(matches) == 1:
                source = matches[0]
        if source is None:
            unmatched.append(nodeid)
            continue
        tests.append(
            {
                "nodeid": nodeid,
                "source_id": source["id"],
                "domain": source["domain"],
                "logical_ops": list(source.get("logical_ops") or [source["domain"]]),
                "classification": "candidate-observe",
                "route_evidence": "not_collected",
                "owner": source["owner"],
            }
        )
    if unmatched:
        sample = ", ".join(unmatched[:3])
        raise ManifestError(
            f"{len(unmatched)} collected nodeids do not map to a selector/candidate: {sample}"
        )
    result = _compact_tests(manifest, sorted(tests, key=lambda item: item["nodeid"]))
    errors = validate_manifest(result)
    if errors:
        raise ManifestError("generated baseline is invalid: " + "; ".join(errors))
    return result


def classify_outcomes(
    manifest: Mapping[str, Any], outcomes_documents: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_nodeid: dict[str, Mapping[str, Any]] = {}
    duplicate_nodeids: set[str] = set()
    for document in outcomes_documents:
        if document.get("artifact") != "l15-pytest-outcomes":
            raise ManifestError("classification input must be l15-pytest-outcomes")
        for record in document.get("tests", []):
            nodeid = record.get("nodeid")
            if not isinstance(nodeid, str):
                raise ManifestError("outcome record has no string nodeid")
            if nodeid in by_nodeid:
                duplicate_nodeids.add(nodeid)
            by_nodeid[nodeid] = record
    if duplicate_nodeids:
        raise ManifestError(
            f"duplicate outcome nodeids across inputs: {sorted(duplicate_nodeids)[:3]}"
        )
    working_tests = _expanded_tests(manifest)
    summary: dict[str, int] = {}
    promoted: list[str] = []
    unobserved: list[str] = []
    for test in working_tests:
        nodeid = test["nodeid"]
        record = by_nodeid.get(nodeid)
        if record is None:
            unobserved.append(nodeid)
            summary["unobserved"] = summary.get("unobserved", 0) + 1
            continue
        outcome = str(record.get("outcome", "not_collected"))
        summary[outcome] = summary.get(outcome, 0) + 1
        if outcome == "passed":
            test["classification"] = "contract-green"
            test["route_evidence"] = "not_collected"
            promoted.append(nodeid)
        elif outcome == "skipped":
            phase = record.get("phases", {}).get("setup", {})
            test["classification"] = "platform-skip"
            test["platform_guard"] = "pytest-runtime-skip"
            test["skip_reason"] = _public_diagnostic(
                phase.get("reason", "pytest skip")
            )
            test["route_evidence"] = "not_collected"
        elif outcome in {"failed", "error", "xfail", "xpass"}:
            phases = record.get("phases", {})
            phase = phases.get("call") or phases.get("setup") or phases.get("teardown") or {}
            fingerprint = phase.get("fingerprint") or phase.get("reason") or outcome
            test["classification"] = "diagnostic-known-fail"
            test["route_evidence"] = "not_collected"
            test["fingerprint"] = _public_diagnostic(fingerprint)
            test["review_condition"] = "the recorded outcome or fingerprint changes"
            test["promotion_condition"] = "stable green execution with route evidence review"
    result = _compact_tests(manifest, working_tests)
    errors = validate_manifest(result)
    if errors:
        raise ManifestError("classified manifest is invalid: " + "; ".join(errors))
    classification_summary: dict[str, int] = {}
    for test in working_tests:
        classification = str(test.get("classification", "unclassified"))
        classification_summary[classification] = (
            classification_summary.get(classification, 0) + 1
        )
    upstream = manifest["upstream"]
    marker_hash = hashlib.sha256()
    for marker in sorted(upstream["tree_markers"]):
        marker_hash.update(marker.encode("utf-8"))
        marker_hash.update(b"\0d")
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "l15-classification",
        "manifest_id": manifest["manifest_id"],
        "upstream": {
            "project": upstream["project"],
            "version": upstream["version"],
            "git_head": None,
            "binding_files": dict(upstream["binding_files"]),
            "marker_hash": marker_hash.hexdigest(),
        },
        "outcome_summary": dict(sorted(summary.items())),
        "classification_summary": dict(sorted(classification_summary.items())),
        "promoted_contract_green_count": len(promoted),
        "unobserved_count": len(unobserved),
        "note": "contract-green does not count as kernel route coverage; core-green requires separate route evidence",
    }
    return result, report


def _selected_exact_tests(
    manifest: Mapping[str, Any],
    classifications: set[str],
    source_ids: set[str] | None = None,
) -> list[str]:
    defaults = manifest["defaults"]
    return [
        test["nodeid"]
        for test in _expanded_tests(manifest)
        if _classification(test, defaults) in classifications
        and (source_ids is None or test.get("source_id") in source_ids)
    ]


def _outcome_summary(outcomes: Mapping[str, Any] | None) -> dict[str, int]:
    summary: dict[str, int] = {}
    if not outcomes:
        return {"not_collected": 0}
    for record in outcomes.get("tests", []):
        outcome = str(record.get("outcome", "not_collected"))
        summary[outcome] = summary.get(outcome, 0) + 1
    return dict(sorted(summary.items()))


def _pytest_process_groups(nodeids: Sequence[str]) -> list[list[str]]:
    """Partition exact nodeids so each pytest process sees unique basenames.

    All nodeids from one file stay together.  Files are considered in sorted
    path order and the Nth path sharing a basename is assigned to group N.
    This is deterministic and avoids pytest's import-file-mismatch behavior.
    """
    by_path: dict[str, list[str]] = {}
    for nodeid in nodeids:
        by_path.setdefault(nodeid.split("::", 1)[0], []).append(nodeid)
    groups: list[list[str]] = []
    basename_counts: dict[str, int] = {}
    for path in sorted(by_path):
        basename = PurePosixPath(path).name
        group_index = basename_counts.get(basename, 0)
        basename_counts[basename] = group_index + 1
        while len(groups) <= group_index:
            groups.append([])
        groups[group_index].extend(by_path[path])
    return groups


def _merge_pytest_outcomes(
    documents: Sequence[Mapping[str, Any]], returncode: int
) -> dict[str, Any]:
    tests: dict[str, Mapping[str, Any]] = {}
    for document in documents:
        if document.get("artifact") != "l15-pytest-outcomes":
            raise ManifestError("run outcome group must be l15-pytest-outcomes")
        for record in document.get("tests", []):
            nodeid = record.get("nodeid")
            if not isinstance(nodeid, str):
                raise ManifestError("run outcome record has no string nodeid")
            if nodeid in tests:
                raise ManifestError(f"duplicate nodeid across run groups: {nodeid}")
            tests[nodeid] = record
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "l15-pytest-outcomes",
        "exitstatus": returncode,
        "tests": [tests[nodeid] for nodeid in sorted(tests)],
    }


def run_manifest(
    manifest: Mapping[str, Any],
    up_tree: Path,
    adapter_root: Path,
    classifications: set[str],
    source_ids: set[str] | None,
    timeout: int,
    outcome_path: Path,
) -> tuple[dict[str, Any], bool]:
    upstream_fingerprint = _tree_fingerprint(up_tree, manifest)
    invalid = classifications - RUN_CLASSES
    if invalid:
        raise ManifestError(f"unsupported run classifications: {sorted(invalid)}")
    nodeids = _selected_exact_tests(manifest, classifications, source_ids)
    if not nodeids:
        raise ManifestError(
            "manifest has no exact nodeids for the selected classes; run collect and classify first"
        )
    process_groups = _pytest_process_groups(nodeids)
    commands: list[list[str]] = []
    group_results: list[dict[str, Any]] = []
    outcome_documents: list[Mapping[str, Any]] = []
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    run_env = _pytest_env(up_tree, adapter_root)
    returncodes: list[int] = []
    any_timed_out = False
    for index, group_nodeids in enumerate(process_groups):
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=line",
            "-p",
            "tools.l15_pytest_report",
            *group_nodeids,
        ]
        commands.append(command)
        group_outcome_path = outcome_path.with_name(
            f".{outcome_path.name}.group-{index + 1}.json"
        )
        try:
            group_outcome_path.unlink()
        except FileNotFoundError:
            pass
        group_env = dict(run_env)
        group_env["L15_PYTEST_REPORT"] = str(group_outcome_path)
        try:
            completed = subprocess.run(
                command,
                cwd=up_tree,
                env=group_env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            timed_out = False
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            returncode = 124
        outcomes_present = False
        if group_outcome_path.is_file():
            try:
                document = _read_json(group_outcome_path)
                if document.get("artifact") == "l15-pytest-outcomes":
                    outcome_documents.append(document)
                    outcomes_present = True
            except ManifestError:
                outcomes_present = False
            finally:
                try:
                    group_outcome_path.unlink()
                except FileNotFoundError:
                    pass
        returncodes.append(returncode)
        any_timed_out = any_timed_out or timed_out
        stdout_parts.append(f"=== pytest group {index + 1}/{len(process_groups)} ===\n{stdout}")
        stderr_parts.append(f"=== pytest group {index + 1}/{len(process_groups)} ===\n{stderr}")
        group_results.append(
            {
                "index": index + 1,
                "nodeids": group_nodeids,
                "command": command,
                "returncode": returncode,
                "timed_out": timed_out,
                "outcomes_present": outcomes_present,
            }
        )
    returncode = 124 if any_timed_out else next(
        (code for code in returncodes if code != 0), 0
    )
    outcomes = _merge_pytest_outcomes(outcome_documents, returncode)
    _write_json(outcome_path, outcomes)
    all_outcomes_present = all(group["outcomes_present"] for group in group_results)
    passed = returncode == 0 and not any_timed_out and all_outcomes_present
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "l15-run",
        "manifest_id": manifest["manifest_id"],
        "upstream": upstream_fingerprint,
        "classifications": sorted(classifications),
        "source_ids": sorted(source_ids) if source_ids else ["all"],
        "nodeids": nodeids,
        "command": commands[0],
        "commands": commands,
        "process_groups": group_results,
        "timeout_seconds_per_group": timeout,
        "returncode": returncode,
        "timed_out": any_timed_out,
        "passed": passed,
        "outcomes_file": str(outcome_path),
        "outcome_summary": _outcome_summary(outcomes),
        "stdout": "\n".join(stdout_parts),
        "stderr": "\n".join(stderr_parts),
    }
    return artifact, passed


def _manifest_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, type=Path)


def _up_tree_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--up-tree", required=True, type=Path)


def _csv_set(value: str) -> set[str]:
    result = {item.strip() for item in value.split(",") if item.strip()}
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate manifest schema")
    validate.add_argument("manifest", type=Path)

    validate_artifact_parser = subparsers.add_parser(
        "validate-artifact", help="validate inventory/collect/verify/run artifact"
    )
    validate_artifact_parser.add_argument("artifact_file", type=Path)

    inventory = subparsers.add_parser("inventory", help="check candidate paths")
    _manifest_arg(inventory)
    _up_tree_arg(inventory)
    inventory.add_argument("--out", required=True, type=Path)

    collect = subparsers.add_parser("collect", help="collect exact upstream nodeids")
    _manifest_arg(collect)
    _up_tree_arg(collect)
    collect.add_argument("--out-dir", required=True, type=Path)
    collect.add_argument("--batches", type=_csv_set)
    collect.add_argument("--timeout", type=int, default=900)

    verify = subparsers.add_parser("verify", help="compare collect artifact with baseline")
    _manifest_arg(verify)
    verify.add_argument("--collect", required=True, type=Path)
    verify.add_argument("--mode", choices=("observe", "blocking"), default="observe")
    verify.add_argument("--out", type=Path)

    baseline = subparsers.add_parser(
        "baseline", help="generate candidate-observe exact-nodeid baseline"
    )
    _manifest_arg(baseline)
    baseline.add_argument("--collect", required=True, type=Path)
    baseline.add_argument("--out", required=True, type=Path)

    classify = subparsers.add_parser(
        "classify", help="apply structured pytest outcomes to exact nodeids"
    )
    _manifest_arg(classify)
    classify.add_argument("--outcomes", required=True, nargs="+", type=Path)
    classify.add_argument("--out-manifest", required=True, type=Path)
    classify.add_argument("--out-report", required=True, type=Path)

    run = subparsers.add_parser("run", help="run exact classified nodeids")
    _manifest_arg(run)
    _up_tree_arg(run)
    run.add_argument("--out-dir", required=True, type=Path)
    run.add_argument("--classifications", type=_csv_set, required=True)
    run.add_argument("--source-ids", type=_csv_set)
    run.add_argument("--timeout", type=int, default=1800)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adapter_root = Path(__file__).resolve().parents[1]
    try:
        if args.command == "validate":
            manifest = load_manifest(args.manifest.resolve())
            print(
                f"L1.5 manifest OK: {manifest['manifest_id']} "
                f"({len(manifest['selectors'])} selectors, "
                f"{len(manifest['candidates'])} candidates, "
                f"{len(_expanded_tests(manifest))} exact nodeids)"
            )
            return 0
        if args.command == "validate-artifact":
            artifact = _read_json(args.artifact_file.resolve())
            errors = validate_artifact(artifact)
            if errors:
                raise ManifestError("; ".join(errors))
            print(
                f"L1.5 artifact OK: {artifact['artifact']} "
                f"({artifact['manifest_id']})"
            )
            return 0

        manifest = load_manifest(args.manifest.resolve())
        if args.command == "baseline":
            collected = _read_json(args.collect.resolve())
            baseline_manifest = build_candidate_baseline(manifest, collected)
            _write_json(args.out.resolve(), baseline_manifest)
            print(
                f"L1.5 baseline: {len(_expanded_tests(baseline_manifest))} exact nodeids -> "
                f"{args.out.resolve()}"
            )
            return 0
        if args.command == "classify":
            outcomes = [_read_json(path.resolve()) for path in args.outcomes]
            classified, report = classify_outcomes(manifest, outcomes)
            _write_json(args.out_manifest.resolve(), classified)
            _write_json(args.out_report.resolve(), report)
            print(
                f"L1.5 classify: {report['outcome_summary']} -> "
                f"{args.out_manifest.resolve()}"
            )
            return 0
        if args.command == "inventory":
            artifact = build_inventory(manifest, args.up_tree.resolve())
            _write_json(args.out.resolve(), artifact)
            summary = artifact["summary"]
            print(f"L1.5 inventory: {summary['ready']} ready / {summary['missing']} missing")
            return 0
        if args.command == "collect":
            artifact = collect_manifest(
                manifest,
                args.up_tree.resolve(),
                adapter_root,
                args.batches,
                args.timeout,
            )
            output = args.out_dir.resolve() / "collect.json"
            _write_json(output, artifact)
            print(
                f"L1.5 collect: {artifact['summary']['collected']} nodeids, "
                f"{artifact['summary']['collection_errors']} errors, "
                f"{artifact['summary']['missing_paths']} missing paths -> {output}"
            )
            return 0 if (
                artifact["returncode"] == 0
                and not artifact["timed_out"]
                and not artifact["collection_errors"]
            ) else 1
        if args.command == "verify":
            collected = _read_json(args.collect.resolve())
            artifact, passed = verify_collection(manifest, collected, args.mode)
            if args.out:
                _write_json(args.out.resolve(), artifact)
            print(
                f"L1.5 verify ({args.mode}): {'PASS' if passed else 'FAIL'}; "
                f"actual={artifact['summary']['actual']} "
                f"added={artifact['summary']['added']} missing={artifact['summary']['missing']}"
            )
            for warning in artifact["warnings"]:
                print(f"WARN: {warning}")
            for failure in artifact["hard_failures"]:
                print(f"FAIL: {failure}")
            return 0 if passed else 1
        if args.command == "run":
            out_dir = args.out_dir.resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            outcome_path = out_dir / "outcomes.json"
            artifact, passed = run_manifest(
                manifest,
                args.up_tree.resolve(),
                adapter_root,
                args.classifications,
                args.source_ids,
                args.timeout,
                outcome_path,
            )
            output = out_dir / "run.json"
            _write_json(output, artifact)
            print(
                f"L1.5 run: {'PASS' if passed else 'FAIL'}; "
                f"{len(artifact['nodeids'])} nodeids -> {output}"
            )
            return 0 if passed else 1
    except ManifestError as exc:
        print(f"L1.5 manifest error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
