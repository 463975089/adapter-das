#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Audit immutable SGLang revisions without importing either upstream tree."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.l15_runtime import git_stamp, write_json


def git(tree, *args, **kwargs):
    return subprocess.run(
        ["git", "-c", f"safe.directory={tree.resolve()}", "-C", str(tree), *args],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs,
    ).stdout


def snapshot(tree, revision):
    commit = git(tree, "rev-parse", "--verify", revision + "^{commit}").decode().strip()
    rows = git(tree, "ls-tree", "-rz", commit, "--", "python", "test").split(b"\0")
    entries = []
    for row in rows:
        if not row:
            continue
        metadata, path = row.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        name = path.decode()
        if kind == b"blob" and Path(name).suffix in {".py", ".cuh", ".h", ".hpp", ".cpp", ".toml"}:
            entries.append((oid, name))
    if not entries:
        raise ValueError(f"no upstream sources at {revision}")
    output = git(tree, "cat-file", "--batch", input=b"".join(oid + b"\n" for oid, _ in entries))
    stream = io.BytesIO(output)
    files = {}
    for _, name in entries:
        header = stream.readline().split()
        size = int(header[-1])
        files[name] = stream.read(size).decode("utf-8", errors="replace")
        if stream.read(1) != b"\n":
            raise ValueError("invalid git object stream")
    return commit, files


def adapter_contracts(root):
    """Discover literal registered targets; report dynamic sites explicitly."""
    targets, dynamic = set(), []
    for path in sorted((root / "hcu_adapter").rglob("*.py")):
        source = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(source):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name != "register":
                continue
            for keyword in node.keywords:
                if keyword.arg != "target":
                    continue
                if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    if keyword.value.value.startswith("sglang."):
                        targets.add(keyword.value.value)
                else:
                    dynamic.append(f"{path.relative_to(root).as_posix()}:{node.lineno}")
    registry = ast.parse((root / "hcu_adapter/registry.py").read_text(encoding="utf-8"))
    needles = None
    for node in registry.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TEXT_PATCH_ALLOWLIST"
            for target in node.targets
        ):
            needles = sorted(ast.literal_eval(node.value))
    if needles is None:
        raise ValueError("TEXT_PATCH_ALLOWLIST was not found")
    matrix = json.loads((root / "tests/manifests/l15_family_matrix_v1.json").read_text())
    nodeids = [case["nodeid"] for family in matrix["families"] for case in family["cases"]
               if case["kind"] == "upstream-pytest"]
    # Public startup entry plus the computational chains used by the matrix.
    targets.update({
        "sglang.srt.entrypoints.engine.Engine.__init__",
        "sglang.srt.entrypoints.engine.Engine.generate",
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts_impl",
        "sglang.srt.layers.quantization.fp8_utils.apply_fp8_linear",
        "sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update",
    })
    return sorted(targets), needles, nodeids, dynamic


def describe(node, path):
    signature = ast.dump(node.args, include_attributes=False) if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef)
    ) else None
    return {"path": path, "line": node.lineno, "signature": signature,
            "fingerprint": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()}


def find_node(source, names):
    nodes = ast.parse(source).body
    for name in names:
        found = None
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                found = node
                break
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets
            ):
                found = node
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
                found = node
        if found is None:
            return None
        nodes = getattr(found, "body", [])
    return found


def symbol(files, target):
    components = target.split(".")
    for split in range(len(components) - 1, 0, -1):
        module = "python/" + "/".join(components[:split])
        for path in (module + ".py", module + "/__init__.py"):
            if path in files:
                node = find_node(files[path], components[split:])
                return describe(node, path) if node is not None else None
    return None


def compare(baseline, candidate, targets, needles, nodeids):
    changes, baseline_gaps, checked = [], [], []
    for target in targets:
        before, after = symbol(baseline, target), symbol(candidate, target)
        if before is None:
            baseline_gaps.append({"kind": "symbol", "target": target})
            continue
        checked.append({"kind": "symbol", "target": target})
        if after is None:
            changes.append({"kind": "missing-symbol", "target": target, "before": before})
        elif before["signature"] != after["signature"]:
            changes.append({"kind": "signature", "target": target, "before": before, "after": after})
        elif before["fingerprint"] != after["fingerprint"]:
            changes.append({"kind": "source", "target": target, "before": before, "after": after})
    for op, needle in needles:
        locations = {path: text.count(needle) for path, text in baseline.items() if needle in text}
        if not locations:
            baseline_gaps.append({"kind": "needle", "op": op, "needle": needle})
            continue
        checked.append({"kind": "needle", "op": op, "needle": needle, "locations": locations})
        candidate_locations = {path: text.count(needle) for path, text in candidate.items()
                               if needle in text}
        for path in sorted(locations.keys() | candidate_locations.keys()):
            count = locations.get(path, 0)
            actual = candidate.get(path, "").count(needle)
            if actual != count:
                changes.append({"kind": "needle", "op": op, "path": path,
                                "needle": needle, "before_count": count, "after_count": actual})
    for nodeid in nodeids:
        path, *names = nodeid.split("::")
        before = find_node(baseline[path], names) if path in baseline else None
        after = find_node(candidate[path], names) if path in candidate else None
        if before is None:
            baseline_gaps.append({"kind": "nodeid", "target": nodeid})
        else:
            checked.append({"kind": "nodeid", "target": nodeid})
            if after is None:
                changes.append({"kind": "missing-nodeid", "target": nodeid})
            elif ast.dump(before, include_attributes=False) != ast.dump(after, include_attributes=False):
                changes.append({"kind": "test-source", "target": nodeid})
    return {"changes": changes, "baseline_gaps": baseline_gaps, "checked": checked,
            "passed": not changes, "scope": "static-declared-contracts"}


def audit(tree, baseline_ref, candidate_ref, root=ROOT):
    old_commit, baseline = snapshot(tree, baseline_ref)
    new_commit, candidate = snapshot(tree, candidate_ref)
    targets, needles, nodeids, dynamic = adapter_contracts(root)
    result = compare(baseline, candidate, targets, needles, nodeids)
    matrix = json.loads((root / "tests/manifests/l15_family_matrix_v1.json").read_text())
    bindings = json.loads((root / matrix["upstream_manifest"]).read_text())["upstream"]["binding_files"]
    result["binding_files"] = {}
    for path, expected in bindings.items():
        before = hashlib.sha256(baseline[path].encode()).hexdigest() if path in baseline else None
        after = hashlib.sha256(candidate[path].encode()).hexdigest() if path in candidate else None
        result["binding_files"][path] = {"baseline": before, "candidate": after, "manifest": expected}
        if before != after:
            result["changes"].append({"kind": "binding-file", "path": path,
                                      "before": before, "after": after})
    result["passed"] = not result["changes"]
    result.update(
        artifact="l15-upstream-diff", baseline=old_commit, candidate=new_commit,
        same_revision=old_commit == new_commit, runtime_tree=git_stamp(tree),
        adapter=git_stamp(root), dynamic_registration_sites=dynamic,
        limitation="Static literals only; unresolved baseline symbols/reexports and dynamic registrations require runtime gates.",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("up_tree", type=Path)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/l15-readiness/upstream-diff.json"))
    args = parser.parse_args()
    try:
        result = audit(args.up_tree, args.baseline_ref, args.candidate_ref)
        write_json(args.out, result)
        print(f"upstream audit: {len(result['changes'])} changes, "
              f"{len(result['baseline_gaps'])} baseline gaps, "
              f"{len(result['dynamic_registration_sites'])} dynamic registration sites")
        return 0 if result["passed"] else 1
    except (OSError, ValueError, SyntaxError, subprocess.SubprocessError) as exc:
        write_json(args.out, {"artifact": "l15-upstream-diff", "passed": False, "error": str(exc)})
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
