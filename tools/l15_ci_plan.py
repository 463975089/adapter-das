# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Assign current upstream green nodes once between exact and family gates."""

import argparse
import copy
from pathlib import Path

from tools import l15_manifest as manifest_tool
from tools.l15_family_matrix import load as load_matrix
from tools.l15_runtime import write_json


ROOT = Path(__file__).resolve().parents[1]
CURRENT = {"attention-core-current", "mechanism-current"}


def plan(baseline, matrix, families):
    known = {f["id"] for f in matrix["families"]}
    if not families or families - known:
        raise ValueError("select valid blocking matrix families")
    family_nodes = [c["nodeid"] for f in matrix["families"] if f["id"] in families
                    for c in f["cases"] if c["kind"] == "upstream-pytest"]
    current = [t for t in manifest_tool._expanded_tests(baseline)
               if t.get("source_id") in CURRENT]
    if {t["source_id"] for t in current} != CURRENT:
        raise ValueError("baseline must contain both current source groups")
    retained, transferred, observation, partial = [], [], [], []
    for test in current:
        classification = manifest_tool._classification(test, baseline["defaults"])
        if classification not in {"contract-green", "core-green"}:
            if classification == "candidate-observe" and test.get("subtest_observation"):
                partial.append(test)
            else:
                observation.append(test)
        elif any(test["nodeid"] == node or test["nodeid"].startswith(node + "[")
                 for node in family_nodes):
            transferred.append(test["nodeid"])
        else:
            retained.append(test)
    if not retained:
        raise ValueError("no residual upstream green tests")
    derived = copy.deepcopy(baseline)
    derived.pop("test_groups", None)
    derived.pop("test_overrides", None)
    derived["tests"] = retained
    derived["manifest_id"] += "-gate-residual"
    errors = manifest_tool.validate_manifest(derived)
    if errors:
        raise ValueError("; ".join(errors))
    return derived, {
        "artifact": "l15-ci-ownership", "families": sorted(families),
        "exact_gate": [t["nodeid"] for t in retained],
        "family_gate": transferred,
        "observation_gate": partial,
        "baseline_observations_not_executed": observation,
        "source_groups": sorted(CURRENT),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "tests/manifests/l15_v0522_classified.json")
    parser.add_argument("--family-manifest", type=Path, default=ROOT / "tests/manifests/l15_family_matrix_v1.json")
    parser.add_argument("--families", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        derived, ownership = plan(manifest_tool.load_manifest(args.baseline),
                                  load_matrix(args.family_manifest), set(args.families.split(",")))
        write_json(args.out_dir / "manifest.json", derived)
        write_json(args.out_dir / "ownership.json", ownership)
        observe = copy.deepcopy(derived)
        observe["manifest_id"] += "-observe"
        observe["tests"] = ownership["observation_gate"]
        write_json(args.out_dir / "observation-manifest.json", observe)
        nodes = [t["nodeid"] for t in ownership["observation_gate"]]
        (args.out_dir / "observe-nodeids.txt").write_text(
            "\n".join(nodes) + ("\n" if nodes else ""), encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"L1.5 ownership planning failed: {exc}\n")
    print(f"L1.5 ownership: {len(ownership['exact_gate'])} exact, "
          f"{len(ownership['family_gate'])} transferred")


if __name__ == "__main__":
    main()
