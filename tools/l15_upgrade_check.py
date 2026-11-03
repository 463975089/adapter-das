"""Serial upgrade workflow: static drift, strict components, real model smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ci.upstream_diff import audit
from tools.l15_runtime import git_stamp, run_job, test_env, write_json


def run(args):
    root = Path(__file__).resolve().parents[1]
    tree = args.up_tree.resolve()
    output = args.out.resolve()
    work = output.parent / (output.stem + "-stages")
    work.mkdir(parents=True, exist_ok=True)
    reviewed_path = getattr(args, "reviewed_drift", None)
    # Read before writing stage outputs, even when the caller explicitly
    # points at the previous run's upstream-diff.json.
    reviewed = json.loads(reviewed_path.read_text()) if reviewed_path is not None else None
    static = audit(tree, args.baseline_ref, args.candidate_ref)
    verdict = {"artifact": "l15-upgrade-check", "ready": False,
               "baseline": static["baseline"], "candidate": static["candidate"],
               "same_revision": static["same_revision"], "stages": [],
               "adapter": git_stamp(root), "upstream": git_stamp(tree)}
    write_json(work / "upstream-diff.json", static)
    write_json(output, verdict)
    if not static["passed"]:
        if reviewed is None:
            verdict["error"] = "upstream contracts changed; review drift report before GPU validation"
            write_json(output, verdict)
            return False
        if (
            reviewed.get("artifact") != "l15-upstream-diff"
            or any(reviewed.get(key) != static.get(key)
                   for key in ("baseline", "candidate", "changes"))
        ):
            raise ValueError("reviewed drift report does not match these exact revisions and changes")
        verdict["reviewed_drift"] = str(reviewed_path.resolve())
        write_json(output, verdict)
    if git_stamp(tree)["commit"] != static["candidate"]:
        raise ValueError("runtime tree HEAD must equal the audited candidate commit")
    if args.audit_only:
        verdict["audit_only"] = True
        write_json(output, verdict)
        return True
    if args.checkpoint is None:
        raise ValueError("upgrade readiness requires a local real --checkpoint")
    matrix_manifest = json.loads((root / "tests/manifests/l15_family_matrix_v1.json").read_text())
    env = test_env(tree, root)
    env["HCU_ADAPTER_HOST"] = "none"
    commands = [
        ("components", "l15-family-matrix-run",
         [sys.executable, "-m", "tools.l15_family_matrix", "run", "--up-tree", str(tree)]),
        ("models", "l15-model-smoke",
         [sys.executable, "-m", "tools.l15_model_smoke", "--up-tree", str(tree),
          "--checkpoint", str(args.checkpoint.resolve()),
          "--binding-manifest", str(root / matrix_manifest["upstream_manifest"])]),
    ]
    for name, artifact, argv in commands:
        report = work / (name + ".json")
        report.unlink(missing_ok=True)
        stage = run_job(argv + ["--out", str(report)], cwd=root, env=env,
                        timeout=args.timeout, log=work / (name + ".log"))
        try:
            evidence = json.loads(report.read_text())
        except (OSError, ValueError):
            evidence = {}
        stage.update(name=name, report=str(report))
        stage["passed"] = (
            stage["returncode"] == 0 and not stage["timed_out"]
            and evidence.get("artifact") == artifact and evidence.get("passed") is True
            and evidence.get("runtime", {}).get("upstream", {}).get("commit") == static["candidate"]
        )
        verdict["stages"].append(stage)
        write_json(output, verdict)
        print(f"upgrade stage: {name} {'PASS' if stage['passed'] else 'FAIL'}", flush=True)
        if not stage["passed"]:
            return False
    verdict["ready"] = True
    write_json(output, verdict)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--up-tree", type=Path, required=True)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--reviewed-drift", type=Path,
                        help="explicitly accept an already reviewed, exactly matching static drift report")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--out", type=Path, default=Path("results/l15-readiness/upgrade.json"))
    args = parser.parse_args()
    try:
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        return 0 if run(args) else 1
    except Exception as exc:
        write_json(args.out, {"artifact": "l15-upgrade-check", "ready": False, "error": str(exc)})
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
