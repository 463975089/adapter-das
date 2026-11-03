"""Compare community source bytes with an immutable Git commit.

Unlike git status, this reads every tracked file and does not trust the index,
mtime cache, assume-unchanged or skip-worktree flags. Ignored added source
files are also reported because they can change Python/JIT import behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat
import subprocess

_SOURCE_SUFFIXES = {".py", ".pyi", ".cu", ".cuh", ".cpp", ".cc", ".c",
                    ".h", ".hpp", ".hip"}


def _git(tree, *args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={tree}", "-C", str(tree), *args],
        stderr=subprocess.PIPE, timeout=120,
    )


def audit(tree):
    tree = Path(tree).resolve()
    commit = _git(tree, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    rows = _git(tree, "ls-tree", "-rz", commit).split(b"\0")
    changed, checked, tracked_names = [], 0, set()
    for row in rows:
        if not row:
            continue
        metadata, raw_name = row.split(b"\t", 1)
        mode, kind, expected = metadata.decode().split()
        name = os.fsdecode(raw_name)
        tracked_names.add(name)
        path = tree / name
        if kind != "blob":
            changed.append({"path": name, "reason": "unverified-submodule"})
            continue
        checked += 1
        try:
            path.parent.resolve().relative_to(tree)
            before = path.lstat()
            digest = hashlib.new("sha1" if len(expected) == 40 else "sha256")
            if mode == "120000":
                if not stat.S_ISLNK(before.st_mode):
                    raise ValueError("tracked symlink replaced")
                data = os.fsencode(os.readlink(path))
                digest.update(f"blob {len(data)}\0".encode())
                digest.update(data)
            else:
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError("tracked file is not a regular file")
                digest.update(f"blob {before.st_size}\0".encode())
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            after = path.lstat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("file changed during integrity audit")
            if digest.hexdigest() != expected:
                changed.append({"path": name, "reason": "content",
                                "expected": expected, "actual": digest.hexdigest()})
        except (OSError, ValueError) as exc:
            changed.append({"path": name, "reason": str(exc)})
    # Do not use --exclude-standard: .gitignore must not hide added source.
    # Include index entries too: staging an added source must not hide it.
    candidates = {
        os.fsdecode(name)
        for name in _git(tree, "ls-files", "--cached", "--others", "-z").split(b"\0")
        if name
    }
    extra = sorted(name for name in candidates - tracked_names
                   if Path(name).suffix.lower() in _SOURCE_SUFFIXES)
    head_after = _git(tree, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    return {
        "artifact": "l15-upstream-source-integrity", "root": str(tree),
        "commit": commit, "head_after": head_after, "tracked_files": checked,
        "changes": changed, "added_sources": extra,
        "passed": bool(checked) and not changed and not extra and head_after == commit,
        "scope": "all tracked file bytes plus untracked/ignored source files",
    }


def main():
    try:
        from tools.l15_runtime import write_json
    except ModuleNotFoundError:
        from l15_runtime import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--up-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("results/l15-readiness/upstream-source.json"))
    args = parser.parse_args()
    try:
        report = audit(args.up_tree)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report = {"artifact": "l15-upstream-source-integrity",
                  "passed": False, "error": str(exc)}
    write_json(args.out, report)
    print("upstream source: " + ("PASS" if report["passed"] else "FAIL")
          + f"; {len(report.get('changes', []))} changed/unverified files"
          + f"; {len(report.get('added_sources', []))} added sources")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
