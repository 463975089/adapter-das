# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Add HYGON H1 file headers to original source files (HYGON 合规整改 v1.7).

Rules (docs/compliance/hygon-oss-v1.7.md):
- Only tracked source files: .py/.sh use ``#``, .h/.cuh use ``//``.
- Skip any file whose first 100 lines already carry Copyright/SPDX (idempotent,
  and keeps third-party notices in vendored files untouched).
- Skip the explicit exclusion list below: third-party vendored bodies are
  registered in THIRD_PARTY_NOTICES.md instead of getting a HYGON header.
- Insertion point: after shebang (``#!``) and after a PEP-263 coding line if
  present; header is followed by one blank line unless the next line is blank.

Usage: python tools/compliance/add_hygon_headers.py [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

HEADER_HASH = [
    "# Copyright (c) 2026 Hygon Information Technology Co., Ltd.",
    "# SPDX-License-Identifier: Apache-2.0",
]
HEADER_SLASH = [
    "// Copyright (c) 2026 Hygon Information Technology Co., Ltd.",
    "// SPDX-License-Identifier: Apache-2.0",
]

COMMENT_STYLE = {".py": HEADER_HASH, ".sh": HEADER_HASH, ".h": HEADER_SLASH, ".cuh": HEADER_SLASH}

# Third-party vendored files (HYGON 合规扫描 §4): keep upstream notices,
# register provenance in THIRD_PARTY_NOTICES.md, never add a HYGON header.
# engram_gate_triton.py / slimquant.py are handled by hand (H2-style).
EXCLUDE = {
    "hcu_adapter/fla_hcu/chunk_delta_h.py",
    "hcu_adapter/fla_hcu/chunk_intra.py",
    "hcu_adapter/fla_hcu/chunk_intra_token_parallel.py",
    "hcu_adapter/fla_hcu/conv1d_triton.py",
    "hcu_adapter/fla_hcu/cumsum.py",
    "hcu_adapter/fla_hcu/index.py",
    "hcu_adapter/fla_hcu/kda.py",
    "hcu_adapter/fla_hcu/l2norm.py",
    "hcu_adapter/fla_hcu/op.py",
    "hcu_adapter/fla_hcu/utils.py",
    "hcu_adapter/fla_hcu/fused_sigmoid_gating_recurrent.py",
    "hcu_adapter/kernels/engram_gate_triton.py",
    "hcu_adapter/slimquant.py",
}

CODING_MARKERS = ("-*- coding:", "# coding:", "coding: utf-8")


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def skip_past_prefix(lines):
    """Return the number of leading lines a header must be inserted after."""
    n = 0
    if lines and lines[0].startswith("#!"):
        n = 1
    if len(lines) > n and any(m in lines[n] for m in CODING_MARKERS):
        n += 1
    return n


def needs_header(rel, text):
    head = "\n".join(text.splitlines()[:100])
    return "Copyright" not in head and "SPDX" not in head


def read_keep_eol(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def write_keep_eol(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def add_header(path, header):
    lines = read_keep_eol(path).splitlines(keepends=True)
    n = skip_past_prefix(lines)
    block = [ln + "\n" for ln in header]
    if not lines or (len(lines) > n and lines[n].strip()):
        block.append("\n")
    write_keep_eol(path, "".join(lines[:n]) + "".join(block) + "".join(lines[n:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed, skipped_existing, excluded = [], [], []
    for rel in tracked_files():
        path = REPO / rel
        suffix = path.suffix
        if suffix not in COMMENT_STYLE:
            continue
        if rel in EXCLUDE:
            excluded.append(rel)
            continue
        try:
            text = read_keep_eol(path)
        except UnicodeDecodeError:
            print(f"UNDECODABLE (skipped): {rel}", file=sys.stderr)
            continue
        if not needs_header(rel, text):
            skipped_existing.append(rel)
            continue
        if not args.dry_run:
            add_header(path, COMMENT_STYLE[suffix])
        changed.append(rel)

    print(f"headers added : {len(changed)}")
    print(f"already have : {len(skipped_existing)}")
    print(f"excluded (3p): {len(excluded)}")
    if args.dry_run:
        for rel in changed:
            print(f"  would-add: {rel}")


if __name__ == "__main__":
    main()
