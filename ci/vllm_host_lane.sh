#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# vLLM host lane -- the vLLM-side adapter suite. Every file runs in its own
# fresh interpreter with the vLLM host bridges installed
# (HCU_ADAPTER_HOST=vllm): the suites' isolated fixtures demand a CUDA-clean
# process, so one pytest per file is the contracted lane form (see the
# docstring in tests/adapter/conftest.py, which scopes these files out of
# every other host's session).
#
# Usage:  bash ci/vllm_host_lane.sh [GPU_ID]     # omitted -> device 0
# Budget: ~10-15 min on one gfx938 card; each file pays one vllm import
#         (~40-60 s) and is capped at 300 s x HCU_L1_TIMEOUT_SCALE.
# Exit:   blocking mode (default) 0 green / 1 red.
#         VLLM_LANE_MODE=observe reports the same per-file inventory but
#         exits 0 unless the lane itself is broken (no vllm / no files) --
#         graduation tier for the known-red backlog, same pattern as
#         L15_MANIFEST_MODE=observe.
# Summary: "VLLM-LANE: <file> PASS|FAIL|EMPTY".
#
# VLLM_TARGET_ROOT (optional): the audited vLLM checkout the source-byte
# provenance tests pin to. Default = parent of the installed vllm package
# (self-consistency: proves the loaded modules resolve inside one install,
# catching shadow/overlay dists; pin a real checkout via env for full
# provenance).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT" || exit 1

GPU="${1:-0}"
TSCALE="${HCU_L1_TIMEOUT_SCALE:-1}"
export HIP_VISIBLE_DEVICES="$GPU"
# The suites' docstrings claim "no Plugin import/activation": tests drive the
# bridge modules directly (compat.install() et al). With vllm's platform
# plugins at their default, `import vllm._aiter_ops` resolves current_platform
# (HcuPlatform probe -> torch.cuda initialized, verified 2026-11-17), which
# kills the CUDA-clean precondition the isolated fixtures assert. Plugins off
# is the contracted lane form -- same as the clean-subprocess reruns in
# tests/adapter/conftest.py.
export VLLM_PLUGINS="${VLLM_PLUGINS:-none}"

if [ -z "${VLLM_TARGET_ROOT:-}" ]; then
    # find_spec locates the package without executing vllm/__init__.py, so
    # this probe stays cheap; HCU_ADAPTER_HOST=none keeps sitecustomize quiet.
    VLLM_TARGET_ROOT="$(env -u PYTHONPATH HCU_ADAPTER_HOST=none python3 - <<'PY'
import importlib.util, pathlib
spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise SystemExit(1)
print(pathlib.Path(spec.origin).resolve().parent.parent)
PY
)" || { echo "VLLM-LANE: vllm not importable on this stack"; exit 1; }
fi
export VLLM_TARGET_ROOT
# The phase root can expose the installed wheel through package symlinks.
# Preserve explicit phase overrides; the provenance test resolves the exact file.
export AR_ADAPTER_ROOT="${AR_ADAPTER_ROOT:-$ROOT}"
echo "VLLM-LANE: host=vllm gpu=$GPU target_root=$VLLM_TARGET_ROOT adapter_root=$AR_ADAPTER_ROOT"

# Git metadata belongs to this test phase, before source packages are resolved.
# Installed-wheel CI preserves the checkout and merge parents alongside symlinks.
if [ -e "$ROOT/.git" ]; then
    git -C "$ROOT" log -1 --format='VLLM-LANE: source_head=%H source_parents=%P' \
        || { echo "VLLM-LANE: unable to record checkout provenance"; exit 1; }
fi

# Record the installed release/owner paths without importing GPU frameworks.
# The shared Adapter lane also supports other admitted vLLM release lines;
# version strings here are evidence, not a new exact-wheel acceptance gate.
if ! env -u PYTHONPATH HCU_ADAPTER_HOST=none python3 - <<'PY'
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import sys

rows = []
for distribution, module in (
    ("vllm", "vllm"), ("hcu-adapter", "hcu_adapter"),
    ("hcu-vllm-das-plugin", "hcu_vllm_das"),
    ("torch", "torch"), ("triton", "triton"),
):
    spec = importlib.util.find_spec(module)
    try:
        dist = metadata.distribution(distribution)
        version = dist.version
        location = str(Path(dist.locate_file("")).resolve())
    except metadata.PackageNotFoundError:
        version = location = None
    rows.append(dict(distribution=distribution, version=version, location=location,
                     module_origin=str(Path(spec.origin).resolve()) if spec and spec.origin else None))
print("VLLM-LANE: inventory " + json.dumps(dict(python=sys.executable, packages=rows)))
PY
then
    echo "VLLM-LANE: unable to record component inventory"
    exit 1
fi

# Mirror of tests/adapter/conftest.py _is_vllm_host_file. conftest remains
# the source of truth for the host-domain scoping; the per-file counts below
# make any drift between the two lists visible in every run.
FILES=()
for f in tests/adapter/test_vllm_*.py; do [ -f "$f" ] && FILES+=("$f"); done
for f in tests/adapter/cases/vllm_*.py;  do [ -f "$f" ] && FILES+=("$f"); done
[ -f tests/adapter/test_model_mutation.py ] && FILES+=("tests/adapter/test_model_mutation.py")
[ ${#FILES[@]} -gt 0 ] || { echo "VLLM-LANE: no vllm-host files found"; exit 1; }

FAILS=()
for f in "${FILES[@]}"; do
    log="/tmp/l1_vllm_$(basename "$f" .py).log"
    timeout $((300 * TSCALE)) env HCU_ADAPTER_HOST=vllm \
        python3 -m pytest "$f" -q --no-header > "$log" 2>&1
    rc=$?
    summary="$(grep -E '= .*(passed|failed|no tests ran)' "$log" | tail -1)"
    if [ "$rc" = 0 ]; then
        echo "VLLM-LANE: $f PASS  ${summary:+-- $summary}"
    elif [ "$rc" = 5 ]; then
        # helper/case modules importable by pytest but defining no tests:
        # collection itself already exercised their import path
        echo "VLLM-LANE: $f EMPTY (no tests collected -- import-only module)"
    else
        echo "VLLM-LANE: $f FAIL (rc=$rc) -- tail:"
        tail -n 12 "$log" | sed 's/^/    /'
        FAILS+=("$f")
    fi
done

if [ ${#FAILS[@]} -gt 0 ]; then
    echo "VLLM-LANE: RED (${#FILES[@]} files, ${#FAILS[@]} failed)"
    [ "${VLLM_LANE_MODE:-blocking}" = observe ]         && { echo "VLLM-LANE: observe mode -- reported, not gating"; exit 0; }
    exit 1
fi
echo "VLLM-LANE: GREEN (${#FILES[@]} files)"
