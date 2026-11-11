# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Execute the gate's ownership-transfer block without launching GPU tests."""

import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="Linux Bash gate")


@pytest.mark.parametrize("mode,tree,families,transferred", [
    ("off", "available", "dense", set()),
    ("observe", "available", "dense", set()),
    ("blocking", "", "dense", set()),
    ("blocking", "available", "dense", {"dense"}),
    ("blocking", "available", "mla,moe", {"mla", "moe"}),
    ("blocking", "available", "dense,mla,moe,quantization,mamba",
     {"dense", "mla", "moe", "quantization", "mamba"}),
])
def test_gate_transfers_only_selected_blocking_cases(mode, tree, families, transferred):
    result = _run_block(mode, tree, families)
    assert result.returncode == 0, result.stderr
    args = result.stdout.split("CAPTURE\n", 1)[1].splitlines()
    assert args[:7] == ["kernels", "900", "python3", "-m", "pytest",
                       "tests/kernels", "tests/adapter"]
    actual = [args[i + 1] for i, value in enumerate(args) if value == "--deselect"]
    manifest = json.loads((ROOT / "tests/manifests/l15_family_matrix_v1.json").read_text())
    expected = [case["nodeid"] for family in manifest["families"]
                if family["id"] in transferred for case in family["cases"]
                if case["kind"] == "adapter-pytest"]
    assert actual == expected


def test_invalid_family_stops_before_broad_test_execution():
    result = _run_block("blocking", "available", "does-not-exist")
    assert result.returncode == 2
    assert "CAPTURE" not in result.stdout


def _run_block(mode, tree, families):
    source = (ROOT / "ci/l1_gpu_gate.sh").read_text()
    start = source.index("# Transfer ownership only")
    end = source.index("# L1.5: upstream attention", start)
    # Keep the production selection and actual pytest argv construction. Only
    # replace the executor, so this CPU test cannot allocate a device.
    script = 'set -u\nrun_seg() { printf "CAPTURE\\n"; printf "%s\\n" "$@"; }\n'
    script += source[start:end]
    env = dict(os.environ, ROOT=str(ROOT), UP_TREE=tree,
               L15_FAMILY_MATRIX_MODE=mode, L15_FAMILY_MATRIX_FAMILIES=families,
               HCU_ADAPTER_HOST="none", PYTHONPATH=str(ROOT))
    return subprocess.run(["bash", "-c", script], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=30)
