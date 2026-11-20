# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Gates-lane runner for the fresh_process test class (W4-2, 2026-11-19).

These tests spawn fresh interpreters or torch.distributed workers
(23-53s each, ~370s serial). They used to run inside the L1 kernels
segment where their interpreter-startup cost was pure serial wall time;
the gate now runs them HERE as one more concurrent lane -- spawn
semantics are unchanged, only the scheduling moved. Exit code is
pytest's, which the gate harness already understands.
"""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

rc = subprocess.call(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/kernels",
        "tests/adapter",
        "-m",
        "fresh_process",
        "-q",
        "--no-header",
        "--tb=short",
    ],
    cwd=ROOT,
)
print("fresh_process lane rc=%d" % rc)
sys.exit(rc)
