# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run CPU-only report cases in a clean interpreter when the suite loaded a host."""

import os
from pathlib import Path
import subprocess
import sys


def rerun_if_host_loaded(request):
    if not any(name == "sglang" or name.startswith(("sglang.", "hcu_adapter.bridges."))
               for name in sys.modules):
        return False
    root = Path(__file__).resolve().parents[2]
    environment = dict(
        os.environ, HCU_ADAPTER_HOST="none", VLLM_PLUGINS="none",
        HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
        PYTHONPATH=str(root), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", request.node.nodeid],
        cwd=root, env=environment, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True
