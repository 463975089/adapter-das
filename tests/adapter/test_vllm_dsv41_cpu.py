"""Run CPU interpreter oracles independently of the HCU suite's device state."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("case", ["vllm_dsv41_mhc_cpu.py", "vllm_dsv41_indexer_cpu.py"])
def test_cpu_oracles(case):
    pytest.importorskip("vllm.models.deepseek_v41")
    path = Path(__file__).with_name("cases") / case
    env = dict(os.environ, HCU_ADAPTER_HOST="none", HIP_VISIBLE_DEVICES="",
               CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", VLLM_PLUGINS="",
               VLLM_ROCM_USE_AITER="0")
    subprocess.run([sys.executable, "-m", "pytest", "-q", str(path)],
                   env=env, check=True)
