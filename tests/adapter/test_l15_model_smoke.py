"""Offline failure contracts for the real-model smoke launcher."""

import copy
import os
from pathlib import Path
import sys

import pytest

from tools.l15_model_smoke import tiny_config, validate_generation, validate_mode
from tools.l15_runtime import run_job


def test_tiny_models_are_separate_architectures():
    configs = [tiny_config(f) for f in ("dense", "moe", "mla", "mamba", "hybrid")]
    assert len({c["architectures"][0] for c in configs}) == 5
    assert all(c["num_hidden_layers"] == 2 for c in configs)
    # Upstream fused RoPE's admitted head sizes begin at 64.
    assert configs[1]["head_dim"] == 64
    assert configs[-1]["layer_types"] == ["linear_attention", "full_attention"]
    with pytest.raises(ValueError, match="unknown"):
        tiny_config("typo")


@pytest.mark.parametrize("fault", ["batch", "tokens", "vocab", "logprobs", "nan", "prob-token"])
def test_invalid_generation_is_never_a_pass(fault):
    document = [
        {"output_ids": [7] * 8,
         "meta_info": {"output_token_logprobs": [[-1.0, 7, None]] * 8}}
        for _ in range(2)
    ]
    validate_generation(document, 256)
    broken = copy.deepcopy(document)
    if fault == "batch":
        broken.pop()
    elif fault == "tokens":
        broken[0]["output_ids"].pop()
    elif fault == "vocab":
        broken[0]["output_ids"][0] = 256
    elif fault == "logprobs":
        broken[0]["meta_info"]["output_token_logprobs"] = []
    elif fault == "nan":
        broken[0]["meta_info"]["output_token_logprobs"][0][0] = float("nan")
    else:
        broken[0]["meta_info"]["output_token_logprobs"][0][1] = 8
    with pytest.raises(ValueError):
        validate_generation(broken, 256)


def test_graph_flag_without_capture_is_not_graph_evidence():
    validate_mode({"internal_states": [{"decode_cuda_graph_bs": [],
                   "memory_usage": {"graph": {"decode": 0.02}}}]}, "graph")
    validate_mode({"internal_states": [{"decode_cuda_graph_bs": [],
                   "memory_usage": {"graph": {"decode": 0}}}]}, "eager")
    for info in ({}, {"internal_states": [{}]},
                 {"internal_states": [{"decode_cuda_graph_bs": []}]},
                 {"internal_states": [{"decode_cuda_graph_bs": [1, 2],
                                      "memory_usage": {"graph": {"decode": 0}}}]}):
        with pytest.raises(ValueError):
            validate_mode(info, "graph")


@pytest.mark.skipif(os.name != "posix", reason="remote Linux job lifecycle")
@pytest.mark.parametrize("nested", [False, True])
def test_timeout_stops_own_descendants(tmp_path, nested):
    script = (
        "import subprocess,sys,time\n"
        "p=subprocess.Popen([sys.executable,'-S','-c','import time;time.sleep(120)'])\n"
        "print(p.pid,flush=True)\n"
        "time.sleep(120)\n"
    )
    log = tmp_path / "job.log"
    pid_log = log
    env = dict(os.environ)
    if nested:
        pid_log = tmp_path / "inner.log"
        script = (
            "import os,sys\n"
            "from tools.l15_runtime import run_job\n"
            f"run_job([sys.executable,'-S','-c',{script!r}],cwd={str(tmp_path)!r},"
            f"env=dict(os.environ),timeout=100,log={str(pid_log)!r})\n"
        )
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = run_job(
        [sys.executable, "-S", "-c", script], cwd=tmp_path, env=env,
        timeout=2, log=log,
    )
    assert result["timed_out"] and result["returncode"] == 124
    child = int(pid_log.read_text().strip())
    stat = Path(f"/proc/{child}/stat")
    if stat.exists():
        assert stat.read_text().split()[2] == "Z", "timed-out child still executing"
