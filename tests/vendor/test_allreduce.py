# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Aiter custom all-reduce: vendor stress + adapter kill-switch gate tests.

The two failure classes this file pins down (both seen live on gfx93x):
  1. size/dtype-dependent numeric bugs   -> eager stress vs exact fp32 gold;
  2. CUDA-graph replay corruption        -> capture/replay with fresh inputs
     (eager passes, replay is garbage — bug 1eb038d).

Running: plain `pytest tests/vendor/test_allreduce.py` on a node with >=2
free GPUs drives everything, including the torchrun subprocess. The gate
tests need 1 GPU. On non-gfx93 archs the auto-refuse test self-skips.
"""

import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))  # tests/vendor/
TESTS = os.path.dirname(HERE)
ROOT = os.path.dirname(TESTS)  # adapter root (PYTHONPATH entry)

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)


def _has_sglang():
    try:
        import sglang  # noqa: F401

        return True
    except ImportError:
        return False


requires_sglang = pytest.mark.skipif(
    not _has_sglang(), reason="sglang not installed on this host"
)


def _arch_name():
    try:
        return (
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        )
    except Exception:
        return ""


def _sglang_root():
    import sglang

    return os.path.dirname(os.path.dirname(os.path.abspath(sglang.__file__)))


def _run_gate_probe(extra_env):
    """Subprocess: import the bridge, report whether dispatch is refused."""
    code = (
        "import os,sys\n"
        "import torch\n"
        "assert torch.version.hip is not None, 'HIP only'\n"
        "import sglang.srt.distributed.device_communicators."
        "custom_all_reduce as car\n"
        "fn = car.dispatch_custom_allreduce\n"
        "print('GATE', 'REFUSED' if fn.__name__ == '_hcu_no_aiter_ar'"
        " else 'ACTIVE', fn.__name__)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + _sglang_root()
    env.pop("HCU_AITER_AR", None)
    env.update(extra_env)
    # fresh interpreter: sitecustomize must be exercised exactly once
    p = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, timeout=300,
    )
    for line in p.stdout.splitlines():
        if line.startswith("GATE "):
            return line.split()[1], p
    return None, p


def _sglang_uses_new_aiter_ca():
    try:
        import sglang.srt.distributed.device_communicators.\
            custom_all_reduce as car

        with open(car.__file__, encoding="utf-8", errors="ignore") as f:
            return "aiter.dist.device_communicators" in f.read()
    except Exception:
        return False


@requires_gpu
@requires_sglang
def test_gate_auto_rule_matches_aiter_layout():
    """gfx93*: old-aiter sglang dispatch -> REFUSED; new aiter.dist
    GraphSafe path (stress-passed) -> ACTIVE without any env var."""
    if not _arch_name().startswith("gfx93"):
        pytest.skip(f"auto-rule targets gfx93*, this host is {_arch_name()}")
    want = "ACTIVE" if _sglang_uses_new_aiter_ca() else "REFUSED"
    state, p = _run_gate_probe({})
    assert state == want, (
        f"auto-rule should resolve to {want} on this layout, got {state}"
        f"\n{p.stdout}\n{p.stderr[-2000:]}"
    )


@requires_gpu
@requires_sglang
def test_gate_opt_in_env_restores_dispatch():
    if not _arch_name().startswith("gfx93") or _sglang_uses_new_aiter_ca():
        pytest.skip("opt-in is meaningful on old-aiter gfx93* hosts")
    state, p = _run_gate_probe({"HCU_AITER_AR": "1"})
    assert state == "ACTIVE", (
        f"HCU_AITER_AR=1 should bypass the auto-rule: {state}"
        f"\n{p.stdout}\n{p.stderr[-2000:]}"
    )


@requires_gpu
@requires_sglang
def test_gate_disable_list_refuses_any_arch():
    state, p = _run_gate_probe({"HCU_DISABLE_OPS": "aiter_fa,aiter_ar"})
    assert state == "REFUSED", (
        f"HCU_DISABLE_OPS must refuse on any arch: {state}"
        f"\n{p.stdout}\n{p.stderr[-2000:]}"
    )


@requires_gpu
@requires_sglang
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="stress needs >=2 GPUs"
)
def test_aiter_ar_stress_eager_and_graph():
    """Drive vendor/_ar_worker.py under torchrun (HCU_AITER_AR=1 on purpose:
    we test the vendor kernel; the gfx93* gate would otherwise refuse)."""
    env = dict(os.environ)
    env["HCU_AITER_AR"] = "1"
    env["PYTHONPATH"] = ROOT + os.pathsep + _sglang_root()
    env.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29711")
    p = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node=2", os.path.join(HERE, "_ar_worker.py"),
        ],
        capture_output=True, text=True, env=env, timeout=900,
    )
    print(p.stdout[-8000:])
    print(p.stderr[-3000:], file=sys.stderr)
    lines = [
        ln for ln in p.stdout.splitlines() if ln.startswith("ARSTRESS")
    ]
    assert lines, f"worker produced no ARSTRESS output\n{p.stderr[-3000:]}"
    bad = [
        ln for ln in lines
        if ln.split()[1] in ("FAIL", "ERROR")
    ]
    assert not bad, "vendor all-reduce stress failures:\n" + "\n".join(bad)
    if any(ln.split()[1] == "SKIP" for ln in lines):
        pytest.skip("aiter AR unavailable: " + next(
            ln for ln in lines if ln.split()[1] == "SKIP"
        ))
