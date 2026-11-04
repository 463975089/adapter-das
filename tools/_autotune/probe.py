# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Target-machine platform probe for --autotune.

Builds the machine picture the tier picker reasons over. Reuses the
adapter's own detectors wherever they exist (arch_name / has_fp8 / the
auto-rule family) so the probe and the runtime裁决 never drift apart;
everything here is read-only snapshotting, no tier is chosen here.

Known-trap quick rules are hardcoded from the bootstrap §5 ledger: they
turn documented box-lessons (hyhal half-pairing, tilelang 0.1.15
regression, baked NCCL topology) into FAIL-FAST/WARN lines before any
bench burns 20 minutes on a machine that cannot answer anyway.
"""

import importlib.util
import os
import pathlib
import platform
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hcu_adapter.hcu_env import arch_name, has_fp8  # noqa: E402


def _pip_version(dist_name):
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as dist_version

    try:
        return dist_version(dist_name)
    except PackageNotFoundError:
        return None


def _find_spec_name(name):
    spec = importlib.util.find_spec(name)
    return None if spec is None else (spec.origin or "namespace")


def _rocm_version():
    p = pathlib.Path("/opt/dtk/.info/rocm_version")
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _torch_dtk_tail():
    # torch builds carry the dtk stamp in the local version, e.g.
    # 2.11+dtk2610161756 (models.md image-stack rows)
    v = getattr(__import__("torch"), "__version__", "")
    m = re.search(r"\+dtk(\d+)", v)
    return m.group(1) if m else None


def _gpu_real_init_check(timeout_s=60):
    """验活二连: device_count + a real matmul. NVML enumeration succeeding
    while HSA submission is dead is the documented 'half-pairing' trap
    (bootstrap §5) — only the second probe catches it."""
    import concurrent.futures as cf

    code = (
        "import torch;"
        "n=torch.cuda.device_count();"
        "x=torch.randn(512,512,device='cuda:0');"
        "print(n, float((x@x).sum()))"
    )
    try:
        with cf.ThreadPoolExecutor(1) as ex:
            fut = ex.submit(
                subprocess.run,
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
            )
            r = fut.result()
        if r.returncode == 0:
            n, acc = r.stdout.split()
            return {"ok": True, "device_count": int(n), "real_matmul": float(acc)}
        return {"ok": False, "stage": "init", "stderr": r.stderr[-300:]}
    except cf.TimeoutError:
        return {
            "ok": False,
            "stage": "timeout",
            "hint": "torch real-init hung (hyhal mount missing? driver mismatch?)",
        }
    except Exception as e:  # subprocess spawn issues etc.
        return {"ok": False, "stage": "spawn", "error": str(e)[:200]}


def _vram_rows():
    """rocm-smi VRAM% table (same parse l1_gpu_gate.pick_gpu uses)."""
    try:
        out = subprocess.run(
            ["rocm-smi"], capture_output=True, text=True, timeout=20
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.splitlines():
        m = re.match(r"^\s*(\d+)\s+\S+\s+\S+\s+\S+\s+\S+\s+(\d+)%", line)
        if m:
            rows.append({"gpu": int(m.group(1)), "vram_pct": int(m.group(2))})
    return rows


def _known_trap_rules(pic):
    """bootstrap §5 box-lessons as mechanical checks. Each rule:
    (level, message) with level in fail/warn/ok/info."""
    rules = []

    # hyhal half-pairing: torch real init dead = nothing downstream works
    init = pic["gpu_real_init"]
    if init["ok"] is None:
        rules.append(("info", "torch real-init probe skipped (fast/mock mode)"))
    elif not init["ok"]:
        rules.append(
            ("fail", "torch real-init failed ({}): /opt/hyhal mount or "
             "driver pairing broken -- benches would hang; abort".format(
                 init.get("stage")))
        )
    else:
        rules.append(("ok", "torch real-init OK ({} devices)".format(
            init.get("device_count"))))

    # tilelang 0.1.15 ThreadSync regression (10-18 镜像升级役)
    tl = pic["versions"].get("tilelang")
    if tl is not None:
        good = tl.startswith("0.1.12")
        rules.append(
            ("ok" if good else "warn",
             "tilelang {} -- {}".format(
                 tl,
                 "mHC fused path safe" if good else
                 "NOT 0.1.12: mhc_pre_big_fuse ThreadSync hard-error "
                 "(0.1.15 regression) -- pin 0.1.12 before serving")))

    # baked NCCL topology from another box (the 508->BW1101 trap)
    topo = os.environ.get("NCCL_TOPO_FILE")
    if topo:
        rules.append(("warn", "NCCL_TOPO_FILE={} -- image-baked topology "
                     "from a foreign box causes ncclCommInitRank internal "
                     "error here; recipe: unset NCCL_TOPO_FILE".format(topo)))
    if os.environ.get("ROCR_VISIBLE_DEVICES"):
        rules.append(("warn", "ROCR_VISIBLE_DEVICES set together with CUDA "
                     "semantics = documented CUDA-init假死 double-set; "
                     "recipe: unset ROCR_VISIBLE_DEVICES"))

    # torch dtk tail vs /opt/dtk roster
    tail = pic["versions"].get("torch_dtk_tail")
    rocm = pic["versions"].get("rocm_version")
    if tail and rocm:
        # torch stamp 2610161756 style vs rocm_version 6.3-style -- compare
        # only the loose yymmdd window when both carry it; mismatch here is
        # informational (mirrors rebuilt at different nights are normal)
        rules.append(("info", "torch dtk-stamp {} / /opt/dtk rocm_version {}"
                      .format(tail, rocm)))
    elif pic["versions"].get("torch_hip") and rocm is None:
        rules.append(("warn", "HIP torch but /opt/dtk/.info/rocm_version "
                     "unreadable -- deepgemm bridge and JIT toolchain may "
                     "refuse"))

    return rules


def probe(deep_gpu_check=True):
    """Collect the platform picture. deep_gpu_check=False skips the
    subprocess 真init probe (mock/test mode)."""
    pic = {
        "host": {
            "hostname": platform.node(),
            "os": platform.platform(),
            "python": platform.python_version(),
        },
        "gpu": {
            "arch": arch_name(),
            "has_fp8": has_fp8(),
        },
        "versions": {
            "torch": _pip_version("torch") or getattr(
                __import__("torch"), "__version__", None),
            "torch_hip": getattr(__import__("torch"), "version").hip is not None,
            "torch_dtk_tail": _torch_dtk_tail(),
            "rocm_version": _rocm_version(),
            "aiter": _pip_version("aiter"),
            "tilelang": _pip_version("tilelang"),
            "deep_gemm": _pip_version("deep_gemm") or _pip_version("deepgemm"),
            "sglang": _pip_version("sglang"),
            "sgl_kernel": _pip_version("sgl-kernel"),
        },
        "paths": {
            "aiter_spec": _find_spec_name("aiter"),
            "hyhal_mounted": pathlib.Path("/opt/hyhal").exists(),
        },
        "vram_rows": _vram_rows() if deep_gpu_check else [],
        "gpu_real_init": (
            _gpu_real_init_check() if deep_gpu_check
            else {"ok": None, "note": "skipped (mock mode)"}
        ),
    }
    pic["traps"] = [
        {"level": lv, "msg": msg} for lv, msg in _known_trap_rules(pic)
    ]
    pic["aborted"] = any(t["level"] == "fail" for t in pic["traps"])
    return pic


def format_pic(pic):
    lines = []
    h, g, v = pic["host"], pic["gpu"], pic["versions"]
    lines.append("host: {} / py{}".format(h["hostname"], h["python"]))
    lines.append("gpu: arch={} fp8={} hyhal={}".format(
        g["arch"] or "?", g["has_fp8"], pic["paths"]["hyhal_mounted"]))
    lines.append("versions: torch={} (hip={}) aiter={} tilelang={} "
                 "deep_gemm={} sglang={}".format(
                     v["torch"], v["torch_hip"], v["aiter"], v["tilelang"],
                     v["deep_gemm"], v["sglang"]))
    for t in pic["traps"]:
        mark = {"fail": "FAIL", "warn": "WARN", "ok": "ok", "info": "note"}[t["level"]]
        lines.append("[{:4}] {}".format(mark, t["msg"]))
    if pic["aborted"]:
        lines.append("probe verdict: ABORT (fix the FAIL rows first)")
    else:
        lines.append("probe verdict: GO")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(format_pic(probe(deep_gpu_check="--fast" not in sys.argv)))
