# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared execution contracts for the HCU L1.5 readiness gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def audit_source(tree, expected_commit=None):
    try:
        from tools.upstream_source import audit
    except ModuleNotFoundError:
        from upstream_source import audit
    report = audit(tree)
    if expected_commit is not None and report["commit"] != expected_commit:
        report["passed"] = False
        report["expected_commit"] = expected_commit
    return report


class UpstreamSourceError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(
            f"community source is not pristine: {len(report['changes'])} changed/unverified "
            f"files, {len(report['added_sources'])} added source files; run tools.upstream_source"
        )


def run_job(argv, *, cwd, env, timeout, log):
    """Bound a Linux test job and all descendants in its own process group."""
    if os.name != "posix":
        raise RuntimeError("HCU test jobs require a POSIX execution host")
    started = time.monotonic()
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        terminating = False
        previous_term = signal.getsignal(signal.SIGTERM)

        def terminate(signum, frame):
            # Let an outer upgrade timeout unwind this controller's finally
            # block, including workers in their own process groups.
            nonlocal terminating
            terminating = True
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, terminate)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                if not terminating and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                # A crashed launcher may leave scheduler children alive even
                # when its own wait completed. Only this job owns this group.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            finally:
                signal.signal(signal.SIGTERM, previous_term)
    return {"argv": list(argv), "returncode": 124 if timed_out else process.returncode,
            "timed_out": timed_out, "seconds": time.monotonic() - started,
            "log": str(log)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(name, 0o644)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def test_env(up_tree, adapter_root):
    env = dict(os.environ)
    # Do not inherit an older source tree or a disabled/different host.
    env["PYTHONPATH"] = os.pathsep.join((str(adapter_root), str(up_tree / "python")))
    env["HCU_ADAPTER_HOST"] = "sglang"
    env["VLLM_PLUGINS"] = "none"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def git_stamp(root):
    def git(*args):
        result = subprocess.run(
            ["git", "-c", f"safe.directory={Path(root).resolve()}",
             "-C", str(root), *args], capture_output=True, text=True,
            timeout=30, check=True,
        )
        return result.stdout.strip()
    return {"root": str(root), "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def preflight(up_tree, adapter_root, binding_manifest):
    """Verify the exact source binding and allocate on one real HIP device."""
    integrity = audit_source(up_tree)
    if not integrity["passed"]:
        raise UpstreamSourceError(integrity)
    manifest = json.loads(Path(binding_manifest).read_text(encoding="utf-8"))
    hashes = {}
    for relative, expected in manifest["upstream"]["binding_files"].items():
        path = (up_tree / relative).resolve()
        path.relative_to(up_tree.resolve())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"upstream binding mismatch: {relative}")
        hashes[relative] = actual
    env = test_env(up_tree, adapter_root)
    env["HCU_ADAPTER_HOST"] = "none"
    probe = """
import importlib.util, json, os, socket, torch
assert torch.version.hip, 'a HIP build of Torch is required'
assert torch.cuda.is_available(), 'no usable HIP device'
assert torch.cuda.device_count() == 1, 'expose exactly one HCU device'
x = torch.ones(1, device='cuda')
assert x.add(1).item() == 2
torch.cuda.synchronize()
p = torch.cuda.get_device_properties(0)
print('L15_RUNTIME ' + json.dumps({
 'hostname': socket.gethostname(), 'torch': torch.__version__,
 'hip': torch.version.hip, 'device_count': torch.cuda.device_count(),
 'device': p.name, 'arch': getattr(p, 'gcnArchName', ''),
 'visibility': {k: os.environ.get(k) for k in
                ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')},
 'origins': {name: importlib.util.find_spec(name).origin for name in
             ('sglang', 'sgl_kernel', 'hcu_adapter')}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], env=env, cwd=adapter_root,
        capture_output=True, text=True, timeout=120, check=False,
    )
    if result.returncode:
        raise ValueError(f"HCU preflight failed: {result.stderr[-3000:]}")
    lines = [line[12:] for line in result.stdout.splitlines()
             if line.startswith("L15_RUNTIME ")]
    if len(lines) != 1:
        raise ValueError("HCU preflight did not emit runtime evidence")
    runtime = json.loads(lines[0])
    expected_roots = {"sglang": up_tree / "python",
                      "sgl_kernel": adapter_root, "hcu_adapter": adapter_root}
    for name, root in expected_roots.items():
        Path(runtime["origins"][name]).resolve().relative_to(root.resolve())
    runtime.update(adapter=git_stamp(adapter_root), upstream=git_stamp(up_tree),
                   binding_files=hashes, source_integrity=integrity)
    return runtime


def check_outcomes(document, expected):
    """Zero skips/xfails, a real call phase, and the requested node are required."""
    if document.get("artifact") != "l15-pytest-outcomes":
        return False
    if document.get("exitstatus") != 0 or document.get("collection_errors"):
        return False
    tests = document.get("tests", [])
    if not tests:
        return False
    for item in tests:
        node = item.get("nodeid", "")
        if not (node == expected or node.startswith(expected + "[")):
            return False
        phases = item.get("phases", {})
        if item.get("outcome") != "passed" or phases.get("call", {}).get("outcome") != "passed":
            return False
        if any(p.get("outcome") != "passed" or "wasxfail" in p
               for p in phases.values()):
            return False
        if any(p.get("outcome") != "passed" or "wasxfail" in p
               for p in item.get("subtests", [])):
            return False
    return True
