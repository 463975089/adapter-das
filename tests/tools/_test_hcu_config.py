# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Gate for the config-file backend layer (hcu_adapter.yaml).

Covers: discovery, precedence (env > config > auto > default), loud
rejection of typos/planned backends, compound allreduce mapping, the
wizard catalog/check modes, and GPU parity for the newly wired aiter
rms_norm + torch silu tiers.

Run: HIP_VISIBLE_DEVICES=6 HCU_CONFIG=<file> python tests/tools/_test_hcu_config.py
"""

import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hcu_adapter import hcu_config, hcu_env  # noqa: E402

FAILS = []


def ok(tag, cond, detail=""):
    print(f"{tag}: {'OK' if cond else 'FAIL ' + detail}", flush=True)
    if not cond:
        FAILS.append(tag)


def write_cfg(text):
    f = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8", newline="\n"
    )
    f.write(text)
    f.close()
    return f.name


def with_cfg(text, env_extra=None):
    """Fresh-load a config file and return (state, restore fn)."""
    path = write_cfg(text)
    old = os.environ.get("HCU_CONFIG")
    old_impl = os.environ.get("HCU_SILU_MUL_IMPL")
    os.environ["HCU_CONFIG"] = path
    if env_extra:
        for k, v in env_extra.items():
            os.environ[k] = v
    hcu_config.load(force=True)
    # elementwise/layernorm cache their resolved choice per process; the
    # resolve() chain itself is what we gate here, plus a fresh-process
    # parity call at the bottom.
    return path

    # (restore handled by caller)


# 1. no config file -> defaults, quiet
os.environ.pop("HCU_CONFIG", None)
os.environ.pop("HCU_SILU_MUL_IMPL", None)
hcu_config.load(force=True)
ok("no-file-path-none", hcu_config.path() is None)
v, src = hcu_env.resolve("HCU_SILU_MUL_IMPL")
ok("no-file-default", (v, src) == ("auto", "default"), f"{v}({src})")

# 2. valid file -> config source + effective value
p2 = with_cfg("version: 1\nops:\n  silu_and_mul: lightop\n")
v, src = hcu_env.resolve("HCU_SILU_MUL_IMPL")
ok("file-config-source", (v, src) == ("lightop", "config"), f"{v}({src})")

# 3. env beats file (debug tier)
p3 = with_cfg("version: 1\nops:\n  silu_and_mul: lightop\n",
              {"HCU_SILU_MUL_IMPL": "torch"})
v, src = hcu_env.resolve("HCU_SILU_MUL_IMPL")
ok("env-beats-file", (v, src) == ("torch", "explicit"), f"{v}({src})")

# 4. unknown op -> loud
try:
    with_cfg("version: 1\nops:\n  silu_and_mull: triton\n")
    hcu_config.load(force=True)
    ok("unknown-op-rejected", False, "no error raised")
except hcu_config.HcuConfigError as e:
    ok("unknown-op-rejected", "silu_and_mull" in str(e))

# 5. unknown backend -> loud
try:
    with_cfg("version: 1\nops:\n  silu_and_mul: tensorrt\n")
    hcu_config.load(force=True)
    ok("unknown-backend-rejected", False, "no error raised")
except hcu_config.HcuConfigError as e:
    ok("unknown-backend-rejected", "silu_and_mul" in str(e))

# 6. planned backend -> loud
try:
    with_cfg("version: 1\nops:\n  paged_mqa_logits: lightop\n")
    hcu_config.load(force=True)
    ok("planned-rejected", False, "no error raised")
except hcu_config.HcuConfigError as e:
    ok("planned-rejected", "planned" in str(e))

# 7. compound allreduce mapping (custom AR retired 2026-10-19: aiter/nccl only)
with_cfg("version: 1\nops:\n  allreduce: nccl\n")
v2, s2 = hcu_env.resolve("HCU_AITER_AR")
ok("compound-nccl", (v2, s2) == ("0", "config"),
   f"AITER={v2}({s2})")
with_cfg("version: 1\nops:\n  allreduce: aiter\n")
v2, s2 = hcu_env.resolve("HCU_AITER_AR")
ok("compound-aiter", (v2, s2) == ("1", "config"),
   f"AITER={v2}({s2})")

# 7b. retired backend refused loudly (R-01 conviction, retired 2026-10-19)
try:
    with_cfg("version: 1\nops:\n  allreduce: custom\n")
    hcu_config.load(force=True)
    ok("retired-rejected", False, "no error raised")
except hcu_config.HcuConfigError as e:
    ok("retired-rejected", "custom" in str(e) or "unknown" in str(e).lower())

# 8. bad version -> loud
try:
    with_cfg("version: 2\nops: {}\n")
    hcu_config.load(force=True)
    ok("version-rejected", False, "no error raised")
except hcu_config.HcuConfigError as e:
    ok("version-rejected", "version" in str(e))

# 9. rms_norm config -> resolve
with_cfg("version: 1\nops:\n  rms_norm: aiter\n")
v, src = hcu_env.resolve("HCU_RMS_NORM_IMPL")
ok("rms-config", (v, src) == ("aiter", "config"), f"{v}({src})")

# 10. wizard modes (catalog listing + file check) as subprocesses
root = os.path.join(os.path.dirname(__file__), "..", "..")
r = subprocess.run(
    [sys.executable, os.path.join(root, "tools", "hcu_config_wizard.py"),
     "--list"],
    capture_output=True, text=True, timeout=120,
)
ok("wizard-list", "silu_and_mul" in r.stdout and "planned" in r.stdout,
   r.stdout[-200:] + r.stderr[-200:])
good = write_cfg("version: 1\nops:\n  silu_and_mul: torch\n")
r2 = subprocess.run(
    [sys.executable, os.path.join(root, "tools", "hcu_config_wizard.py"),
     "--check", good],
    capture_output=True, text=True, timeout=120,
)
ok("wizard-check", "OK" in r2.stdout and "HCU_SILU_MUL_IMPL = torch" in r2.stdout,
   r2.stdout[-200:] + r2.stderr[-200:])

# 11. GPU parity in fresh processes (backend caches resolve once/process)
if torch.cuda.is_available():
    import torch  # refresh

    def parity(cfg_ops, func, tag):
        pcfg = write_cfg(f"version: 1\nops:\n{cfg_ops}")
        code = (
            "import sys, os, torch;"
            f"sys.path.insert(0, {root!r});"
            "torch.manual_seed(3);"
            "from sgl_kernel import rmsnorm, silu_and_mul;"
            f"fn = {func};"
            "x = torch.randn(256, 1024, device='cuda', dtype=torch.bfloat16);"
            "w = torch.randn(1024, device='cuda', dtype=torch.bfloat16);"
            "got = fn(x, w) if fn is rmsnorm else fn(x);"
            "xf, wf = x.float(), (w.float() if fn is rmsnorm else None);"
            "ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * wf"
            ").to(torch.bfloat16).float() if fn is rmsnorm else "
            "(torch.nn.functional.silu(xf[:, :512]) * xf[:, 512:]"
            ").to(torch.bfloat16).float();"
            "md = (got.float() - ref).abs().max().item();"
            "print('MAXDIFF', md)"
        )
        env = dict(os.environ, HCU_CONFIG=pcfg, HIP_VISIBLE_DEVICES=os.environ.get(
            "HIP_VISIBLE_DEVICES", "6"))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=300, env=env)
        md = None
        for line in r.stdout.splitlines():
            if line.startswith("MAXDIFF"):
                md = float(line.split()[1])
        ok(tag, md is not None and md < 0.05, f"md={md} {r.stderr[-200:]}")

    parity("  rms_norm: aiter\n", "rmsnorm", "parity-rms-aiter")
    parity("  rms_norm: torch\n", "rmsnorm", "parity-rms-torch")
    parity("  silu_and_mul: torch\n", "silu_and_mul", "parity-silu-torch")
    parity("  silu_and_mul: lightop\n", "silu_and_mul", "parity-silu-lightop")

print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
sys.exit(0 if not FAILS else 1)
