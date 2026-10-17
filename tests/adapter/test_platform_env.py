"""R-08 A-tier platform env defaults: semantics pin.

platform_seam.platform_env_defaults() is pure os.environ.setdefault calls —
it must (a) fill every A-tier var when absent, (b) NEVER clobber an explicit
export (env三原则: explicit override is top priority), (c) be idempotent.
No GPU / no sglang import needed — the function is import-safe standalone.

Run: python tests/adapter/test_platform_env.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sgl_kernel._adapters.platform_seam import platform_env_defaults  # noqa: E402

FAILS = []

A_TIER = {
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
    "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
    "SGLANG_RUST_BUILD_MODE": "never",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
    "HCU_DEEPGEMM_BRIDGE": "1",
}


def ok(tag, cond, detail=""):
    print(f"{tag}: {'OK' if cond else 'FAIL ' + detail}", flush=True)
    if not cond:
        FAILS.append(tag)


saved = {k: os.environ.get(k) for k in A_TIER}
try:
    # 1. fills when absent
    for k in A_TIER:
        os.environ.pop(k, None)
    platform_env_defaults()
    ok("fills-absent", all(os.environ.get(k) == v for k, v in A_TIER.items()),
       str({k: os.environ.get(k) for k in A_TIER}))

    # 2. explicit export wins (unset-one-by-one across both value kinds)
    for k, explicit in (
        ("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "true"),
        ("HCU_DEEPGEMM_BRIDGE", "0"),
        ("SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND", "rust"),
    ):
        os.environ[k] = explicit
    platform_env_defaults()
    ok("explicit-wins",
       os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "true"
       and os.environ["HCU_DEEPGEMM_BRIDGE"] == "0"
       and os.environ["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] == "rust")

    # 3. idempotent
    platform_env_defaults()
    platform_env_defaults()
    ok("idempotent", all(os.environ.get(k) for k in A_TIER))
finally:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
sys.exit(1 if FAILS else 0)
