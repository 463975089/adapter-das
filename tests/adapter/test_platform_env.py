# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""R-08 A-tier platform env defaults: semantics pin.

platform_seam.platform_env_defaults() is pure os.environ.setdefault calls —
it must (a) fill every A-tier var when absent, (b) NEVER clobber an explicit
export (env三原则: explicit override is top priority), (c) be idempotent.
No GPU / no sglang import needed — the function is import-safe standalone.

Pytest-collectable (L1 kernels segment collects tests/adapter); direct run
also works: python tests/adapter/test_platform_env.py
(2026-10-19: module-level sys.exit crashed pytest collection -> INTERNALERROR,
refactored to function + __main__ guard, same 3 checks).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hcu_adapter.platform_seam import platform_env_defaults  # noqa: E402

A_TIER = {
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
    "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
    "SGLANG_RUST_BUILD_MODE": "never",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
    "HCU_DEEPGEMM_BRIDGE": "1",
}


class _Env:
    """save A_TIER on enter, restore on exit -- order-independent tests."""

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in A_TIER}
        for k in A_TIER:
            os.environ.pop(k, None)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_platform_env_defaults():
    # 1. fills when absent
    with _Env():
        platform_env_defaults()
        got = {k: os.environ.get(k) for k in A_TIER}
        assert got == A_TIER, f"fills-absent: {got}"

    # 2. explicit export wins (unset-one-by-one across both value kinds)
    with _Env():
        for k, explicit in (
            ("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "true"),
            ("HCU_DEEPGEMM_BRIDGE", "0"),
            ("SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND", "rust"),
        ):
            os.environ[k] = explicit
        platform_env_defaults()
        assert os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "true", (
            "explicit-wins")
        assert os.environ["HCU_DEEPGEMM_BRIDGE"] == "0", "explicit-wins"
        assert os.environ["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] == "rust", (
            "explicit-wins")

    # 3. idempotent
    with _Env():
        platform_env_defaults()
        platform_env_defaults()
        assert all(os.environ.get(k) for k in A_TIER), "idempotent"


if __name__ == "__main__":
    try:
        test_platform_env_defaults()
        print("PASS test_platform_env_defaults")
        print("RESULT: PASS")
    except AssertionError as e:
        print(f"FAIL test_platform_env_defaults: {e}")
        print("RESULT: FAIL")
        sys.exit(1)
