# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Explicit opt-in installation of the candidate LightOp BF16 GEMM bridge."""

import importlib
import sys
import types

from hcu_adapter import registry


def install():
    """Expose one verified provider symbol; this is not a complete vendor shim."""
    from hcu_adapter.kernels.gemm_lightop import bf16_gemm_nt

    try:
        module = importlib.import_module("deep_gemm")
    except ModuleNotFoundError as error:
        if error.name != "deep_gemm":
            raise
        module = types.ModuleType("deep_gemm")
        module.__doc__ = "Partial HCU provider; only explicitly installed operators are available."
        sys.modules["deep_gemm"] = module
    current = getattr(module, "bf16_gemm_nt", None)
    if current is not None and current is not bf16_gemm_nt:
        raise RuntimeError("An existing deep_gemm.bf16_gemm_nt provider is already installed")
    module.bf16_gemm_nt = bf16_gemm_nt
    registry.register(
        "deep_gemm.bf16_gemm_nt",
        "hcu_adapter.kernels.gemm_lightop.bf16_gemm_nt",
        "deep_gemm.bf16_gemm_nt",
        "rebind",
    )
    print("HCU adapter runtime: deep_gemm.bf16_gemm_nt -> lightop (candidate)")
