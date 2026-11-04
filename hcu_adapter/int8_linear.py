# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Bind the existing scaled INT8 GEMM to both official SGLang linear routes."""

from importlib import import_module
import types


_TARGETS = (
    ("sglang.srt.layers.quantization.compressed_tensors.schemes."
     "compressed_tensors_w8a8_int8", "CompressedTensorsW8A8Int8", "apply_weights"),
    ("sglang.srt.layers.quantization.w8a8_int8", "W8A8Int8LinearMethod", "apply"),
)


def install():
    from hcu_adapter import registry
    from sgl_kernel import int8_scaled_mm

    modules = []
    for path, owner, entry in _TARGETS:
        module = import_module(path)
        function = getattr(getattr(module, owner), entry)
        if (not isinstance(function, types.FunctionType)
                or function.__globals__ is not vars(module)
                or "int8_scaled_mm" not in function.__code__.co_names):
            raise RuntimeError(f"INT8 linear GEMM consumer changed: {path}.{owner}.{entry}")
        modules.append(module)
    # Validate both consumers before binding either. Keep the original methods,
    # including earlier aliases, and repair only their missing global symbol.
    for module in modules:
        module.int8_scaled_mm = int8_scaled_mm
        registry.register(
            "int8_linear_gemm", int8_scaled_mm,
            target=module.__name__ + ".int8_scaled_mm",
            mechanism=registry.MECH_REBIND,
            detail="existing Triton INT8 GEMM; original linear methods retained",
        )
