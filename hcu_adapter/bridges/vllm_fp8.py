# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Register the existing official dense FP8 fallback on validated HCU hosts.

No math, quantization, model or architecture impersonation lives here. Keep
all official priorities and support/can_implement/disabled-kernel checks.
"""
from __future__ import annotations

from importlib import import_module
import inspect

# gfx938 has no evidence in this change; leave it and all other arches alone.
VALIDATED_ARCHES = frozenset({'gfx936'})


def install() -> bool:
    from hcu_adapter.bridges.vllm import detect_hcu, VllmCompatibilityError

    detection = detect_hcu()
    if not detection.detected or detection.architecture not in VALIDATED_ARCHES:
        return False

    def incompatible(detail):
        raise VllmCompatibilityError('HCU official FP8 fallback ABI changed: ' + detail)

    try:
        linear = import_module('vllm.model_executor.kernels.linear')
        pytorch = import_module('vllm.model_executor.kernels.linear.scaled_mm.pytorch')
        triton = import_module('vllm.model_executor.kernels.linear.scaled_mm.triton')
        platforms = import_module('vllm.platforms')
        register = linear.register_linear_kernel
        kernel = triton.TritonFP8ScaledMMLinearKernel
        native_support = pytorch._supports_torch_fp8_scaled_mm
        platform = platforms.PlatformEnum.ROCM
    except (ImportError, AttributeError) as exc:
        incompatible(str(exc))
    if tuple(inspect.signature(register).parameters) != ('kernel_class', 'platform', 'kernel_type'):
        incompatible('register_linear_kernel(kernel_class, platform, kernel_type) required')
    if tuple(inspect.signature(kernel.apply_scaled_mm).parameters) != (
        'self', 'A', 'B', 'out_dtype', 'As', 'Bs', 'bias', 'output_shape',
    ):
        incompatible('official TritonFP8ScaledMMLinearKernel.apply_scaled_mm contract required')
    if inspect.signature(native_support).parameters:
        incompatible('zero-argument official native support predicate required')
    native_available = native_support()
    if type(native_available) is not bool:
        incompatible('official native support predicate must return bool')
    # Exit when official native support arrives; never falsify its predicate.
    if native_available:
        return False
    marker = '_hcu_official_fp8_fallback_registered'
    if getattr(linear, marker, False):
        return False

    from hcu_adapter import registry

    # Public seam appends at the end: existing tuned/vendor/native kernels win.
    register(kernel, platform, kernel_type='fp8')
    registry.register(
        op='vllm_fp8_linear', impl='official-triton',
        target='vllm.model_executor.kernels.linear.TritonFP8ScaledMMLinearKernel',
        mechanism=registry.MECH_SEAM, source=__name__,
        detail='gfx936; official register_linear_kernel, last-priority fallback; no native predicate change',
    )
    setattr(linear, marker, True)
    return True
