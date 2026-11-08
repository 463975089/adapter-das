# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Install a narrow execution bridge for the official FP8/INT8 Triton providers.

Public linear registration appends providers and cannot replace the existing
INT8 fallback's execution backend. Bind its shared module global instead of
copying native apply/load/quantization or changing selector priority. All math
and metadata admission remain in the framework-free kernel module.
"""
from __future__ import annotations

from importlib import import_module
import inspect
import logging

_LOG = logging.getLogger('hcu.adapter')
_MARKER = '_hcu_lightop_scaled_mm'


def install() -> bool:
    from hcu_adapter import hcu_config, registry
    from hcu_adapter.bridges.vllm import detect_hcu, VllmCompatibilityError
    from hcu_adapter.kill_switch import op_disabled

    detection = detect_hcu()
    if not detection.detected or detection.architecture not in ('gfx936', 'gfx938'):
        return False
    choice = hcu_config.load()['ops'].get('vllm_scaled_mm', 'auto')
    disabled = op_disabled('vllm_scaled_mm')
    module = import_module('vllm.model_executor.kernels.linear.scaled_mm.triton')
    installed = getattr(module, _MARKER, None)
    if installed is not None:
        binding, policy = installed
        if module.triton_scaled_mm is not binding or policy != (choice, disabled):
            raise VllmCompatibilityError('HCU scaled-MM binding/policy changed; start a fresh process')
        return False
    if disabled or choice == 'triton':
        return False
    if choice not in ('auto', 'lightop'):
        raise VllmCompatibilityError('Unsupported vllm_scaled_mm backend: ' + choice)
    try:
        vendor = import_module('lightop.gemm_ops').hipblaslt_w8a8_channelwise_gemm
    except (ImportError, AttributeError) as exc:
        if choice == 'auto':
            _LOG.info('HCU scaled-MM: LightOp unavailable; retain official Triton')
            return False
        raise VllmCompatibilityError('LightOp scaled-MM explicitly requested but unavailable') from exc
    expected = ('a','b','scale_a','scale_b','m','n','k','transpose_flag','out_dtype','bias')
    if tuple(inspect.signature(vendor).parameters) != expected:
        raise VllmCompatibilityError('LightOp channelwise GEMM ABI changed')
    original = module.triton_scaled_mm
    parameters = tuple(inspect.signature(original).parameters.values())
    if (tuple(p.name for p in parameters[:6]) != ('input','weight','scale_a','scale_b','out_dtype','bias')
            or any(p.default is inspect.Parameter.empty for p in parameters[6:])):
        raise VllmCompatibilityError('Official Triton scaled-MM ABI changed')
    fp8 = module.TritonFP8ScaledMMLinearKernel.apply_scaled_mm
    int8 = module._int8_scaled_mm_func
    if any(fn.__globals__ is not vars(module) or 'triton_scaled_mm' not in fn.__code__.co_names
           for fn in (fp8, int8)):
        raise VllmCompatibilityError('Official FP8/INT8 shared GEMM call site changed')

    import torch
    from hcu_adapter.kernels.scaled_mm_lightop import scaled_mm, supported

    architecture = detection.architecture

    # Opaque to Dynamo; actual dispatch uses concrete runtime dimensions. The
    # original Triton INT8 custom op remains intact and delegates here as well.
    def dispatch(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor,
                 Bs: torch.Tensor, out_dtype: torch.dtype,
                 bias: torch.Tensor | None = None) -> torch.Tensor:
        if supported(A, B, As, Bs, out_dtype, bias, architecture):
            return scaled_mm(A, B, As, Bs, out_dtype, bias, gemm=vendor)
        return original(A, B, As, Bs, out_dtype, bias)

    def fake(A, B, As, Bs, out_dtype, bias=None):
        return torch.empty((A.shape[0], B.shape[1]), dtype=out_dtype, device=A.device)

    op = torch.library.custom_op(
        'hcu_adapter::vllm_lightop_scaled_mm', dispatch, mutates_args=(),
        schema='(Tensor A, Tensor B, Tensor As, Tensor Bs, ScalarType out_dtype, Tensor? bias=None) -> Tensor',
    )
    op.register_fake(fake)

    def routed(input, weight, scale_a, scale_b, out_dtype, bias=None, *args, **kwargs):
        if args or kwargs:
            return original(input, weight, scale_a, scale_b, out_dtype, bias, *args, **kwargs)
        return op(input, weight, scale_a, scale_b, out_dtype, bias)

    registry.register(
        op='vllm_scaled_mm', impl='lightop',
        target=module.__name__ + '.triton_scaled_mm',
        mechanism=registry.MECH_REBIND, source=__name__,
        detail=f'{architecture}; native FP8/INT8 providers and loading preserved; '
               'token/channel LightOp with metadata fallback to official Triton',
    )
    module.triton_scaled_mm = routed
    setattr(module, _MARKER, (routed, (choice, disabled)))
    _LOG.info('HCU scaled-MM: LightOp enabled on %s; unsupported contracts use official Triton', architecture)
    return True
