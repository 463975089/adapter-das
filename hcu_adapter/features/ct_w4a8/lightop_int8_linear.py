# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""LightOp token INT8 producers over native CT loading and linear execution.

Opt-in callers select this method. Native CT still creates, loads and finalizes
the weights; native RowParallelLinear still owns reduction and bias placement.
"""
from __future__ import annotations

from math import prod
from dataclasses import replace
from types import MethodType
import weakref

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonInt8ScaledMMLinearKernel
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation, expose_input_quant_key, get_input_quant_key,
)
from vllm.model_executor.layers.linear import register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization.utils.quant_utils import kInt8DynamicTokenSym

from hcu_adapter.features.dsv41.dsv41_int8_linear import HcuDSV41Int8LinearMethod
from hcu_adapter.infra.native_namespace import bind_native


def _device_supported(device):
    from vllm.platforms import current_platform
    return (device.type == 'cuda' and current_platform.device_name == 'hcu'
            and torch.cuda.get_device_properties(device).gcnArchName.split(':')[0] == 'gfx938')


def _token_quant(x):
    from lightop.quant import per_token_quant_int8
    return per_token_quant_int8(x)


def _fused_quant(x, *, output, scales):
    from lightop.activation import fuse_silu_mul_per_token_quant
    fuse_silu_mul_per_token_quant(x, output=output, scales=scales)


class _QuantOps:
    """Private native function namespace; other ops keep their native identity."""
    @staticmethod
    def scaled_int8_quant(x, scale=None, azp=None, symmetric=True):
        if (scale is None and azp is None and symmetric and x.ndim == 2
                and x.dtype == torch.bfloat16 and x.numel() and x.is_contiguous()):
            q, s = _token_quant(x)
            if (q.shape != x.shape or q.dtype != torch.int8 or q.device != x.device
                    or s.shape != (x.shape[0], 1) or s.dtype != torch.float32
                    or s.device != x.device or not q.is_contiguous() or not s.is_contiguous()):
                raise RuntimeError('LightOp token quant returned invalid activation metadata')
            return q, s, None
        return ops.scaled_int8_quant(x, scale, azp, symmetric=symmetric)

    def __getattr__(self, name):
        return getattr(ops, name)


_APPLY = bind_native(TritonInt8ScaledMMLinearKernel.apply_weights, {'ops': _QuantOps()})


@register_weight_loader_v2_supported_method
class HcuLightopInt8LinearMethod(HcuDSV41Int8LinearMethod):
    """Dynamic symmetric channel INT8, with canonical pre-quantized inputs."""

    @staticmethod
    def input_quant_key():
        return kInt8DynamicTokenSym

    def process_weights_after_loading(self, layer):
        previous = getattr(layer, '_hcu_lightop_int8_witness', None)
        empty = getattr(layer, '_hcu_int8_empty_witness', None)
        restore = None
        layer._hcu_lightop_int8_witness = None
        # The method is selected before any producer can advertise this key.
        layer.__dict__.pop('_input_quant_key', None)
        if previous is not None and empty is not None:
            from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
            info = get_layerwise_info(layer)
            params = None if info.kernel_tensors is None else info.kernel_tensors[0]
            # Core materializes checkpoint-layout tensors, then copies the
            # processed result into the original Graph storage. This is a new
            # loading lifecycle, not a second transpose of finalized weights.
            if (params is not None and params.get('weight') is empty.weight
                    and layer.weight is not empty.weight and info.loaded_weights
                    and info.load_numel_total is not None
                    and info.load_numel >= info.load_numel_total > 0
                    and tuple(layer.weight.shape) == empty.shape[::-1]
                    and layer.scheme is previous[0] and layer.scheme.kernel is previous[1]
                    and layer.scheme.kernel.apply_weights is previous[2]):
                restore = params
                layer.__dict__.pop('_hcu_int8_empty_state', None)
                layer.__dict__.pop('_hcu_int8_empty_witness', None)
                layer.scheme.kernel.apply_weights = MethodType(
                    TritonInt8ScaledMMLinearKernel.apply_weights, layer.scheme.kernel)
        super().process_weights_after_loading(layer)
        kernel = layer.scheme.kernel
        config = kernel.config
        if (type(kernel) is not TritonInt8ScaledMMLinearKernel
                or config.is_static_input_scheme or not config.input_symmetric
                or not config.is_channelwise or not _device_supported(layer.weight.device)
                or layer.weight.dtype != torch.int8):
            return
        # Detect API drift before changing the owned instance. No class/global
        # rebinding: other models and methods keep the official quantizer.
        if kernel.apply_weights.__func__ is not TritonInt8ScaledMMLinearKernel.apply_weights:
            raise RuntimeError('CT INT8 native apply ownership changed')
        binding = MethodType(_APPLY, kernel)
        kernel.apply_weights = binding
        scales = layer.weight_scale
        if restore is not None:
            weight, scales = restore['weight'], restore['weight_scale']
            current = layer._hcu_int8_empty_witness
            if (weight.shape != layer.weight.shape or scales.shape != layer.weight_scale.shape
                    or weight.dtype != layer.weight.dtype or weight.device != layer.weight.device):
                raise RuntimeError('CT INT8 reload storage metadata changed')
            layer._hcu_int8_empty_witness = replace(current, weight=weight,
                shape=tuple(weight.shape), stride=tuple(weight.stride()),
                data_ptr=weight.data_ptr(), device=weight.device, dtype=weight.dtype)
        layer._hcu_lightop_int8_witness = (layer.scheme, kernel, binding, scales)
        expose_input_quant_key(layer, self)

    def _check_witness(self, layer):
        witness = getattr(layer, '_hcu_lightop_int8_witness', None)
        empty = getattr(layer, '_hcu_int8_empty_witness', None)
        if (witness is None or empty is None
                or getattr(layer, '_hcu_int8_empty_state', None) != 'ready'):
            raise RuntimeError('INT8 activation requires finalized postprocess witness')
        scheme, kernel, binding, scales = witness
        w = layer.weight
        if (layer.scheme is not scheme or scheme.kernel is not kernel
                or kernel.apply_weights is not binding or layer.weight_scale is not scales
                or w is not empty.weight or tuple(w.shape) != empty.shape
                or tuple(w.stride()) != empty.stride or w.data_ptr() != empty.data_ptr
                or w.device != empty.device or w.dtype != empty.dtype
                or kernel.config.is_static_input_scheme or not kernel.config.input_symmetric
                or not kernel.config.is_channelwise):
            raise RuntimeError('INT8 activation postprocess witness no longer matches the layer')

    def apply(self, layer, x, bias=None):
        if not isinstance(x, QuantizedActivation):
            return super().apply(layer, x, bias)
        if getattr(layer, '_hcu_lightop_int8_witness', None) is None:
            return super().apply(layer, x, bias)
        self._check_witness(layer)
        w, ws, ins, inz, azp = layer.scheme.kernel._get_layer_params(layer)
        shape = x.orig_shape
        if (get_input_quant_key(layer) != kInt8DynamicTokenSym
                or x.quant_key != kInt8DynamicTokenSym or x.orig_dtype != torch.bfloat16
                or len(shape) != 2 or shape[-1] != w.shape[0]
                or not isinstance(x.data, torch.Tensor) or not isinstance(x.scale, torch.Tensor)
                or x.data.shape != shape or x.data.dtype != torch.int8
                or x.scale.shape != (prod(shape[:-1]), 1) or x.scale.dtype != torch.float32
                or x.data.device != w.device or x.scale.device != w.device
                or not x.data.is_contiguous() or not x.scale.is_contiguous()
                or ins is not None or inz is not None or azp is not None
                or ws.dtype != torch.float32 or ws.device != w.device
                or ws.shape not in ((w.shape[1],), (w.shape[1], 1)) or not ws.is_contiguous()):
            raise ValueError('INT8 activation metadata does not match the consumer contract')
        if bias is not None and (bias.shape != (w.shape[1],) or bias.dtype != x.orig_dtype
                                 or bias.device != w.device or not bias.is_contiguous()):
            raise ValueError('INT8 activation bias metadata does not match the consumer contract')
        if shape[0] == 0:
            return torch.empty((0, w.shape[1]), device=w.device, dtype=x.orig_dtype)
        return torch.ops.vllm.int8_scaled_mm_func(x.data, w, x.scale, ws, x.orig_dtype, bias)


class FusedSiluInt8Producer(torch.nn.Module):
    """Silu/mul quantization only while the complete consumer accepts INT8.

    The consumer is not registered as a child module: its state_dict and native
    parameter names must remain unchanged, including for checkpoint reload.
    """
    def __init__(self, original, consumer):
        super().__init__()
        self.original = original
        self._consumer = weakref.ref(consumer)

    def forward(self, x):
        consumer = self._consumer()
        if (consumer is None or get_input_quant_key(consumer) != kInt8DynamicTokenSym
                or x.dtype != torch.bfloat16 or x.ndim != 2 or not x.numel()
                or not x.is_contiguous() or x.shape[-1] % 2):
            return self.original(x)
        shape = torch.Size((x.shape[0], x.shape[-1] // 2))
        q = torch.empty(shape, device=x.device, dtype=torch.int8)
        s = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
        _fused_quant(x, output=q, scales=s)
        return QuantizedActivation(q, s, x.dtype, shape, kInt8DynamicTokenSym)
