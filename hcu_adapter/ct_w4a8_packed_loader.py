# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Generic packed CT W4A8 parameter creation and native loader seam.

Creating the packed parameters and letting the native RoutedExperts loader fill
its shards is ordinary compressed-tensors loading behaviour; it belongs to the
adapter. The HCU HIPC representation itself (nibble reorder + XOR, the /16 scale
domain, guard allocation and the N32 view shape) is plugin-private and stays in
the plugin, which subclasses this base with its conversion step.
"""
from __future__ import annotations

import functools

import torch
from torch.nn import Parameter
from vllm.model_executor.utils import set_weight_attrs


class PackedCTParameters:
    """Own raw CT parameters and a single checked transition to packed storage."""

    def __init__(self, layer, num_experts, hidden_size, intermediate_size, weight_loader):
        if any(type(v) is not int or v <= 0 for v in (num_experts, hidden_size, intermediate_size)):
            raise ValueError("CT W4A8 dimensions must be positive integers")
        if hidden_size % 512 or intermediate_size % 256:
            raise ValueError("CT W4A8 requires H divisible by 512 and I by 256")
        if not callable(weight_loader):
            raise ValueError("CT W4A8 requires the native expert weight loader")
        self.layer = layer
        self.shape = num_experts, hidden_size, intermediate_size
        self.state = "loaded"
        self.weights = None
        self.scales = None
        self._signature_ready = None

        @functools.wraps(weight_loader)
        def load_once(*args, **kwargs):
            if self.state != "loaded":
                raise RuntimeError("CT W4A8 reload after packing is unsupported; restart worker")
            return weight_loader(*args, **kwargs)

        attrs = {"weight_loader": load_once, "is_transposed": False, "quant_method": "channel"}
        for name, shape, dtype in (
            ("w13_weight_packed", (num_experts, 2 * intermediate_size, hidden_size // 8), torch.int32),
            ("w2_weight_packed", (num_experts, hidden_size, intermediate_size // 8), torch.int32),
            ("w13_weight_scale", (num_experts, 2 * intermediate_size, 1), torch.float32),
            ("w2_weight_scale", (num_experts, hidden_size, 1), torch.float32),
            ("w13_weight_shape", (num_experts, 2), torch.int64),
            ("w2_weight_shape", (num_experts, 2), torch.int64),
        ):
            parameter = Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, parameter)
            set_weight_attrs(parameter, attrs)

    def signature(self):
        return tuple(
            (id(p), p.data_ptr(), p._version, tuple(p.shape), p.dtype, p.stride())
            for p in (
                self.layer.w13_weight_packed,
                self.layer.w2_weight_packed,
                self.layer.w13_weight_scale,
                self.layer.w2_weight_scale,
            )
        )
