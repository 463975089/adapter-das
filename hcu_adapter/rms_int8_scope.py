# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""A per-call handoff between one owned attention norm and native CT GEMMs.

The BF16 normalized tensor remains the public value. The extra INT8 codes and
FP32 scales live only inside the owned decoder call, matched by tensor identity.
No tensor attributes, process-global activation cache or native rebinding.
"""
import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch


@dataclass
class _Call:
    value: object = None
    codes: object = None
    scales: object = None


_CURRENT = ContextVar('hcu_v41_rms_int8_call', default=None)


@contextmanager
def attention_norm_scope():
    token = _CURRENT.set(_Call())
    try:
        yield
    finally:
        _CURRENT.reset(token)


def active():
    return _CURRENT.get() is not None


def publish(value, codes, scales):
    call = _CURRENT.get()
    if call is None:
        raise RuntimeError('RMS/INT8 handoff requires an owned decoder call')
    if (value.ndim != 2 or value.dtype != torch.bfloat16
            or codes.shape != value.shape or codes.dtype != torch.int8
            or scales.shape != (value.shape[0], 1) or scales.dtype != torch.float32
            or codes.device != value.device or scales.device != value.device
            or not codes.is_contiguous() or not scales.is_contiguous()):
        raise ValueError('RMS/INT8 handoff metadata mismatch')
    call.value, call.codes, call.scales = value, codes, scales


def lookup(value):
    call = _CURRENT.get()
    if call is not None and call.value is value:
        return call.codes, call.scales
    return None


def supported(x, weight, eps):
    return (
        isinstance(x, torch.Tensor) and isinstance(weight, torch.Tensor)
        and x.device.type == 'cuda' and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim == 2 and x.shape[0] > 0 and x.shape[1] == 5120
        and tuple(weight.shape) == (5120,) and x.is_contiguous()
        and weight.is_contiguous() and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and isinstance(eps, (int, float)) and math.isfinite(eps) and eps >= 0
        and not torch.is_grad_enabled() and not torch.is_autocast_enabled('cuda')
    )
