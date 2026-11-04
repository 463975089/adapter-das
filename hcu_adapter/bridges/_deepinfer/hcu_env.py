# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Registered debug overrides; behavioral bridges use resolve(), not raw reads."""

import os
import sys

from hcu_adapter import hcu_env as environment
from hcu_adapter.kill_switch import disabled_ops

_NAMES = ("HCU_WORKER_CONTEXT", "HCU_DISABLE_OPS", "HCU_ADAPTER_DUMP", "HCU_CONFIG",
          "HCU_PLATFORM_SEMANTICS", "HCU_BF16_TFLOPS",
          "HCU_LIGHTOP_OPS", "HCU_NATIVE_FLASHMLA")
REGISTRY = {name: environment.REGISTRY[name] for name in _NAMES}


def resolve(name):
    if name == "HCU_DISABLE_OPS":
        return tuple(disabled_ops())
    value, _ = environment.resolve(name)
    if name == "HCU_PLATFORM_SEMANTICS":
        return "gpu" if value == "cuda" else value
    if name in ("HCU_LIGHTOP_OPS", "HCU_NATIVE_FLASHMLA"):
        return int(value)
    if name == "HCU_BF16_TFLOPS":
        return float(value)
    return value


def lightop_ops_enabled():
    value = resolve('HCU_LIGHTOP_OPS')
    if value not in (0, 1):
        raise ValueError('HCU_LIGHTOP_OPS must be 0 or 1')
    return bool(value)


def native_flashmla_enabled():
    value = resolve('HCU_NATIVE_FLASHMLA')
    if value not in (0, 1):
        raise ValueError('HCU_NATIVE_FLASHMLA must be 0 or 1')
    return bool(value)


def set_worker_context(value):
    """Internal inherited provider configuration; users do not hand-author it."""
    os.environ.update({"HCU_WORKER_CONTEXT": value})


def align_device_visibility(physical_device_count):
    """Translate the upstream engine's physical ordinal mask before HIP init.

    The upstream DP launcher narrows CUDA_VISIBLE_DEVICES only. An inherited
    wider HIP_VISIBLE_DEVICES would otherwise keep every DP worker on HIP
    device zero. Never expand an explicit HIP allocation or change an already
    initialized runtime's device order.
    """
    from hcu_adapter import registry

    cuda_mask = os.getenv('CUDA_VISIBLE_DEVICES')
    hip_mask = os.getenv('HIP_VISIBLE_DEVICES')
    result = dict(cuda=cuda_mask, hip_before=hip_mask, hip_after=hip_mask, changed=False)
    if cuda_mask is not None:
        if os.getenv('ROCR_VISIBLE_DEVICES') is not None:
            raise NotImplementedError('HCU ordinal mask alignment with ROCR_VISIBLE_DEVICES is not qualified')

        def ordinals(mask):
            if mask in ('', '-1'):
                return ()
            try:
                values = tuple(int(part) for part in mask.split(','))
            except ValueError as error:
                raise ValueError('HCU visibility requires physical numeric GPU ordinals') from error
            if len(set(values)) != len(values) or any(not 0 <= value < physical_device_count for value in values):
                raise ValueError('HCU device mask has duplicate or out-of-inventory ordinals')
            return values

        desired = ordinals(cuda_mask)
        existing = None if hip_mask is None else ordinals(hip_mask)
        if existing is not None and not set(desired).issubset(existing):
            raise ValueError('Upstream CUDA device assignment would expand the explicit HIP device allocation')
        if existing != desired:
            torch = sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():
                raise RuntimeError('HCU device masks must be aligned before GPU initialization')
            chosen = ','.join(map(str, desired)) if desired else cuda_mask
            os.environ.update(HIP_VISIBLE_DEVICES=chosen)
            result.update(hip_after=chosen, changed=True)
            print(f'HCU adapter visibility: HIP_VISIBLE_DEVICES={chosen!r} from upstream CUDA assignment', flush=True)
    registry.register('runtime.device_visibility', 'hcu_adapter.bridges._deepinfer.hcu_env.align_device_visibility',
                      'HIP_VISIBLE_DEVICES before provider installation', 'seam')
    return result


def apply_upstream_defaults(allow_cuda_graph):
    """Compatible defaults only; explicit upstream choices remain authoritative."""
    from hcu_adapter import registry

    defaults = {'DEEPINFER_BATCH_INVARIANT': '0', 'DEEPINFER_ENGRAM_FORCE_GPU_STORAGE': '1'}
    if not allow_cuda_graph:
        defaults.update(DEEPINFER_ENFORCE_EAGER='1', DEEPINFER_DFLASH_ENFORCE_EAGER='1')
    resolved = {}
    for name, default in defaults.items():
        source = 'environment' if name in os.environ else 'adapter-default'
        resolved[name] = dict(value=os.environ.setdefault(name, default), source=source)
        upstream = sys.modules.get('deepinfer.environ')
        if upstream is not None:
            upstream.rearm(name)
    registry.register('runtime.upstream_defaults', 'hcu_adapter.bridges._deepinfer.hcu_env.apply_upstream_defaults',
                      'upstream setdefault compatibility policy', 'seam')
    print('HCU adapter upstream defaults: ' + ' '.join(
        f'{name}={entry["value"]}({entry["source"]})' for name, entry in resolved.items()), flush=True)
    return resolved
