# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from importlib import import_module
from types import ModuleType

from hcu_adapter.bridges._sglang._publication import Binding, publish
from hcu_adapter.infra import registry

LOG = logging.getLogger("hcu.adapter")

_NUMERICS_MODULES = (
    "sglang.kernels.numerics",
    "sglang.kernels.ops.diffusion.common.numerics",
)


def warn_unsupported_arches(architectures):
    LOG.warning(
        "HCU adapter: skipping SGLang diffusion FP32 numerics "
        "(mul_rn_f32, div_rn_f32, rsqrt_approx_f32). "
        "This adapter implementation requires gfx936 on all visible HIP devices; "
        "detected: %s.",
        ", ".join(arch or "unknown" for arch in architectures),
    )


def _upstream_numerics():
    for path in _NUMERICS_MODULES:
        try:
            return import_module(path)
        except ModuleNotFoundError as exc:
            if not exc.name or not (path == exc.name or path.startswith(exc.name + ".")):
                raise
    return None


def _has_supported_devices():
    import torch

    from hcu_adapter.infra import hip_metadata

    if torch.version.hip is None:
        return False
    # The bindings are process-wide. Check every visible device before publishing.
    architectures = tuple(
        hip_metadata.device_properties(index).gcnArchName.split(":", 1)[0]
        for index in range(torch.cuda.device_count())
    )
    supported = set(architectures) == {"gfx936"}
    if architectures and not supported:
        warn_unsupported_arches(architectures)
    return supported


def install():
    if not _has_supported_devices():
        return False
    numerics = _upstream_numerics()
    if numerics is None:
        return False

    from hcu_adapter.kernels import diffusion_numerics_triton as kernels

    replacements = {
        "mul_rn_f32": kernels.mul_rn_f32,
        "div_rn_f32": kernels.div_rn_f32,
        "rsqrt_approx_f32": kernels.rsqrt_approx_f32,
    }
    originals = {name: getattr(numerics, name) for name in replacements}
    if all(originals[name] is fn for name, fn in replacements.items()):
        return False

    modules = {
        id(module): module
        for path, module in tuple(sys.modules.items())
        if path.startswith("sglang.") and isinstance(module, ModuleType)
    }
    bindings = [
        Binding.prepare(module, name, fn)
        for module in modules.values()
        for name, fn in replacements.items()
        if vars(module).get(name) is originals[name] and originals[name] is not fn
    ]

    def register():
        registry.register(
            "diffusion_fp32_numerics",
            "triton",
            target=numerics.__name__,
            mechanism=registry.MECH_REBIND,
            detail="gfx936 FP32 mul_rn, div_rn and rsqrt for cached and future SGLang imports",
        )

    publish(bindings, register)
    return True
