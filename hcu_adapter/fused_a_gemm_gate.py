# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V3 fused QKV-A GEMM: keep the CUDA-only min-latency path off HIP.

``sglang.kernels.ops.gemm.fused_a_gemm`` answers "may this layer use the
fused A-GEMM kernel" from three facts: bf16 weight, tile-divisible shape,
and ``is_cuda() and get_device_sm() >= 90``. The platform seam answers CUDA
at the model layer on purpose, so on gfx93x the predicate is True and
``prepare_qkv_latent`` / ``q_b_proj_forward`` (deepseek_v2) and the latent
MLP (nemotron_h) route through ``linear_with_fused_a_gemm``.

Both JIT kernels behind it are CUDA-only. The overlay
``csrc/gemm/dsv3_fused_a_gemm.cuh`` is TensorRT-LLM's Hopper kernel with
its whole body inside ``__CUDA_ARCH__ >= 900`` — nothing is emitted for
HIP — and its host launcher stages ~192 KB of dynamic LDS
(``smem_stage_budget = 192 * 1024``) where this device caps a block at
64 KB. The first decode graph capture therefore dies in
``sglang::host::runtime::set_max_dynamic_smem``
(``runtime.cuh:293`` -> ``hipFuncSetAttribute`` invalid argument), logged
as "Capture cuda graph failed" (Kimi-K3.1, 2026-11-11).

The eligibility predicate is the honest seam: it asks whether this layer
may be served by that kernel, and here the answer is no. Rebinding it lets
each layer take its own linear — the path the model already uses for every
other shape — instead of reimplementing the dispatch. The predicate is
patched in its own module and, for importers that already bound it by name
at import time, in their globals too; importers not yet loaded
(``nemotron_h`` imports the name under ``_is_cuda``) get a post-load hook.
"""

import logging

LOG = logging.getLogger("hcu.adapter")

_MARK = "_hcu_fused_a_gemm_gate"
_NAME = "fused_a_gemm_weight_eligible"
_TARGET = "sglang.kernels.ops.gemm.fused_a_gemm"
_IMPORTERS = (
    "sglang.srt.models.deepseek_v2",
    "sglang.srt.models.nemotron_h",
)


def _ineligible(layer, *, backend=None):
    return False


_ineligible.__dict__[_MARK] = True


def _patch(module):
    """Install the shared replacement where `module` holds the original."""
    current = getattr(module, _NAME, None)
    if current is None or current is _ineligible:
        return False
    if getattr(current, _MARK, False):
        return False
    _ineligible.__wrapped__ = getattr(current, "__wrapped__", current)
    setattr(module, _NAME, _ineligible)
    return True


def install():
    import importlib
    import sys

    import torch

    if torch.version.hip is None:
        return  # hardware truth; the kernel is legitimate on CUDA SM90+

    module = sys.modules.get(_TARGET) or importlib.import_module(_TARGET)
    if not _patch(module):
        return

    import hcu_adapter.registry as _reg

    # Validate registration before publishing to the importers: a rejected
    # registration must not leave an active but unreported gate.
    _reg.register(
        "fused_a_gemm",
        "cuda-only-kernel-off",
        target=_TARGET + "." + _NAME,
        mechanism=_reg.MECH_REBIND,
        detail=(
            "HIP: SM90-guarded JIT kernel + ~192KB dynamic LDS over a 64KB"
            " block cap; layers take their own linear"
        ),
    )

    rebound = [_TARGET]
    late = []
    for name in _IMPORTERS:
        importer = sys.modules.get(name)
        if importer is None:
            late.append(name)
        elif _patch(importer):
            rebound.append(name)
    if late:
        from ._base import LibStubFinder, register_finder

        for name in late:
            register_finder(LibStubFinder(name, post_hooks={name: _patch}))
    LOG.warning(
        "HCU adapter: fused QKV-A GEMM min-latency path off on HIP"
        " (CUDA-only kernel; rebound in %s%s)",
        ", ".join(rebound),
        "; late import hook for " + ", ".join(late) if late else "",
    )
