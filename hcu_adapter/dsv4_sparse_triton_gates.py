# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DSV4.1 sparse indexer triton-gate needle (nmz26/gfx938, 2026-10-22).

``sglang/srt/layers/attention/dsv4/dsv41_sparse.py`` gates its two fused
triton kernels behind::

    _fused_dsv41_kernels = lambda x: x.is_cuda and (
        torch.version.cuda is not None or is_gfx95_supported())

On the DTK build torch.version.cuda is None (hip 6.3) and gfx938 fails
is_gfx95_supported, so every decode step takes the torch fallbacks instead:

- ``_rope_fq4``: complex64 rope_tail chain (float copy + contiguous +
  complex mul + cat) then a separate fake-quant pass — the trace's 91/step
  Q-rope kernel family;
- ``RMSNorm.forward``: the torch rsqrt formula instead of ``rmsnorm_fp32``.

Both gated kernels are pure triton (no CUDA JIT under the seam) and the
upstream comment claims bitwise equality with the fallbacks; verified
bitwise on gfx938 (tmp_fetch/parity_sparse_triton.py, 10/10 cases: rope+fp4
2D/3D D=512/256 rd=64 compressed_kv positions-table, rmsnorm D=128/512
bf16/fp32). Rebind the module-global predicate — call sites in the same
module resolve it at runtime.

Gate: ``HCU_DSV41_SPARSE_TRITON`` (default on; ``=0`` restores the torch
chain). Only armed when the stock predicate is platform-wide False (hip
runtime, no CUDA version), so real CUDA / gfx95 boxes are untouched.
"""

from __future__ import annotations

import logging
import os

import torch

LOG = logging.getLogger("hcu.adapter")


def install() -> None:
    if os.environ.get("HCU_DSV41_SPARSE_TRITON") == "0":
        return
    if torch.version.hip is None:
        return  # hardware truth: only meaningful on an HIP runtime
    if torch.version.cuda is not None:
        return  # stock predicate already True everywhere; nothing to flip

    import sglang.srt.layers.attention.dsv4.dsv41_sparse as _sparse

    if getattr(_sparse, "_hcu_sparse_triton_gate", False):
        return  # idempotent across double install

    _sparse._fused_dsv41_kernels = lambda x: bool(x.is_cuda)
    _sparse._hcu_sparse_triton_gate = True
    LOG.warning(
        "HCU adapter: DSV4.1 sparse rope/norm triton gates enabled "
        "(rope_tail_fake_quant_fp4 + rmsnorm_fp32; HCU_DSV41_SPARSE_TRITON=0"
        " restores the torch fallbacks)"
    )
