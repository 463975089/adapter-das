# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Probe for native sgl_kernel registrations, BACKEND_CATALOG "cpp" tier.

Two supported shapes, in priority order:

1. ANY native library that already registered ``sgl_kernel::*`` ops in
   this process (checked via the dispatcher table — import-free, so it
   also sees an official wheel's common_ops.so that exotic dev containers
   may have loaded). Those ops are used as-is.
2. This package's optional ``sgl_kernel/native_ops*.so`` (built by
   setup_hip.py in a DTK container; pure-python installs simply lack it).

Order matters for safety: when the namespace is ALREADY registered we
must not import our .so — its TORCH_LIBRARY_FRAGMENT would attempt
duplicate (op, key) impl registrations and abort the process (C++
terminate, uncatchable from python).

Tier order in every consumer: native (if present) -> triton -> torch emu.
Callers must keep their own precondition guards (contiguity, dtype, layout
constraints) BEFORE dispatching to native — the native TORCH_CHECKs mirror
upstream and raise RuntimeError on violation, while the triton/emu tiers
accept a slightly wider face; a native-only crash on an input the fallback
tiers handle would be a tier divergence.
"""

from __future__ import annotations

import torch

# ops this package's cpp tier provides (csrc/common_extension_hcu.cc)
_PROBE_OPS = (
    "merge_state_v2",
    "concat_mla_k",
    "concat_mla_absorb_q",
    "copy_to_gpu_no_ce",
    "sgl_per_token_group_quant_8bit",
    "sgl_per_token_group_quant_8bit_v2",
)

_STATE = None  # None = unprobed, then True/False


def _has_kernel(name: str) -> bool:
    try:
        return torch._C._dispatch_has_kernel(f"sgl_kernel::{name}")
    except Exception:
        return False


def _load_own_so() -> bool:
    """dlopen sgl_kernel/native_ops*.so shipped beside this file.

    ctypes.CDLL, not import: the module is a pure TORCH_LIBRARY library
    (registrations run from static initializers at dlopen) and defines no
    PyInit entry point."""
    import ctypes
    import glob
    import os

    here = os.path.dirname(__file__)
    for so in sorted(glob.glob(os.path.join(here, "native_ops*.so"))):
        try:
            ctypes.CDLL(so)
        except OSError:
            continue
        if any(_has_kernel(n) for n in _PROBE_OPS):
            return True
    return False


def native_available() -> bool:
    """True iff any sgl_kernel native op is registered in this process."""
    global _STATE
    if _STATE is None:
        if any(_has_kernel(n) for n in _PROBE_OPS):
            _STATE = True  # someone already registered the namespace
        else:
            _STATE = _load_own_so()
    return _STATE


def native_op(name: str):
    """torch.ops.sgl_kernel.<name> if natively registered, else None.

    Per-op granularity: in mixed shapes (an official wheel providing only
    part of the face) missing ops resolve to None and callers fall back
    to the triton/torch tiers."""
    if native_available() and _has_kernel(name):
        return getattr(torch.ops.sgl_kernel, name)
    return None
