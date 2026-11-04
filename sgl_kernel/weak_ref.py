# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""weak_ref_tensor — breakable-graph capture support (behavior restoration).

Upstream sglang's `compilation.weak_ref_tensor` imports this symbol from
sgl_kernel: a custom C++ TensorImpl that does NOT hold a storage reference,
so breakable-graph segments only pin their intermediates via the mempool
use-count (blocks stay reserved for replay addresses) instead of via
Python refs. That decoupling is what lets later capture buckets REUSE the
shared graph pool's blocks — without it every bucket's transient peak stays
resident and the 58-bucket prefill table OOMs partway through capture
(diagnosed 2026-10-14: 0.6B died at bucket 25/58 with each of the first
buckets holding 1.7-6 GB resident).

Primary route: the DTK vendor sgl_kernel wheel ships the SAME compiled
C++ op (`_Z15weak_ref_tensorRKN2at6TensorE` in common_ops.so, built under
HIP toolchain — device-independent host-side TensorImpl surgery). Detect
it once at import and re-export verbatim; the pure-torch view_as fallback
below stays only for wheels without the symbol (semantically safe, memory
pessimistic — capture-time pinning, gap-listed).
"""

import torch


def _load_vendor_impl():
    """Prefer the compiled C++ op (storage-decoupled TensorImpl).

    A vendor sgl_kernel wheel (DTK ships one) may be shadowed by OUR package
    of the same name. Its C++ op registers torch.ops.sgl_kernel.weak_ref_
    tensor from common_ops*.so, but the .so is a pybind module whose
    PyInit name is fixed ("common_ops") — load it under exactly that name
    via importlib with the vendor file location, then read the registered
    torch.ops symbol. Never import the vendor package __init__ (same-name
    shadowing would re-enter this file).
    """
    import glob
    import importlib
    import importlib.util
    import os
    import sys

    import torch

    if "sgl_kernel.common_ops" in sys.modules:
        mod = sys.modules["sgl_kernel.common_ops"]
    else:
        here_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        so_candidates = []
        for entry in sys.path:
            if not entry or not os.path.isabs(entry):
                continue
            pkg = os.path.join(entry, "sgl_kernel")
            if os.path.normpath(pkg) == os.path.normpath(here_pkg):
                continue  # our own tree — skip
            so_candidates += glob.glob(os.path.join(pkg, "common_ops*.so"))
        if not so_candidates:
            return None
        so = sorted(so_candidates)[0]
        spec = importlib.util.spec_from_file_location(
            "sgl_kernel.common_ops", so
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["sgl_kernel.common_ops"] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop("sgl_kernel.common_ops", None)
            return None

    fn = None
    try:
        fn = torch.ops.sgl_kernel.weak_ref_tensor
        fn(torch.zeros(1, device="cuda"))
    except Exception:
        fn = getattr(mod, "weak_ref_tensor", None)
        if fn is None:
            return None
        try:
            fn(torch.zeros(1, device="cuda"))
        except Exception:
            return None

    def _impl(t):
        return fn(t) if isinstance(t, torch.Tensor) else t

    return _impl


_vendor_weak_ref_tensor = _load_vendor_impl()


if _vendor_weak_ref_tensor is not None:
    weak_ref_tensor = _vendor_weak_ref_tensor

    def weak_ref_tensor_lazy(t):  # keep signature parity for docs
        return weak_ref_tensor(t)

else:

    def weak_ref_tensor(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"weak_ref_tensor expects a Tensor, got {type(t)!r}")
        if t.numel() == 0:
            return t
        # Strong-reference stand-in: view with identical shape/stride/layout.
        # (The upstream C++ op returns a TensorImpl sharing the Storage without
        # incrementing its refcount; nothing in torch core can express that, and
        # faking it via storage surgery risks use-after-free inside capture.)
        return t.view_as(t)


def weak_ref_tensors(tensors):
    """Mirror upstream convenience wrapper (tensor / list / tuple)."""
    import torch as _t

    if isinstance(tensors, _t.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    raise ValueError(
        "weak_ref_tensors expects a Tensor or list/tuple of Tensors"
    )
