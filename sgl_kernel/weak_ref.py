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
buckets holding 1.7-6 GB resident; 2026-11-03 HY4 78-layer died at bucket
832 the same way).

Route, decided once at import (`ROUTE`):

1. "vendor"  — a vendor sgl_kernel common_ops*.so ships the compiled C++
   op (`torch.ops.sgl_kernel.weak_ref_tensor`); re-export it verbatim.
2. "torch"   — no .so: rebuild the same non-owning tensor from torch's own
   primitives. The C++ op is `at::from_blob(data_ptr, sizes, strides,
   options)` with no deleter; `torch._C._construct_storage_from_data_pointer`
   builds exactly such a storage and
   `torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata` wraps it
   (the pair torch inductor's cudagraph_trees uses for the same purpose).
   Both are private torch._C entry points; admission checks only that they
   exist. Like the vendor lookup, import allocates nothing on the device, so
   no CUDA context is created before the worker selects its device. Pool
   reuse is covered by tests/kernels/test_weak_ref_decouple.py on a device.
3. "view_as" — the two torch entry points are missing: strong-reference
   stand-in. Semantically safe, memory pessimistic (capture-time pinning,
   gap-listed).
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

    # Inspect registration without allocating a CUDA test tensor. The vendor
    # op has no CPU kernel, so a CPU tensor probe would silently lose the
    # storage-decoupled CUDA implementation and retain graph intermediates.
    fn = None
    try:
        candidate = torch.ops.sgl_kernel.weak_ref_tensor
        if any(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "sgl_kernel::weak_ref_tensor", key
            )
            for key in ("CUDA", "CompositeExplicitAutograd", "CompositeImplicitAutograd")
        ):
            fn = candidate
    except (AttributeError, RuntimeError):
        pass
    if fn is None:
        fn = getattr(mod, "weak_ref_tensor", None)
    if not callable(fn):
        return None

    def _impl(t):
        return fn(t) if isinstance(t, torch.Tensor) else t

    return _impl


_vendor_weak_ref_tensor = _load_vendor_impl()


def _torch_weak_ref_impl(t):
    """Non-owning alias of `t` built from torch primitives (C++ op parity).

    Same data_ptr / sizes / strides / dtype / device, storage_offset 0 and
    storage nbytes = (1 + sum((size-1)*stride)) * itemsize — what
    at::from_blob computes. The storage carries no deleter and no
    reference to the original allocation.
    """
    if not isinstance(t, torch.Tensor):
        return t
    if t.numel() == 0:
        return t
    if t.device.type != "cuda":
        raise ValueError("weak_ref_tensor expects a CUDA tensor")
    span = 1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride()))
    storage = torch._C._construct_storage_from_data_pointer(
        t.data_ptr(), t.device, span * t.element_size()
    )
    meta = {
        "size": t.shape,
        "stride": t.stride(),
        "dtype": t.dtype,
        "device": t.device,
        "storage_offset": 0,
    }
    return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(
        meta, storage
    )


def _probe_pool_reuse(fn, nbytes=1 << 22):
    """True when `fn` leaves a capture-pool block reusable (test/diagnostic
    helper; never run at import).

    Segment 1 allocates a block in a private graph pool, `fn` takes a weak
    view and python drops the original; segment 2 in the same pool must
    land on the same block, and the weak view must still alias it.
    A strong reference (view_as) fails this.
    """
    import gc

    pool = torch.cuda.graph_pool_handle()
    n = nbytes // 4
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1, pool=pool):
        x = torch.empty(n, device="cuda")
        x.fill_(1.0)
    w = fn(x)
    ptr = x.data_ptr()
    if w.data_ptr() != ptr or w.shape != x.shape or w.stride() != x.stride():
        return False
    del x
    gc.collect()
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2, pool=pool):
        y = torch.empty(n, device="cuda")
        y.fill_(2.0)
    reused = y.data_ptr() == ptr
    del w, y, g1, g2
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return reused


def _load_torch_impl():
    """Admit the torch route when both private entry points exist.

    Static check only: no device allocation or capture at import time."""
    if not (
        hasattr(torch._C, "_construct_storage_from_data_pointer")
        and hasattr(torch._C, "_construct_CUDA_Tensor_From_Storage_And_Metadata")
    ):
        return None, "torch._C non-owning storage entry points missing"
    return _torch_weak_ref_impl, None


def _view_as_weak_ref_tensor(t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"weak_ref_tensor expects a Tensor, got {type(t)!r}")
    if t.numel() == 0:
        return t
    # Strong-reference stand-in: view with identical shape/stride/layout.
    # Pins the storage, so capture buckets stop sharing pool blocks.
    return t.view_as(t)


_torch_weak_ref_tensor = None
_degrade_reason = None
if _vendor_weak_ref_tensor is not None:
    ROUTE = "vendor"
    weak_ref_tensor = _vendor_weak_ref_tensor
else:
    _torch_weak_ref_tensor, _degrade_reason = _load_torch_impl()
    if _torch_weak_ref_tensor is not None:
        ROUTE = "torch"
        weak_ref_tensor = _torch_weak_ref_tensor
    else:
        import logging

        ROUTE = "view_as"
        weak_ref_tensor = _view_as_weak_ref_tensor
        logging.getLogger("hcu.adapter").warning(
            "weak_ref_tensor: no vendor common_ops.so and torch route "
            "unavailable (%s); using view_as -- breakable prefill graph "
            "capture will pin every bucket (gap list)",
            _degrade_reason,
        )


def weak_ref_tensor_lazy(t):  # keep signature parity for docs
    return weak_ref_tensor(t)


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
