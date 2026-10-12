"""weak_ref_tensor — breakable-graph capture support (behavior restoration).

Upstream sglang's `compilation.weak_ref_tensor` imports this symbol from
sgl_kernel (a custom CUDA C++ TensorImpl that drops the storage refcount so
the capture mempool can reclaim per-layer intermediates between breakable-
graph segments). The pure-torch fallback here keeps the tensor strongly
referenced — semantically identical, only less memory-optimal during prefill
graph capture (per-segment intermediates stay resident). When capture memory
becomes the binding constraint, the real fix is a HIP C++ shim exposing the
same storage-decoupled TensorImpl — gap-listed, not silently wrong.
"""

import torch


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
