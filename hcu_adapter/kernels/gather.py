"""Backend facade（门面）: Validation and backend selection for gather; no numerical implementation."""

from .gather_emu import gather as _torch_gather


def gather(x, index, out, *, backend):
    import torch

    if not x.is_cuda or x.device != index.device or x.device != out.device:
        raise ValueError("gather tensors must share one HCU device")
    if x.ndim != 2 or out.shape != (index.numel(), x.shape[1]):
        raise ValueError("Expected byte matrices x[N,D] and out[M,D]")
    if x.dtype != torch.int8 or out.dtype != torch.int8:
        raise TypeError("The upstream gather factory operates on int8 byte views")
    if index.ndim != 1 or index.dtype not in (torch.int32, torch.int64):
        raise TypeError("Expected one-dimensional int32/int64 indices")
    if not index.is_contiguous() or not out.is_contiguous() or x.stride(1) != 1:
        raise ValueError("Expected dense rows, contiguous indices and output")
    if not out.numel():
        return
    if backend == "triton":
        from . import gather_triton as triton_elementwise

        triton_elementwise.gather(x, index, out)
    elif backend == "torch":
        _torch_gather(x, index, out)
    else:
        raise ValueError("Unknown gather backend: " + backend)
