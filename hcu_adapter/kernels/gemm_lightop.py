"""LightOp implementation of the upstream BF16 GEMM output contract."""


def bf16_gemm_nt(a, b, d, c=None):
    """Write ``a @ b.T + c`` into d without rounding before accumulation."""
    import lightop
    import torch

    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Expected a[M,K] and b[N,K]")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("bf16_gemm_nt requires BF16 inputs")
    if not a.is_cuda or a.device != b.device or a.device != d.device:
        raise ValueError("GEMM inputs and output must share one HCU device")
    if not a.is_contiguous() or not b.is_contiguous() or a.shape[1] % 32:
        raise ValueError("LightOp GEMM requires contiguous inputs and K divisible by 32")
    if d.shape != (a.shape[0], b.shape[0]):
        raise ValueError("Output shape does not match a @ b.T")
    if d.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Output must be BF16 or FP32")
    if c is not None and (c.shape != d.shape or c.device != d.device):
        raise ValueError("Accumulator must match output shape and device")
    result = lightop.gemm(a, b, out_dtype=torch.float32 if c is not None else d.dtype)
    if c is not None:
        result.add_(c)
    d.copy_(result)
