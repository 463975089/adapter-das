"""LightOp provider for the existing SGLang symmetric INT8 scaled-MM API."""

import torch


def supports_scaled_mm_layout(a, b, out_dtype, bias):
    """Check LightOp's layout contract independently of the row count."""
    return (
        torch.version.hip is not None
        and out_dtype == torch.bfloat16
        and bias is None
        and a.shape[1] >= 128
        and a.shape[1] % 128 == 0
        and b.shape[1] % 64 == 0
        and a.is_contiguous()
        and b.stride() == (1, a.shape[1])
    )


def try_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias):
    """Return None when LightOp cannot execute the GEMM."""
    if a.shape[0] < 32 or not supports_scaled_mm_layout(a, b, out_dtype, bias):
        return None
    try:
        from lightop.gemmopt import gemm_w8a8_smooth
    except ImportError:
        return None
    supported, output = gemm_w8a8_smooth(
        a, b, scale_a.view(-1, 1), scale_b.view(-1, 1), None, out_dtype,
    )
    return output if supported else None
