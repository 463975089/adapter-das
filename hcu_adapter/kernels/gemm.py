"""Dense GEMM backend selection; implementations carry tier suffixes."""

import torch

from . import gemm_emu


def cublaslt_gemm_nt(a, b, d, c=None):
    """Keep the existing BF16 LightOp selection and FP32/degenerate emu path."""
    gemm_emu.validate_dense_nt(a, b, d, c)
    if (a.shape[0] and b.shape[0] and a.shape[1]
            and a.dtype == torch.bfloat16 and a.is_contiguous()
            and b.is_contiguous() and a.shape[1] % 32 == 0):
        from .gemm_lightop import bf16_gemm_nt

        return bf16_gemm_nt(a, b, d, c)
    return gemm_emu.cublaslt_gemm_nt(a, b, d, c)
