"""fp8_utils.bmm_fp8 bridge — MLA absorb fp8 batched GEMM without flashinfer.

Upstream `gemm.bmm_fp8` KernelSpec targets
`sglang.srt.layers.quantization.fp8_utils:bmm_fp8`, but that function is
defined only inside fp8_utils' `_is_cuda` block *after* a successful
`from flashinfer import bmm_fp8`. On HCU (§0.5 is_cuda semantics with
flashinfer deliberately pinned unavailable) neither holds, so spec.load()
AttributeError's at the first MLA-absorb bmm (models whose kv_b_proj
stays fp8 after weight slicing, e.g. GLM-5.2-Channel-FP8-w8a8).

Bridge: supply the module attribute with a torch-native implementation —
dequantize both operands to fp32, plain bmm, scale by the two per-tensor
scales, cast to the requested dtype. Arithmetic is exactly what the
cuBLAS epilogue computes; HCU has no batched fp8 GEMM primitive to route
to, and the operands are small (per-head absorbed projections).
"""

import logging

LOG = logging.getLogger("hcu.adapter")


def install():
    import torch

    import sglang.srt.layers.quantization.fp8_utils as fpu

    if getattr(fpu, "bmm_fp8", None) is not None:
        return  # someone (real flashinfer env) already provides it

    def _hcu_bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
        out_dtype = dtype if out is None else out.dtype
        # 2026-10-24 emu audit: keep the scales as 0-d GPU tensors. The old
        # float(A_scale...) form cost TWO DtoH syncs per call (capture-unsafe,
        # and a stall suspect wherever this bridge does run, e.g. prefill
        # MLA-absorb); the tensor form is the same per-tensor fp32 math with
        # zero host round-trips.
        s = A_scale.reshape(-1)[0] * B_scale.reshape(-1)[0]
        result = (
            torch.bmm(A.to(torch.float32), B.to(torch.float32)) * s
        ).to(out_dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    fpu.bmm_fp8 = _hcu_bmm_fp8
    LOG.warning(
        "HCU adapter: fp8_utils.bmm_fp8 bridged to torch-native dequant bmm"
        " (flashinfer cuBLAS fp8 bmm unavailable)"
    )
