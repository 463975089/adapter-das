"""CT token-INT8 quantization retaining the activation storage dtype.

This is a separate component from the FP32-promoting quantizer. It does
not select a serving scheme or establish model numerical equivalence.
Active rows must be finite; padding may contain NaNs.
"""

import math

import torch
import triton
import triton.language as tl

from .w4a8_int8_triton import _counts, w4a8_int8_masked_gemm


# Target for temporary expert computation, excluding the required final output.
# A single expert can exceed it; no counts are read back to choose the batch.
_EXPERT_SCRATCH_BYTES = 64 * 1024 * 1024


@triton.jit
def _native_token_int8_kernel(
    A, Q, S, COUNTS,
    M: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr,
):
    expert, row = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, BLOCK)
    valid = row < tl.load(COUNTS + expert)
    value = tl.load(
        A + (expert * M + row) * K + k, valid & (k < K), other=0,
    ).to(tl.float32)
    # CT stores both scale and quotient in the activation dtype.
    # Explicit casts prevent the Triton expression from eliding those
    # rounding boundaries; RN division also avoids reciprocal substitution.
    scale = tl.div_rn(tl.max(tl.abs(value), 0), 127.5)
    scale = scale.to(A.dtype.element_ty).to(tl.float32)
    scale = tl.where(scale == 0, EPS, scale)
    scale = tl.where(valid, scale, 1.)
    normalized = tl.div_rn(value, scale).to(A.dtype.element_ty).to(tl.float32)
    clamped = tl.minimum(tl.maximum(normalized, -128.), 127.)
    lower = tl.floor(clamped)
    fraction = clamped - lower
    increment = (fraction > .5) | ((fraction == .5) & ((lower.to(tl.int32) & 1) != 0))
    quantized = lower + increment.to(tl.float32)
    tl.store(Q + (expert * M + row) * K + k, quantized.to(tl.int8), k < K)
    tl.store(S + expert * M + row, scale)


def token_int8_native_masked(value, counts):
    """Return INT8 codes and input-dtype scales for expert-major valid rows.

    Valid zero rows use finfo(input dtype).eps; padding uses codes=0,
    scale=1. Counts remain on device, including on changed-input replay.
    To feed the existing integer GEMM, explicitly promote scales to FP32;
    that conversion preserves these already rounded scale values.
    """
    if (value.device.type != "cuda" or value.ndim != 3
            or value.dtype not in (torch.float32, torch.bfloat16, torch.float16)
            or value.shape[0] < 1 or value.shape[2] < 1):
        raise ValueError("requires device FP32/BF16/FP16 activations [E, M, K]")
    experts, capacity, hidden = value.shape
    _counts(counts, value.device, experts, capacity)
    codes = torch.empty(value.shape, dtype=torch.int8, device=value.device)
    scales = torch.empty((experts, capacity, 1), dtype=value.dtype, device=value.device)
    if capacity:
        with torch.cuda.device(value.device):
            _native_token_int8_kernel[(experts, capacity)](
                value.contiguous(), codes, scales, counts.contiguous(),
                capacity, hidden, triton.next_power_of_2(hidden),
                torch.finfo(value.dtype).eps,
                num_warps=4, enable_fp_fusion=False,
            )
    return codes, scales


def w4a8_int8_native_masked_experts(
    value, gate_up_weights, down_weights, gate_up_scales, down_scales, counts,
    *, swiglu_limit=None,
):
    """Integer experts with input-dtype projection and fused-product storage.

    Both projections and the SiLU-times-up product retain the input dtype.
    SiLU and multiplication compute in FP32 before one product store.
    Integer accumulators rescale in FP32 before storing that dtype. The
    output retains expert-major order for a later combine; this component
    itself neither dispatches nor applies router weights.
    """
    if value.ndim != 3:
        raise ValueError("requires expert-major activations [E, M, K]")
    experts, capacity, hidden = value.shape
    if (gate_up_weights.ndim != 3 or gate_up_weights.shape[1] % 2
            or gate_up_weights.shape[1] < 2):
        raise ValueError("gate/up weights must contain two equal projections")
    intermediate = gate_up_weights.shape[1] // 2
    if (hidden % 2 or intermediate % 2
            or gate_up_weights.shape != (experts, 2 * intermediate, hidden // 2)
            or down_weights.shape != (experts, hidden, intermediate // 2)):
        raise ValueError("packed expert dimensions must map K -> I -> K")
    if swiglu_limit is not None and (
        type(swiglu_limit) not in (int, float)
        or not math.isfinite(swiglu_limit) or swiglu_limit <= 0
    ):
        raise ValueError("swiglu_limit must be a finite positive Python scalar")
    if (value.device.type != "cuda"
            or value.dtype not in (torch.float32, torch.bfloat16, torch.float16)
            or experts < 1 or hidden < 1):
        raise ValueError("requires device FP32/BF16/FP16 activations [E, M, K]")
    _counts(counts, value.device, experts, capacity)
    # Process independent experts in bounded groups. The final expert-major
    # buffer is still required by DeepEP combine, but graph capture need not
    # retain full-expert FP32 projection/activation temporaries.
    bytes_per_expert = capacity * 4 * (hidden + 10 * intermediate)
    batch = max(1, _EXPERT_SCRATCH_BYTES // max(1, bytes_per_expert))
    if experts <= batch:
        return _expert_batch(
            value, gate_up_weights, down_weights, gate_up_scales, down_scales,
            counts, swiglu_limit,
        )
    if (gate_up_scales.shape != (experts, 2 * intermediate, 1)
            or down_scales.shape != (experts, hidden, 1)):
        raise ValueError("requires expert-major channel weight scales [E, N, 1]")
    output = torch.empty(value.shape, device=value.device, dtype=value.dtype)
    for begin in range(0, experts, batch):
        end = min(begin + batch, experts)
        result = _expert_batch(
            value[begin:end], gate_up_weights[begin:end], down_weights[begin:end],
            gate_up_scales[begin:end], down_scales[begin:end], counts[begin:end],
            swiglu_limit,
        )
        output[begin:end].copy_(result)
        del result
    return output


def _expert_batch(
    value, gate_up_weights, down_weights, gate_up_scales, down_scales, counts,
    swiglu_limit,
):
    codes, scales = token_int8_native_masked(value, counts)
    projected = w4a8_int8_masked_gemm(
        codes, scales.float(), gate_up_weights, gate_up_scales, counts,
    ).to(value.dtype)
    gate, up = projected.chunk(2, dim=-1)
    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    activated = (torch.nn.functional.silu(gate.float()) * up.float()).to(value.dtype)
    del codes, scales, projected, gate, up
    codes, scales = token_int8_native_masked(activated, counts)
    del activated
    return w4a8_int8_masked_gemm(
        codes, scales.float(), down_weights, down_scales, counts,
    ).to(value.dtype)
