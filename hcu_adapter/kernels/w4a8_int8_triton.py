"""Candidate kernels for checkpoint-declared channel INT4 / token INT8.

These components are not connected to a serving scheme. They implement the
CT FP32 activation-QDQ convention, not the community W4AFP8 tensor-FP8 path.
Only finite active activations and positive finite scales are supported.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _token_int8_kernel(
    A, Q, S, COUNTS,
    M: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr,
):
    expert, row = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, BLOCK)
    valid = row < tl.load(COUNTS + expert)
    value = tl.load(
        A + (expert * M + row) * K + k, valid & (k < K), other=0,
    ).to(tl.float32)
    maximum = tl.max(tl.abs(value), 0)
    # Explicit RN division preserves CT halfway rounding; reciprocal-multiply
    # is not an equivalent quantizer at integer half boundaries.
    scale = tl.div_rn(maximum, 127.5)
    scale = tl.where(scale == 0, 1.1920928955078125e-7, scale)
    scale = tl.where(valid, scale, 1.0)
    normalized = tl.div_rn(value, scale)
    clamped = tl.minimum(tl.maximum(normalized, -128.), 127.)
    lower = tl.floor(clamped)
    fraction = clamped - lower
    # Explicit ties-to-even, including negative half values.
    increment = (fraction > .5) | ((fraction == .5) & ((lower.to(tl.int32) & 1) != 0))
    quantized = lower + increment.to(tl.float32)
    tl.store(Q + (expert * M + row) * K + k, quantized.to(tl.int8), k < K)
    tl.store(S + expert * M + row, scale)


@triton.jit
def _channel_int4_int8_kernel(
    A, AS, W, WS, C, COUNTS,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    expert = tl.program_id(0)
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    columns = tl.program_id(2) * BN + tl.arange(0, BN)
    count = tl.load(COUNTS + expert)
    if tl.program_id(1) * BM < count:
        kk = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.int32)
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + kk
            activation = tl.load(
                A + (expert * M + rows[:, None]) * K + k[None, :],
                (rows[:, None] < count) & (rows[:, None] < M) & (k[None, :] < K),
                other=0,
            )
            packed = tl.load(
                W + (expert * N + columns[None, :]) * (K // 2) + k[:, None] // 2,
                (columns[None, :] < N) & (k[:, None] < K), other=0,
            ).to(tl.int32)
            weight = (((packed >> ((k[:, None] % 2) * 4)) & 15) - 8).to(tl.int8)
            acc = tl.dot(activation, weight, acc=acc, out_dtype=tl.int32)
        activation_scale = tl.load(
            AS + expert * M + rows, (rows < count) & (rows < M), other=1,
        )
        weight_scale = tl.load(
            WS + expert * N + columns, columns < N, other=1,
        ).to(tl.float32)
        output = acc.to(tl.float32) * activation_scale[:, None] * weight_scale[None, :]
        tl.store(
            C + (expert * M + rows[:, None]) * N + columns[None, :],
            output, (rows[:, None] < count) & (rows[:, None] < M)
            & (columns[None, :] < N),
        )


def _counts(counts, device, experts, capacity):
    if (counts.device != device or counts.dtype != torch.int32
            or counts.shape != (experts,)):
        raise ValueError("counts must be device-local int32 [experts]")
    torch._assert_async(
        torch.all((counts >= 0) & (counts <= capacity)),
        f"CT W4A8 counts must be within [0, {capacity}]",
    )


def token_int8_masked(value, counts):
    """Quantize valid expert-major rows; padding codes=0 and scales=1.

    BF16/FP16 inputs are promoted to FP32 for computing CT's absmax/127.5
    scales, rounding and clamp. Zero valid rows use FP32 epsilon. Dispatch
    counts remain on device and are rechecked on graph replay.
    """
    if (value.device.type != "cuda" or value.ndim != 3
            or value.dtype not in (torch.float32, torch.bfloat16, torch.float16)
            or value.shape[0] < 1 or value.shape[2] < 1):
        raise ValueError("requires device FP32/BF16/FP16 activations [E, M, K]")
    experts, capacity, hidden = value.shape
    _counts(counts, value.device, experts, capacity)
    codes = torch.empty(value.shape, dtype=torch.int8, device=value.device)
    scales = torch.empty((experts, capacity, 1), dtype=torch.float32, device=value.device)
    if capacity:
        with torch.cuda.device(value.device):
            _token_int8_kernel[(experts, capacity)](
                value.contiguous(), codes, scales, counts.contiguous(),
                capacity, hidden, triton.next_power_of_2(hidden),
                num_warps=4, enable_fp_fusion=False,
            )
    return codes, scales


def w4a8_int8_masked_gemm(codes, activation_scales, weights, weight_scales, counts):
    """Integer INT8 x signed INT4 GEMM with token/channel FP32 rescaling.

    Weights are raw low-even-nibble uint8 [E, N, K/2], without a second
    persistent repack. Channel scales are [E, N, 1]. The FP32 result has zero
    padded rows, including empty experts. No BF16 dequantization occurs
    inside the integer dot product.
    """
    if (codes.device.type != "cuda" or codes.dtype != torch.int8 or codes.ndim != 3):
        raise ValueError("requires device int8 codes [E, M, K]")
    experts, capacity, hidden = codes.shape
    # |int8 * int4| <= 1024; keep the full reduction within signed int32.
    if experts < 1 or hidden < 2 or hidden % 2 or hidden > 2097151:
        raise ValueError("requires an even positive K without int32 accumulation overflow")
    outputs = weights.shape[1] if weights.ndim == 3 else 0
    if (weights.device != codes.device or weights.dtype != torch.uint8
            or outputs < 1 or weights.shape != (experts, outputs, hidden // 2)):
        raise ValueError("requires device-local uint8 packed weights [E, N, K/2]")
    if (activation_scales.device != codes.device
            or activation_scales.dtype != torch.float32
            or activation_scales.shape != (experts, capacity, 1)):
        raise ValueError("requires device-local FP32 activation scales [E, M, 1]")
    if (weight_scales.device != codes.device
            or weight_scales.dtype not in (torch.float32, torch.bfloat16)
            or weight_scales.shape != (experts, outputs, 1)):
        raise ValueError("requires device-local channel weight scales [E, N, 1]")
    _counts(counts, codes.device, experts, capacity)
    output = torch.zeros((experts, capacity, outputs), dtype=torch.float32,
                         device=codes.device)
    if capacity:
        with torch.cuda.device(codes.device):
            _channel_int4_int8_kernel[
                (experts, triton.cdiv(capacity, 16), triton.cdiv(outputs, 64))
            ](
                codes.contiguous(), activation_scales.contiguous(), weights.contiguous(),
                weight_scales.contiguous(), output, counts.contiguous(),
                capacity, outputs, hidden, 16, 64, 128,
                num_warps=4, enable_fp_fusion=False,
            )
    return output
