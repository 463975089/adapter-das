# SPDX-License-Identifier: Apache-2.0
"""Framework-free BF16 clamped-SiLU and token-FP8 fusion.

Both supported quantizers preserve their own baseline arithmetic: upstream
uses value/scale with a scale floor; LightOp uses value*(448/amax) without
a floor. The two contracts differ at FP8 ties and on zero/tiny rows.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _silu_clamp_bf16_quant(X, Q, S, N: tl.constexpr, LIMIT: tl.constexpr,
                         BLOCK: tl.constexpr, LIGHTOP: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    gate = tl.load(X + row * (2 * N) + col, col < N, other=0).to(tl.float32)
    up = tl.load(X + row * (2 * N) + N + col, col < N, other=0).to(tl.float32)
    # The native clamp stores gate back to BF16 before evaluating SiLU.
    gate = tl.minimum(gate, LIMIT).to(tl.bfloat16).to(tl.float32)
    # fmaxf(fminf(up, limit), -limit): the order also selects +limit for NaN.
    up = tl.maximum(tl.minimum(up, LIMIT), -LIMIT)
    if N % 8:
        # Native BF16 widths not divisible by its 128-bit vector width use
        # compute<scalar_t>, which also rounds the clamped up operand.
        up = up.to(tl.bfloat16).to(tl.float32)
    # Stock computes (up + beta), including beta=+0: -0 + +0 becomes +0.
    # Preserve that sign rule explicitly, since a compiler may fold up + 0.
    up = tl.where(up == 0.0, 0.0, up)
    # _C.silu_and_mul_with_clamp rounds the activation, then the product.
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    value = (activated * up).to(tl.bfloat16).to(tl.float32)
    # Both native providers reduce with fmaxf, ignoring a NaN operand.
    # In particular an all-NaN row has amax=0, not a NaN LightOp scale.
    amax = tl.max(tl.where((col < N) & (value == value), tl.abs(value), 0.0), 0)
    scale = tl.div_rn(amax, 448.0)
    if not LIGHTOP:
        scale = tl.maximum(scale, 1.0 / (448.0 * 512.0))
    # Stock scaled_fp8_conversion<false> divides by scale. Multiplying by a
    # rounded reciprocal changes FP8 tie decisions for some BF16 rows.
    if LIGHTOP:
        divided = value * tl.div_rn(448.0, tl.where(amax == 0, 1.0, amax))
    else:
        divided = tl.div_rn(value, scale)
    quant = tl.maximum(-448.0, tl.minimum(divided, 448.0))
    if LIGHTOP:
        # LightOp preserves the negative canonical NaN from 0 * inf when a
        # tiny nonzero row overflows 448/amax. Native fmin/fmax would instead
        # replace it with +448; only the LightOp contract restores NaN.
        nan = tl.full((), 0xffc00000, tl.uint32).to(tl.float32, bitcast=True)
        quant = tl.where(divided != divided, nan, quant)
    tl.store(Q + row * N + col, quant, col < N)
    tl.store(S + row, scale)


@triton.jit
def _shared_bf16_value(gate, up, LIMIT: tl.constexpr):
    gate = tl.minimum(gate.to(tl.float32), LIMIT).to(tl.bfloat16).to(tl.float32)
    up = tl.maximum(tl.minimum(up.to(tl.float32), LIMIT), -LIMIT)
    up = tl.where(up == 0.0, 0.0, up)
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    return (activated * up).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _silu_clamp_bf16_quant_2304(X, Q, S, LIMIT: tl.constexpr, LIGHTOP: tl.constexpr):
    # Model width is 2304. A 4096-column tile spends almost half its work on
    # padding; reduce the 2048 main columns and 256 tail separately instead.
    row = tl.program_id(0)
    col = tl.arange(0, 2048)
    tail = 2048 + tl.arange(0, 256)
    value = _shared_bf16_value(tl.load(X + row * 4608 + col),
                              tl.load(X + row * 4608 + 2304 + col), LIMIT)
    last = _shared_bf16_value(tl.load(X + row * 4608 + tail),
                             tl.load(X + row * 4608 + 2304 + tail), LIMIT)
    amax = tl.maximum(
        tl.max(tl.where(value == value, tl.abs(value), 0.0), 0),
        tl.max(tl.where(last == last, tl.abs(last), 0.0), 0))
    scale = tl.div_rn(amax, 448.0)
    if not LIGHTOP:
        scale = tl.maximum(scale, 1.0 / (448.0 * 512.0))
    # Preserve correctly rounded division before FP8 conversion. A residual-
    # corrected reciprocal still changes FP8 tie decisions for some rows.
    if LIGHTOP:
        inverse = tl.div_rn(448.0, tl.where(amax == 0, 1.0, amax))
        divided = value * inverse
        divided_tail = last * inverse
    else:
        divided = tl.div_rn(value, scale)
        divided_tail = tl.div_rn(last, scale)
    quant = tl.maximum(-448.0, tl.minimum(divided, 448.0))
    quant_tail = tl.maximum(-448.0, tl.minimum(divided_tail, 448.0))
    if LIGHTOP:
        nan = tl.full((), 0xffc00000, tl.uint32).to(tl.float32, bitcast=True)
        quant = tl.where(divided != divided, nan, quant)
        quant_tail = tl.where(divided_tail != divided_tail, nan, quant_tail)
    tl.store(Q + row * 2304 + col, quant)
    tl.store(Q + row * 2304 + tail, quant_tail)
    tl.store(S + row, scale)


def silu_clamp_bf16_quant(x: torch.Tensor, limit: float, *, lightop: bool = False):
    if (not x.is_cuda or x.ndim != 2 or x.dtype != torch.bfloat16
            or not x.is_contiguous()):
        raise ValueError("shared activation fusion requires contiguous BF16 [M,2N]")
    m, two_n = x.shape
    if two_n <= 0 or two_n % 2 or two_n > 16384:
        raise ValueError("unsupported shared activation width")
    n = two_n // 2
    # The installed LightOp scalar/tail conversion differs from its vector
    # kernel for overflowing inverse scales. Keep that route unfused.
    if lightop and n % 16:
        raise ValueError("LightOp fusion requires a vector-aligned width")
    q = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    if m:
        if n == 2304:
            _silu_clamp_bf16_quant_2304[(m,)](
                x, q, s, limit, lightop, num_warps=4, enable_fp_fusion=False)
        else:
            _silu_clamp_bf16_quant[(m,)](
                x, q, s, n, limit, triton.next_power_of_2(n), lightop,
                num_warps=4, enable_fp_fusion=False)
    return q, s
