"""Triton implementation of the existing norm contract."""

import triton
import triton.language as tl


@triton.jit
def _norm(X, Y, W, R, RO, SF, Extra, Origin,
          D: tl.constexpr, XS: tl.constexpr, XB: tl.constexpr,
          YS: tl.constexpr, YB: tl.constexpr, RS: tl.constexpr, RB: tl.constexpr,
          ROS: tl.constexpr, ROB: tl.constexpr, OS: tl.constexpr, OB: tl.constexpr,
          SFS: tl.constexpr, SFB: tl.constexpr, SFG: tl.constexpr, PACK: tl.constexpr,
          ES0: tl.constexpr, ES1: tl.constexpr, ES2: tl.constexpr,
          RESIDUAL: tl.constexpr, EPS: tl.constexpr, OUT_SCALE: tl.constexpr,
          QUANT: tl.constexpr, NPC: tl.constexpr, ROUND: tl.constexpr,
          PACKED: tl.constexpr, ORIGIN: tl.constexpr, SHARED_BM: tl.constexpr,
          BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + batch * XB + row * XS + col, col < D, other=0).to(tl.float32)
    if RESIDUAL:
        residual = tl.load(R + batch * RB + row * RS + col, col < D, other=0).to(tl.float32)
        value = (value + residual).to(X.dtype.element_ty).to(tl.float32)
        tl.store(RO + batch * ROB + row * ROS + col, value, col < D)
    weight = tl.load(W + col, col < D, other=0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(value * value, 0) / D + EPS)
    result = (value * rstd) * (weight * OUT_SCALE)
    if ORIGIN:
        tl.store(Origin + batch * OB + row * OS + col, result, col < D)
    if not QUANT:
        tl.store(Y + batch * YB + row * YS + col, result, col < D)
    else:
        q = result.to(X.dtype.element_ty).to(tl.float32).reshape((BLOCK // NPC, NPC))
        amax = tl.maximum(tl.max(tl.abs(q), 1), 1e-4)
        if ROUND:
            bits = (amax / 448.0).to(tl.int32, bitcast=True)
            exponent_byte = ((bits - 1) >> 23) + 1
            multiplier = ((254 - exponent_byte) << 23).to(tl.float32, bitcast=True)
            scale = (exponent_byte << 23).to(tl.float32, bitcast=True)
        else:
            multiplier = 448.0 / amax
            scale = amax / 448.0
        quantized = (q * multiplier[:, None]).reshape((BLOCK,))
        tl.store(Y + batch * YB + row * YS + col, quantized, col < D)
        group = tl.arange(0, BLOCK // NPC)
        sf_offset = batch * SFB + row * SFS + group // PACK * SFG + group % PACK
        if PACKED:
            tl.store(SF + sf_offset, exponent_byte, group < D // NPC)
        else:
            tl.store(SF + sf_offset, scale, group < D // NPC)
        if SHARED_BM:
            local = row % SHARED_BM
            mapped = (row // SHARED_BM * triton.cdiv(SHARED_BM, 128) * 128 +
                      local // 128 * 128 + local % 32 * 4 + local % 128 // 32)
            extra_offset = mapped * ES0 + group // 4 * ES1 + group % 4 * ES2
            tl.store(Extra + extra_offset, exponent_byte, group < D // NPC)


def norm(x, out, weight, residual, residual_out, sf, extra_sf, origin, *,
         has_residual, eps, out_scale, do_fp8_cast, num_per_channels,
         col_packed, round_sf, use_packed_ue8m0, return_origin_out, shared_sf_block_m):
    import torch

    sf_strides = sf.stride() if do_fp8_cast else (0, 0, 0)
    extra_strides = extra_sf.stride() if shared_sf_block_m else (0, 0, 0)
    with torch.cuda.device(x.device):
        _norm[(x.shape[1], x.shape[0])](
            x, out, weight, residual, residual_out, sf, extra_sf, origin,
            x.shape[-1], x.stride(1), x.stride(0), out.stride(1), out.stride(0),
            residual.stride(1), residual.stride(0), residual_out.stride(1), residual_out.stride(0),
            origin.stride(1), origin.stride(0), sf_strides[1], sf_strides[0], sf_strides[2],
            4 if col_packed else 1, *extra_strides,
            has_residual, eps, out_scale, do_fp8_cast, num_per_channels or 1,
            round_sf, use_packed_ue8m0, return_origin_out, shared_sf_block_m,
            triton.next_power_of_2(x.shape[-1]), num_warps=4, enable_fp_fusion=False)
