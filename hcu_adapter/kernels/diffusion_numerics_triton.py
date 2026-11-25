# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl


@triton.jit
def mul_rn_f32(x, y):
    return tl.inline_asm_elementwise(
        asm="v_mul_f32 $0, $1, $2",
        constraints="=v,v,v",
        args=[x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def div_rn_f32(x, y):
    # DTK lowers precise FP32 division to v_div_scale/fmas/fixup.
    return tl.div_rn(x, y)


@triton.jit
def rsqrt_approx_f32(x):
    # V_RSQ_F32 flushes subnormal inputs on gfx936.
    magnitude = x.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    subnormal = (magnitude > 0) & (magnitude < 0x00800000)
    scaled = tl.where(subnormal, x * 16777216.0, x)
    result = tl.inline_asm_elementwise(
        asm="v_rsq_f32 $0, $1",
        constraints="=v,v",
        args=[scaled],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return tl.where(subnormal, result * 4096.0, result)
