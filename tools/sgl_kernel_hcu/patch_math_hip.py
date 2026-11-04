# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Patch flashinfer math.cuh PTX bodies for GCN (HCU/DTK) builds.

The pinned flashinfer commit (bc29697b) ships exactly ten `asm volatile`
statements, all in include/flashinfer/math.cuh — CUDA-only PTX that cannot
assemble on GCN ("invalid output constraint '=f'"). (Newer flashinfer
renames this file math_hip.cuh; targeting that name was the round-7
FileNotFoundError.) Three of the ten (rsqrt / tanh / shfl_xor_sync) ARE on
the fused_add_rms_norm code path, so they get portable bodies (libm-exact
or hip builtins — accuracy-neutral or better vs the approx PTX), and the
ptx_* helpers get float-math ports. Verified against the pinned commit
(2026-11-03): 10/10 replacements apply, zero asm statements remain.

Run against the BUILD WORKSPACE copy only.
"""

import re
import sys
from pathlib import Path

REPLACEMENTS = [
    (
        r"__forceinline__ __device__ float ptx_exp2\(float x\) \{[^{}]*\}",
        "__forceinline__ __device__ float ptx_exp2(float x) { return exp2f(x); }",
    ),
    (
        r"__forceinline__ __device__ float ptx_log2\(float x\) \{[^{}]*\}",
        "__forceinline__ __device__ float ptx_log2(float x) { return log2f(x); }",
    ),
    (
        r"__forceinline__ __device__ half2 ptx_exp2\(half2 x\) \{[^{}]*\}",
        "__forceinline__ __device__ half2 ptx_exp2(half2 x) {\n"
        "  float2 f = __half22float2(x);\n"
        "  return __floats2half2_rn(exp2f(f.x), exp2f(f.y));\n}",
    ),
    (
        r"__forceinline__ __device__ half ptx_exp2\(half x\) \{[^{}]*\}",
        "__forceinline__ __device__ half ptx_exp2(half x) "
        "{ return __float2half(exp2f(__half2float(x))); }",
    ),
    (
        r"__forceinline__ __device__ float ptx_rcp\(float x\) \{[^{}]*\}",
        "__forceinline__ __device__ float ptx_rcp(float x) { return 1.0f / x; }",
    ),
    (
        r"__forceinline__ __device__ float shfl_xor_sync\(float x, int lane_mask\) \{[^{}]*\}",
        # warp-32 butterfly semantics on wave64: width=32 keeps the xor
        # pattern inside each 32-lane segment; full 64-bit mask (dtk2610
        # requires 64-bit masks in warp sync builtins)
        "__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {\n"
        "  return __shfl_xor_sync(0xffffffffffffffffull, x, lane_mask, 32);\n}",
    ),
    (
        r"__forceinline__ __device__ float rsqrt\(float x\) \{[^{}]*\}",
        # exact 1/sqrt (IEEE sqrt + divide) — bit-safer than the PTX approx
        "__forceinline__ __device__ float rsqrt(float x) "
        "{ return 1.0f / sqrtf(x); }",
    ),
    (
        r"__forceinline__ __device__ float tanh\(float x\) \{[^{}]*\}",
        "__forceinline__ __device__ float tanh(float x) { return tanhf(x); }",
    ),
    (
        r"__forceinline__ __device__ half2 tanh\(half2 x\) \{[^{}]*\}",
        "__forceinline__ __device__ half2 tanh(half2 x) {\n"
        "  float2 f = __half22float2(x);\n"
        "  return __floats2half2_rn(tanhf(f.x), tanhf(f.y));\n}",
    ),
    (
        r"__forceinline__ __device__ half tanh\(half x\) \{[^{}]*\}",
        "__forceinline__ __device__ half tanh(half x) "
        "{ return __float2half(tanhf(__half2float(x))); }",
    ),
]


def main(path: str) -> int:
    p = Path(path)
    src = p.read_text(encoding="utf-8")
    replaced = 0
    for pat, rep in REPLACEMENTS:
        src, n = re.subn(pat, rep, src, count=1)
        replaced += n
    p.write_text(src, encoding="utf-8")
    left = len(re.findall(r"asm volatile", src))
    print(f"patch_math_hip: {replaced}/10 functions ported, "
          f"{left} asm statements left in {p}")
    return 0 if replaced == 10 and left == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
