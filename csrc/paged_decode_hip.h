// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Paged attention decode kernel (fa3-facing) — HIP C++ v3, token-serial.
//
// ONE workgroup (64 lanes) per (batch, kv-head); all G QA heads served from
// one pass over the paged KV (the triton face loads K/V once per QA head —
// here G amortizes the page-walk and page-table loads).
//
// Lane mapping (v3, corrected): lane l owns q/k/v dims (2l, 2l+1) — a full
// D=128 dot is carried by the whole wave with ONE tree reduction per token;
// tokens are walked serially inside a page (the triton face instead pads to
// BLOCK_N and reduces per block). The v2 lane=token mapping only ever
// touched 2 of 128 dims per token — caught by the minimal-case probe.
//
// Numerics: fp32 accumulation, per-token online-softmax checkpoint (safe
// for long tails), exp via __expf. bf16/fp16 via 4B half2 lanes.
//
// HCU constraints: wave64 full butterflies only, int64 page addressing, no
// warp32 assumptions. D=128 / PAGE=64 are the compile-time fast path.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>

#define DEVINL __device__ __forceinline__

namespace sgl_kernel_hcu {

// full-wave butterfly: every lane ends with the wave-wide max
DEVINL float wave_max64(float v) {
#pragma unroll
  for (int off = 32; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, 64));
  return v;
}
// full-wave butterfly: every lane ends with the wave-wide sum
DEVINL float wave_sum64(float v) {
#pragma unroll
  for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
  return v;
}

template <typename T>
struct H2;

template <>
struct H2<hip_bfloat16> {
  static DEVINL void ld2(const hip_bfloat16* p, float& lo, float& hi) {
    const uint32_t u = *reinterpret_cast<const uint32_t*>(p);
    lo = __uint_as_float((u & 0xFFFFu) << 16);
    hi = __uint_as_float(u & 0xFFFF0000u);
  }
  static DEVINL void st2(hip_bfloat16* p, float lo, float hi) {
    uint32_t a, b;
    __builtin_memcpy(&a, &lo, 4);
    __builtin_memcpy(&b, &hi, 4);
    *reinterpret_cast<uint32_t*>(p) = ((a >> 16) & 0xFFFFu) | (b & 0xFFFF0000u);
  }
};

template <>
struct H2<__half> {
  static DEVINL void ld2(const __half* p, float& lo, float& hi) {
    const uint32_t u = *reinterpret_cast<const uint32_t*>(p);
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&u));
    lo = f.x;
    hi = f.y;
  }
  static DEVINL void st2(__half* p, float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    *reinterpret_cast<uint32_t*>(p) = *reinterpret_cast<const uint32_t*>(&h);
  }
};

// D=128, PAGE=64 fast path. Launch: grid (B, HKV), block 64, smem unused.
template <typename T, int D, int PAGE>
__global__ void __launch_bounds__(64)
paged_decode_w64(
    const T* __restrict__ q,   // [B, H, D]
    const T* __restrict__ k,   // [P, PAGE, HKV, D]
    const T* __restrict__ v,
    const int32_t* __restrict__ pt,   // [B, PT_W]
    const int32_t* __restrict__ seqlens,
    T* __restrict__ o,                // [B, H, D]
    float* __restrict__ lse,          // [B, H]
    float scale,
    int H, int HKV, int PT_W) {
  static_assert(D == 128 && PAGE == 64, "fast path");
  const int b = blockIdx.x;
  const int kvh = blockIdx.y;
  const int lane = threadIdx.x;
  const int G = H / HKV;
  const int seq_len = seqlens[b];
  const int pages = (seq_len + PAGE - 1) / PAGE;

  for (int g = 0; g < G; ++g) {
    const int h = kvh * G + g;
    // q in registers: lane l holds dims (2l, 2l+1)
    float q0, q1;
    H2<T>::ld2(q + ((int64_t)(b * H + h)) * D + 2 * lane, q0, q1);

    float m_i = -INFINITY, l_i = 0.f, a0 = 0.f, a1 = 0.f;

    for (int p = 0; p < pages; ++p) {
      const int32_t ph = pt[(int64_t)b * PT_W + p];
      const int64_t poff = (int64_t)ph * PAGE * HKV * D;
      const int ntok = min(PAGE, seq_len - p * PAGE);
      // token t row: [P, PAGE, HKV, D] -> poff + (t*HKV + kvh)*D
      const T* __restrict__ kpage = k + poff + (int64_t)kvh * D;
      const T* __restrict__ vpage = v + poff + (int64_t)kvh * D;
      for (int t = 0; t < ntok; ++t) {
        const int64_t row = (int64_t)t * HKV * D;
        float k0, k1, v0, v1;
        H2<T>::ld2(kpage + row + 2 * lane, k0, k1);
        const float part = (k0 * q0 + k1 * q1) * scale;
        const float stot = wave_sum64(part);            // full-wave dot
        const float m_new = fmaxf(m_i, stot);
        const float alpha = __expf(m_i - m_new);
        const float p_t = __expf(stot - m_new);
        l_i = l_i * alpha + p_t;
        H2<T>::ld2(vpage + row + 2 * lane, v0, v1);
        a0 = a0 * alpha + p_t * v0;
        a1 = a1 * alpha + p_t * v1;
        m_i = m_new;
      }
    }

    const float denom = fmaxf(l_i, 1e-10f);
    H2<T>::st2(o + ((int64_t)(b * H + h)) * D + 2 * lane, a0 / denom, a1 / denom);
    if (lane == 0) lse[(int64_t)b * H + h] = m_i + logf(denom);
  }
}

}  // namespace sgl_kernel_hcu
