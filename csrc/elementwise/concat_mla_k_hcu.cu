// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// concat_mla_k, HCU port of upstream csrc/elementwise/concat_mla.cu
// (sgl-kernel v0.5.22). The CUDA original vectorizes through NV PTX
// ld/st no-allocate intrinsics (utils.cuh st_na_global_* / prefetch_L2),
// which hipcc cannot compile — this port rewrites the copy as a plain
// uint4 (16-byte vector) per-thread kernel with the same contract:
//
//   k[T, 128, 192] = [ k_nope[T, 128, 128] | k_rope[T, 1, 64] broadcast
//                      over all 128 heads ]
//
// Contract checks mirror the upstream check_tensor() exactly (3D, bf16,
// unit last-dim stride, 16-byte base alignment); row strides must
// additionally be uint4-aligned, which the upstream kernel assumed
// implicitly through its int2/int pointer math.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include <torch/all.h>

#include <cstdint>

namespace {

constexpr int kNumHeads = 128;
constexpr int kNopeDim = 128;
constexpr int kRopeDim = 64;
constexpr int kKDim = kNopeDim + kRopeDim;  // 192
constexpr int kBf16PerVec = 8;              // uint4 = 16B = 8 bf16
constexpr int kNopeVecs = kNopeDim / kBf16PerVec;  // 16
constexpr int kRopeVecs = kRopeDim / kBf16PerVec;  // 8
constexpr int kKVecs = kKDim / kBf16PerVec;        // 24

__global__ void concat_mla_k_hcu_kernel(
    const uint4* __restrict__ k_nope,
    const uint4* __restrict__ k_rope,
    uint4* __restrict__ k,
    int64_t rows,          // num_tokens * kNumHeads
    int64_t kn_s0,         // k_nope strides in uint4 units
    int64_t kn_s1,
    int64_t kr_s0,         // k_rope stride(0) in uint4 units
    int64_t k_s0,          // k strides in uint4 units
    int64_t k_s1) {
  const int64_t vid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t row = vid / kKVecs;
  if (row >= rows) return;
  const int64_t off = vid - row * kKVecs;  // vector index within the head row
  const int64_t token = row / kNumHeads;
  const int64_t head = row - token * kNumHeads;
  uint4* dst = k + token * k_s0 + head * k_s1;
  if (off < kNopeVecs) {
    dst[off] = k_nope[token * kn_s0 + head * kn_s1 + off];
  } else {
    // one rope vector per token, shared by every head of that token
    dst[off] = k_rope[token * kr_s0 + (off - kNopeVecs)];
  }
}

inline void check_concat_mla_k_tensor(
    const at::Tensor& t, int64_t head_dim, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be on an HCU device");
  TORCH_CHECK(t.dim() == 3, name, " must be a 3D tensor");
  TORCH_CHECK(
      t.scalar_type() == at::ScalarType::BFloat16, name, " must be torch.bfloat16");
  TORCH_CHECK(t.size(2) == head_dim, name, ".size(2) must be ", head_dim);
  TORCH_CHECK(t.stride(2) == 1, name, " must be contiguous in its last dimension");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
      name, " base address must be 16-byte aligned");
  // A size-1 dim carries no addressing information — the kernel never
  // multiplies by its stride (k_rope is indexed token*stride(0) only,
  // head fixed at 0), and views like [T,64,1].transpose legitimately
  // carry stride(1)=1 there. Skip the alignment check for that dim
  // (review #80).
  TORCH_CHECK(t.stride(0) % kBf16PerVec == 0, name,
              " token stride must be 16-byte aligned");
  if (t.size(1) != 1) {
    TORCH_CHECK(t.stride(1) % kBf16PerVec == 0, name,
                " row stride must be 16-byte aligned");
  }
}

}  // namespace

void concat_mla_k(at::Tensor k, at::Tensor k_nope, at::Tensor k_rope) {
  check_concat_mla_k_tensor(k, kKDim, "k");
  check_concat_mla_k_tensor(k_nope, kNopeDim, "k_nope");
  check_concat_mla_k_tensor(k_rope, kRopeDim, "k_rope");
  TORCH_CHECK(
      k.size(1) == kNumHeads && k_nope.size(1) == kNumHeads,
      "k/k_nope head dim must be ", kNumHeads);
  TORCH_CHECK(k_rope.size(1) == 1, "k_rope head dim must be 1");
  TORCH_CHECK(
      k.size(0) == k_nope.size(0) && k.size(0) == k_rope.size(0),
      "k/k_nope/k_rope token counts must match");

  const int64_t num_tokens = k.size(0);
  if (num_tokens == 0) return;
  const int64_t rows = num_tokens * kNumHeads;
  const int64_t total_vecs = rows * kKVecs;

  const c10::cuda::OptionalCUDAGuard device_guard(k.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t blocks = (total_vecs + threads - 1) / threads;
  concat_mla_k_hcu_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint4*>(k_nope.data_ptr()),
      reinterpret_cast<const uint4*>(k_rope.data_ptr()),
      reinterpret_cast<uint4*>(k.data_ptr()),
      rows,
      k_nope.stride(0) / kBf16PerVec,
      k_nope.stride(1) / kBf16PerVec,
      k_rope.stride(0) / kBf16PerVec,
      k.stride(0) / kBf16PerVec,
      k.stride(1) / kBf16PerVec);
}
