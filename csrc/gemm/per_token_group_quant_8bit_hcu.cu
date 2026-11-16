// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// sgl_per_token_group_quant_8bit (v1) and ..._v2, HCU ports of upstream
// csrc/gemm/per_token_group_quant_8bit{,_v2}.cu (sgl-kernel v0.5.22).
//
// The CUDA originals vectorize through flashinfer::vec_t, NV PTX
// (st.global / ld.global.nc), __nv_fp8x2 conversion intrinsics and PDL
// (cudaTriggerProgrammaticLaunchCompletion) — none of which hipcc can
// compile. This port keeps the CONTRACTS bit-exact (same host checks,
// same scale layouts incl. the ue8m0 packed-uint32 column-major format,
// same int8-truncating / fp8-RN casts, same eps/floor semantics) while
// running plain 16-lane-per-group kernels with warp-shuffle reductions:
//
//   v1: absmax floored at eps; y_s = absmax / max_8bit (division);
//       q = clamp(x / y_s) cast. ue8m0: y_s = exp2(ceil(log2(max(y_s,1e-10)))),
//       stored as biased-exponent byte (log2(y_s)+127).
//   v2: absmax floored at 1e-10 (eps TORCH_CHECKed == 1e-10, min/max
//       CHECK_EQ'd to the dst dtype range); TWO directions
//       (calculate_fp8_scales, v2.cu:69-80): the STORED scale_inv =
//       amax / dst_max (ue8m0: 2^ceil-log2 via the DeepEP bit trick,
//       byte = bits>>23) is the dequant scale, while the quantized
//       value multiplies the RECIPROCAL y_scale = dst_max / amax
//       (ue8m0: 2^-exp). fuse_silu_and_mul computes silu(primary)
//       cast BACK to the input dtype before the product; masked_m is
//       read ON DEVICE (per-expert row guard) like upstream's
//       MaskedLayoutScheduler, keeping the DeepEP (E, maxT, H) path
//       CUDA-graph capturable.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Float8_e4m3fn.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>

#include "pytorch_extension_utils_rocm.h"

namespace {

constexpr int kThreadsPerGroup = 16;
constexpr float kV2AbsmaxFloor = 1e-10f;

// Wave-size NOTE (DCU is a wave64 platform; CUDA's warp is 32): the
// group reduction below uses __shfl_xor with offsets 8/4/2/1 — offsets
// that only flip bits 0..3, so the shuffle partner always lies within
// the SAME 16-lane group regardless of whether __shfl_xor's implicit
// width is 32 (CUDA legacy) or 64 (wave64). The early-return guards are
// per-GROUP predicates (all 16 lanes share one gid), so no partially
// active group can ever hit a shuffle. Both kernels are therefore
// wave-agnostic by construction.
__device__ __forceinline__ float group_reduce_max16(float v) {
  // HIP's __shfl_xor is the legacy (var, laneMask) signature — NOT the
  // CUDA __shfl_xor_sync(mask, var, laneMask). xor offsets 8/4/2/1 stay
  // within each 16-lane group, so no width argument is needed.
#pragma unroll
  for (int off = kThreadsPerGroup / 2; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor(v, off));
  }
  return v;
}

__device__ __forceinline__ float silu_f(float x) {
  return x / (1.0f + __expf(-x));
}

template <typename DST>
__device__ __forceinline__ DST cast_quant(float q) {
  return static_cast<DST>(q);
}
// int8_t cast truncates (C++ semantics, matching the upstream kernel);
// c10::Float8_e4m3fn(float) rounds to nearest.
template <>
__device__ __forceinline__ c10::Float8_e4m3fn cast_quant<c10::Float8_e4m3fn>(float q) {
  return c10::Float8_e4m3fn(q);
}

// DeepEP fast_log2_ceil: unbiased exponent + (mantissa != 0).
__device__ __forceinline__ int fast_log2_ceil(float x) {
  const uint32_t bits = __float_as_uint(x);
  return ((bits >> 23) & 0xff) - 127 + ((bits & 0x7fffff) != 0 ? 1 : 0);
}

// ---------------------------------------------------------------- v1 ----

template <typename T, typename DST, bool IS_COL_MAJOR, bool SCALE_UE8M0>
__global__ void quant8_v1_kernel(
    const T* __restrict__ input,
    DST* __restrict__ output_q,
    float* __restrict__ output_s,
    int group_size,
    int64_t num_groups,
    int groups_per_block,
    float eps,
    float min_8bit,
    float max_8bit,
    int num_groups_per_row,
    int scale_stride) {
  const int local_group_id = threadIdx.x / kThreadsPerGroup;
  const int lane_id = threadIdx.x % kThreadsPerGroup;
  const int64_t global_group_id =
      static_cast<int64_t>(blockIdx.x) * groups_per_block + local_group_id;
  if (global_group_id >= num_groups) return;
  const int64_t base = global_group_id * group_size;

  float local_absmax = eps;
  for (int i = lane_id; i < group_size; i += kThreadsPerGroup) {
    local_absmax = fmaxf(local_absmax, fabsf(static_cast<float>(input[base + i])));
  }
  local_absmax = group_reduce_max16(local_absmax);

  float y_s = local_absmax / max_8bit;
  if (SCALE_UE8M0) {
    y_s = exp2f(ceilf(log2f(fmaxf(y_s, 1e-10f))));
  }

  if (lane_id == 0) {
    if (IS_COL_MAJOR && SCALE_UE8M0) {
      // packed uint32 column-major: biased-exponent byte, little-endian
      const uint8_t byte = static_cast<uint8_t>(static_cast<int>(log2f(y_s)) + 127);
      const int64_t row_idx = global_group_id / num_groups_per_row;
      const int col_unpacked = global_group_id % num_groups_per_row;
      const int64_t col_idx = col_unpacked / 4;
      const int pack_idx = col_unpacked % 4;
      auto* so = reinterpret_cast<uint8_t*>(output_s);
      so[col_idx * scale_stride * 4 + row_idx * 4 + pack_idx] = byte;
    } else if (IS_COL_MAJOR) {
      const int64_t row_idx = global_group_id / num_groups_per_row;
      const int64_t col_idx = global_group_id % num_groups_per_row;
      output_s[col_idx * scale_stride + row_idx] = y_s;
    } else {
      output_s[global_group_id] = y_s;
    }
  }

  for (int i = lane_id; i < group_size; i += kThreadsPerGroup) {
    const float v = static_cast<float>(input[base + i]);
    const float q = fminf(fmaxf(v / y_s, min_8bit), max_8bit);
    output_q[base + i] = cast_quant<DST>(q);
  }
}

// ---------------------------------------------------------------- v2 ----

template <typename T, typename DST, bool FUSE_SILU, bool SCALE_UE8M0>
__global__ void quant8_v2_kernel(
    const T* __restrict__ input,
    DST* __restrict__ output_q,
    float* __restrict__ output_s,
    const int32_t* __restrict__ row_limit,  // masked_m + expert, or null
    int64_t num_groups,   // total (row, group) pairs for this launch
    int groups_per_row,
    int group_size,
    float min_8bit,
    float max_8bit,
    int scale_stride,  // output_s.stride(-1): uint32 units when ue8m0, floats otherwise
    int is_column_major) {
  const int lane_id = threadIdx.x % kThreadsPerGroup;
  const int64_t gid =
      static_cast<int64_t>(blockIdx.x) * (blockDim.x / kThreadsPerGroup) +
      threadIdx.x / kThreadsPerGroup;
  if (gid >= num_groups) return;
  const int64_t row = gid / groups_per_row;
  const int g = gid - row * groups_per_row;
  // DeepEP masked layout: the expert's live row count is read ON DEVICE
  // (upstream MaskedLayoutScheduler reads masked_m[expert_idx] in the
  // kernel, v2.cu:237) — rows past it are left untouched. A host-side
  // .item() read would sync and break CUDA-graph capture/replay
  // (review #80). The guard is uniform across the whole 16-lane group.
  if (row_limit != nullptr && row >= *row_limit) return;
  // fuse inputs carry BOTH halves per row: (rows, groups_per_row *
  // group_size * 2); the row stride must account for the secondary tail.
  const int64_t row_stride_elems =
      static_cast<int64_t>(groups_per_row) * group_size * (FUSE_SILU ? 2 : 1);
  const int64_t base = row * row_stride_elems + g * group_size;
  const int64_t secondary_off = FUSE_SILU
                                    ? static_cast<int64_t>(groups_per_row) * group_size
                                    : 0;

  float local_absmax = kV2AbsmaxFloor;
  for (int i = lane_id; i < group_size; i += kThreadsPerGroup) {
    float v;
    if (FUSE_SILU) {
      // silu(primary) cast back to the input dtype, times secondary,
      // both amax source and quantization source are this product
      const T p = input[base + i];
      const T s = input[base + secondary_off + i];
      v = static_cast<float>(static_cast<T>(silu_f(static_cast<float>(p))) * s);
    } else {
      v = static_cast<float>(input[base + i]);
    }
    local_absmax = fmaxf(local_absmax, fabsf(v));
  }
  local_absmax = group_reduce_max16(local_absmax);

  // calculate_fp8_scales (v2.cu:69-80): scale_inv (STORED, dequant
  // direction) and y_scale (RECIPROCAL, the quantization multiplier —
  // multiplying by scale_inv zeroes every group; review #80 P1).
  float scale_inv, y_scale;
  if (SCALE_UE8M0) {
    const float ratio = fmaxf(local_absmax / max_8bit, 1e-30f);
    const int e = fast_log2_ceil(ratio);
    scale_inv = exp2f(static_cast<float>(e));
    y_scale = exp2f(static_cast<float>(-e));
  } else {
    scale_inv = local_absmax / max_8bit;
    y_scale = max_8bit / local_absmax;
  }

  if (lane_id == 0) {
    if (SCALE_UE8M0) {
      const uint8_t byte = static_cast<uint8_t>(__float_as_uint(scale_inv) >> 23);
      if (is_column_major) {
        const int64_t col_idx = g / 4;
        const int pack_idx = g % 4;
        auto* so = reinterpret_cast<uint8_t*>(output_s);
        const int64_t slot = col_idx * scale_stride * 4 + row * 4;
        so[slot + pack_idx] = byte;
        // zero-fill the tail of the last (partial) pack on this row
        const int remainder = groups_per_row % 4;
        if (remainder != 0 && g == groups_per_row - 1) {
          for (int extra = remainder; extra < 4; ++extra) so[slot + extra] = 0;
        }
      } else {
        // row-major ue8m0 (not produced upstream; write bytes linearly)
        reinterpret_cast<uint8_t*>(output_s)[gid] = byte;
      }
    } else if (is_column_major) {
      output_s[g * scale_stride + row] = scale_inv;
    } else {
      output_s[row * groups_per_row + g] = scale_inv;
    }
  }

  for (int i = lane_id; i < group_size; i += kThreadsPerGroup) {
    float v;
    if (FUSE_SILU) {
      const T p = input[base + i];
      const T s = input[base + secondary_off + i];
      v = static_cast<float>(static_cast<T>(silu_f(static_cast<float>(p))) * s);
    } else {
      v = static_cast<float>(input[base + i]);
    }
    const float q = fminf(fmaxf(v * y_scale, min_8bit), max_8bit);
    output_q[row * groups_per_row * group_size + g * group_size + i] = cast_quant<DST>(q);
  }
}

inline int dst_min_max(const at::Tensor& output_q, float* lo, float* hi) {
  if (output_q.scalar_type() == at::ScalarType::Char) {
    *lo = -128.0f;
    *hi = 127.0f;
    return 0;  // int8_t
  }
  if (output_q.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    *lo = -448.0f;
    *hi = 448.0f;
    return 1;  // fp8
  }
  TORCH_CHECK(false, "unsupported quant output dtype: ", output_q.scalar_type());
}

inline int input_dtype_kind(const at::Tensor& input) {
  const auto t = input.scalar_type();
  if (t == at::ScalarType::Float) return 0;
  if (t == at::ScalarType::Half) return 1;
  if (t == at::ScalarType::BFloat16) return 2;
  TORCH_CHECK(false, "unsupported quant input dtype: ", t);
}

// output_s is reinterpret_cast to float* and addressed by the kernels as
//   column-major: g * stride(-1) + row   (needs stride(-2) == 1)
//   row-major:    row * groups + g       (needs the exact contiguous layout)
// (ue8m0 packs bytes into the same slots). input/output_q get CHECK_INPUT
// (CUDA+contiguous); output_s legitimately allows the column-major layout,
// so validate its device/dtype/shape/strides here instead — a CPU or
// off-device tensor, a non-float dtype, or padded/overlapping strides
// would fault or silently corrupt the scales (review #80 P3). A size-1
// dim's stride is never dereferenced, so it stays unconstrained (same
// leniency as the k_rope stride(1) fix).
void check_quant_output_s(
    const at::Tensor& output_s,
    const at::Tensor& input,
    int64_t num_rows,
    int64_t groups_per_row) {
  TORCH_CHECK(output_s.is_cuda(), "output_s must be on an HCU device");
  TORCH_CHECK(
      output_s.scalar_type() == at::ScalarType::Float,
      "output_s must be float32, got ", output_s.scalar_type());
  TORCH_CHECK(
      output_s.device() == input.device(),
      "output_s must share input's device");
  TORCH_CHECK(
      output_s.size(-2) == num_rows && output_s.size(-1) == groups_per_row,
      "output_s must be (rows, groups_per_row) = (", num_rows, ", ",
      groups_per_row, "), got ", output_s.sizes());
  const int64_t s0 = output_s.stride(-2), s1 = output_s.stride(-1);
  if (s0 < s1) {  // same predicate the launchers use to pick column-major
    TORCH_CHECK(
        num_rows == 1 || s0 == 1,
        "column-major output_s needs unit row stride, got (", s0, ", ",
        s1, ")");
    // g*s1+row slots stay distinct across groups only when the column
    // stride covers the full row extent — a short s1 (e.g. stride (1,2)
    // on a (4,3) buffer) makes (g, r) collide with (g+1, r-s1) and the
    // kernel silently overwrites earlier scales. A single group has no
    // cross-group collision, so G==1 stays unconstrained (review #80
    // R8 P3).
    TORCH_CHECK(
        groups_per_row == 1 || s1 >= num_rows,
        "column-major output_s stride(-1) must be >= rows, got (", s0,
        ", ", s1, ") for ", num_rows, " rows");
  } else {
    TORCH_CHECK(
        (num_rows == 1 || s0 == groups_per_row) &&
            (groups_per_row == 1 || s1 == 1),
        "row-major output_s must be exactly row-contiguous (stride (",
        groups_per_row, ", 1)), got (", s0, ", ", s1, ")");
  }
}

}  // namespace

void sgl_per_token_group_quant_8bit(
    at::Tensor input,
    at::Tensor output_q,
    at::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  TORCH_CHECK(input.numel() % group_size == 0, "input numel must divide by group_size");
  TORCH_CHECK(
      input.size(input.dim() - 1) % group_size == 0,
      "input's last dim must be a multiple of group_size (whole groups per row)");
  TORCH_CHECK(output_s.dim() == 2, "output_s must be 2-D");
  const int64_t num_groups = input.numel() / group_size;
  const int num_groups_per_row = input.size(input.dim() - 1) / group_size;
  // size(-1) % group_size == 0 plus non-empty input; without this an empty
  // or cross-row-group input makes num_groups_per_row 0 and the size check
  // below would divide by zero (SIGFPE instead of a clean TORCH_CHECK;
  // review #80 R8 P3).
  TORCH_CHECK(num_groups_per_row >= 1, "input must be non-empty");
  check_quant_output_s(
      output_s, input, num_groups / num_groups_per_row, num_groups_per_row);
  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_stride = output_s.stride(1);
  TORCH_CHECK(!scale_ue8m0 || is_column_major, "scale_ue8m0 requires column-major output_s");

  int groups_per_block = 1;
  for (int cand : {16, 8, 4, 2}) {
    if (num_groups % cand == 0) {
      groups_per_block = cand;
      break;
    }
  }
  const int64_t blocks = (num_groups + groups_per_block - 1) / groups_per_block;
  const int threads = groups_per_block * kThreadsPerGroup;

  const c10::cuda::OptionalCUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int in_kind = input_dtype_kind(input);
  float lo, hi;
  const int dst_kind = dst_min_max(output_q, &lo, &hi);
  const float flo = static_cast<float>(min_8bit), fhi = static_cast<float>(max_8bit);

#define LAUNCH_V1(T, DST, COL, UE8)                                              \
  quant8_v1_kernel<T, DST, COL, UE8><<<blocks, threads, 0, stream>>>(           \
      reinterpret_cast<const T*>(input.data_ptr()),                             \
      reinterpret_cast<DST*>(output_q.data_ptr()),                              \
      reinterpret_cast<float*>(output_s.data_ptr()),                            \
      group_size, num_groups, groups_per_block,                                 \
      static_cast<float>(eps), flo, fhi, num_groups_per_row, scale_stride)

#define LAUNCH_V1_UE8(T, DST, COL)          \
  do {                                      \
    if (scale_ue8m0) {                      \
      LAUNCH_V1(T, DST, COL, true);         \
    } else {                                \
      LAUNCH_V1(T, DST, COL, false);        \
    }                                       \
  } while (0)

#define LAUNCH_V1_COL(T, DST)               \
  do {                                      \
    if (is_column_major) {                  \
      LAUNCH_V1_UE8(T, DST, true);          \
    } else {                                \
      LAUNCH_V1_UE8(T, DST, false);         \
    }                                       \
  } while (0)

#define LAUNCH_V1_TYPE(T)                                       \
  do {                                                          \
    if (dst_kind == 0) {                                        \
      LAUNCH_V1_COL(T, int8_t);                                \
    } else {                                                    \
      LAUNCH_V1_COL(T, c10::Float8_e4m3fn);                     \
    }                                                           \
  } while (0)

  if (in_kind == 0) {
    LAUNCH_V1_TYPE(float);
  } else if (in_kind == 1) {
    LAUNCH_V1_TYPE(c10::Half);
  } else {
    LAUNCH_V1_TYPE(c10::BFloat16);
  }
#undef LAUNCH_V1_TYPE
#undef LAUNCH_V1_COL
#undef LAUNCH_V1_UE8
#undef LAUNCH_V1
}

namespace {

// One v2 launch over a (rows, hidden) slab; shared by the plain and the
// per-expert masked paths. row_limit (masked_m + expert offset) is read
// on device; a full-slab launch with static shape keeps the masked path
// CUDA-graph capturable.
void launch_v2(
    const at::Tensor& input,
    at::Tensor& output_q,
    at::Tensor& output_s,
    int64_t group_size,
    bool scale_ue8m0,
    bool fuse_silu_and_mul,
    const int32_t* row_limit = nullptr) {
  const int64_t num_rows = output_q.size(-2);
  const int64_t hidden = output_q.size(-1);
  const int groups_per_row = hidden / group_size;
  TORCH_CHECK(
      hidden % group_size == 0 && groups_per_row >= 1,
      "hidden must be a non-empty multiple of group_size");
  const int64_t num_groups = num_rows * groups_per_row;

  check_quant_output_s(output_s, input, num_rows, groups_per_row);
  const bool is_column_major = output_s.stride(-2) < output_s.stride(-1);
  const int scale_stride = output_s.stride(-1);

  const c10::cuda::OptionalCUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int groups_per_block = 16;  // 16 groups x 16 lanes = 256 threads
  const int64_t blocks = (num_groups + groups_per_block - 1) / groups_per_block;
  const int threads = groups_per_block * kThreadsPerGroup;

  const int in_kind = input_dtype_kind(input);
  float lo, hi;
  const int dst_kind = dst_min_max(output_q, &lo, &hi);
  const int col = is_column_major ? 1 : 0;
  const int64_t hidden_full = fuse_silu_and_mul ? hidden * 2 : hidden;

#define LAUNCH_V2(T, DST, FUSE, UE8)                                             \
  quant8_v2_kernel<T, DST, FUSE, UE8><<<blocks, threads, 0, stream>>>(          \
      reinterpret_cast<const T*>(input.data_ptr()),                             \
      reinterpret_cast<DST*>(output_q.data_ptr()),                              \
      reinterpret_cast<float*>(output_s.data_ptr()),                            \
      row_limit, num_groups, groups_per_row, group_size, lo, hi, scale_stride, col)

#define LAUNCH_V2_UE8(T, DST, FUSE)       \
  do {                                    \
    if (scale_ue8m0) {                    \
      LAUNCH_V2(T, DST, FUSE, true);      \
    } else {                              \
      LAUNCH_V2(T, DST, FUSE, false);     \
    }                                     \
  } while (0)

#define LAUNCH_V2_FUSE(T, DST)            \
  do {                                    \
    if (fuse_silu_and_mul) {              \
      LAUNCH_V2_UE8(T, DST, true);        \
    } else {                              \
      LAUNCH_V2_UE8(T, DST, false);       \
    }                                     \
  } while (0)

#define LAUNCH_V2_DST(T)                  \
  do {                                    \
    if (dst_kind == 0) {                  \
      LAUNCH_V2_FUSE(T, int8_t);          \
    } else {                              \
      LAUNCH_V2_FUSE(T, c10::Float8_e4m3fn); \
    }                                     \
  } while (0)

  if (in_kind == 0) {
    LAUNCH_V2_DST(float);
  } else if (in_kind == 1) {
    LAUNCH_V2_DST(c10::Half);
  } else {
    LAUNCH_V2_DST(c10::BFloat16);
  }
#undef LAUNCH_V2_DST
#undef LAUNCH_V2_FUSE
#undef LAUNCH_V2_UE8
#undef LAUNCH_V2
}

}  // namespace

void sgl_per_token_group_quant_8bit_v2(
    at::Tensor input,
    at::Tensor output_q,
    at::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0,
    bool fuse_silu_and_mul,
    const std::optional<at::Tensor>& masked_m) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  TORCH_CHECK(input.numel() > 0, "input must be non-empty");
  TORCH_CHECK(std::abs(kV2AbsmaxFloor - eps) < 1e-13, "v2 requires eps == 1e-10");
  TORCH_CHECK(
      group_size == 16 || group_size == 32 || group_size == 64 || group_size == 128,
      "Unsupported group_size: ", group_size);
  float lo, hi;
  dst_min_max(output_q, &lo, &hi);
  CHECK_EQ(static_cast<float>(min_8bit), lo);
  CHECK_EQ(static_cast<float>(max_8bit), hi);

  const bool masked_layout = masked_m.has_value();
  TORCH_CHECK(output_s.dim() == (masked_layout ? 3 : 2), "unexpected output_s rank");
  if (masked_layout) {
    // DeepEP layout: input (E, maxT, H(*2 fuse)), output_q (E, maxT, H),
    // output_s (E, maxT, groups_per_row); rows [masked_m[e], maxT) are
    // left untouched. The per-expert row limit is read ON DEVICE by the
    // kernel (row guard), so each launch covers the FULL static slab —
    // CUDA-graph capturable, and a replay with an updated masked_m
    // re-masks correctly (upstream MaskedLayoutScheduler, v2.cu:203-247;
    // the previous host .item() read synced per expert and froze the
    // counts into the captured topology; review #80 P2).
    TORCH_CHECK(output_q.dim() == 3, "output_q must be 3-D with masked_m");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == output_q.size(0),
                "DeepEP layout mismatch");
    TORCH_CHECK(masked_m->dim() == 1 && masked_m->size(0) == output_q.size(0),
                "masked_m must be (num_experts,)");
    TORCH_CHECK(masked_m->scalar_type() == at::ScalarType::Int,
                "masked_m must be int32");
    // The row limit is dereferenced ON DEVICE; a CPU tensor would hand
    // the kernel a host pointer (illegal memory access) and a strided
    // view would read garbage counts (review #80).
    TORCH_CHECK(masked_m->is_cuda(), "masked_m must be on an HCU device");
    TORCH_CHECK(masked_m->is_contiguous(), "masked_m must be contiguous");
    const int32_t* limit_base = masked_m->data_ptr<int32_t>();
    for (int64_t e = 0; e < output_q.size(0); ++e) {
      auto in_slab = input.select(0, e);
      auto q_slab = output_q.select(0, e);
      auto s_slab = output_s.select(0, e);
      launch_v2(in_slab, q_slab, s_slab, group_size, scale_ue8m0,
                fuse_silu_and_mul, limit_base + e);
    }
    return;
  }
  TORCH_CHECK(input.dim() == 2, "input must be 2-D without masked_m");
  launch_v2(input, output_q, output_s, group_size, scale_ue8m0, fuse_silu_and_mul);
}
