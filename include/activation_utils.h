// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Minimal excerpt of upstream sgl-kernel aot include/utils.h (v0.5.22),
// carrying ONLY what the vendored gated-activation chain needs: the ROCm
// dispatch-case macros, the HIP vec/math header includes and the
// DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16 macro (all verbatim).
//
// The full utils.h is deliberately NOT vendored: its FP8 section is arch
// gated (#if HIP_FP8_TYPE_E4M3 ... #error "fp8 is not supported in this
// processor (arch < gfx942)") and hard-fails gfx936 builds through the
// DTK in-place hipify path (observed 2026-11-19, utils_hip.h:389).

#pragma once

#ifdef USE_ROCM

#include <sstream>

#include "hip/hip_math_def.h"
#include "hip/hip_vec_dtypes.h"

// Adapted from flashinfer-rocm [PR#491](https://github.com/flashinfer-ai/flashinfer/pull/491)
#define _DISPATCH_CASE_F16(c_type, ...) \
  case at::ScalarType::Half: {          \
    using c_type = __half;              \
    return __VA_ARGS__();               \
  }

#define _DISPATCH_CASE_BF16(c_type, ...) \
  case at::ScalarType::BFloat16: {       \
    using c_type = __hip_bfloat16;       \
    return __VA_ARGS__();                \
  }
#endif  // USE_ROCM

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(pytorch_dtype, c_type, ...)           \
  [&]() -> bool {                                                                        \
    switch (pytorch_dtype) {                                                             \
      case at::ScalarType::Float: {                                                      \
        using c_type = float;                                                            \
        return __VA_ARGS__();                                                            \
      }                                                                                  \
        _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                          \
        _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                         \
      default:                                                                           \
        std::ostringstream oss;                                                          \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << pytorch_dtype; \
        TORCH_CHECK(false, oss.str());                                                   \
        return false;                                                                    \
    }                                                                                    \
  }()
