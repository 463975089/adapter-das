// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once
/* HCU/DTK build shim — flashinfer's _hip headers (vec_dtypes_hip.cuh)
 * #include <cuda_fp8.h> and use the CUDA spellings. DTK has no standalone
 * cuda_fp8.h; its cuda-12 compat tree ships a forwarding shell whose
 * closure drags in headers that conflict with hipcc (thrust/vector_types,
 * cudaError enums). This shim maps the handful of CUDA spellings onto
 * DTK's native hip fp8 API (structs __hip_fp8_e4m3/__hip_fp8_e5m2 and
 * the __hip_cvt_* intrinsics) with zero new code paths.
 */

#include <hip/amd_detail/amd_hip_fp8.h>

using __nv_fp8_e4m3 = __hip_fp8_e4m3;
using __nv_fp8_e5m2 = __hip_fp8_e5m2;
using __nv_fp8_storage_t = __hip_fp8_storage_t;
using __nv_fp8x2_storage_t = __hip_fp8x2_storage_t;

#define __NV_SATFINITE __HIP_SATFINITE
#define __NV_E4M3 __HIP_E4M3
#define __NV_E5M2 __HIP_E5M2

#define __nv_cvt_float2_to_fp8x2(f2, sat, interp) \
  __hip_cvt_float2_to_fp8x2((f2), (sat), (interp))
