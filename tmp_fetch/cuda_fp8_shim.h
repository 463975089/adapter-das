// HCU adapter cuda->hip forwarder + __nv_* token aliases
// (image sglang-0.5.18 JIT csrc uses CUDA fp8/bf16 tokens; DTK hip headers
//  provide __hip_* equivalents. Aliases let those sources compile as-is.)
#pragma once
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

using __nv_fp8_storage_t         = __hip_fp8_storage_t;
using __nv_fp8x2_storage_t       = __hip_fp8x2_storage_t;
using __nv_fp8x4_storage_t       = __hip_fp8x4_storage_t;
using __nv_fp8_interpretation_t  = __hip_fp8_interpretation_t;
using __nv_saturation_t          = __hip_saturation_t;
using __nv_fp8x2_e4m3            = __hip_fp8x2_e4m3_fnuz;
using __nv_fp8x4_e4m3            = __hip_fp8x4_e4m3_fnuz;
using __nv_bfloat16              = __hip_bfloat16;
using __nv_bfloat162             = __hip_bfloat162;
using __nv_bfloat16_raw          = __hip_bfloat16_raw;
using __nv_bfloat162_raw         = __hip_bfloat162_raw;

#define __NV_E4M3      __HIP_E4M3
#define __NV_E5M2      __HIP_E5M2
#define __NV_E4M3_FNUZ __HIP_E4M3_FNUZ
#define __NV_SATFINITE __HIP_SATFINITE
#define __NV_NOSAT     __HIP_NOSAT

#define __nv_cvt_float2_to_fp8x2   __hip_cvt_float2_to_fp8x2
#define __nv_cvt_float_to_fp8      __hip_cvt_float_to_fp8
#define __nv_cvt_fp8x2_to_fp16x2   __hip_cvt_fp8x2_to_halfraw2
#define __nv_cvt_fp16x2_to_fp8x2   __hip_cvt_halfraw2_to_fp8x2
#define __nv_cvt_bfloat16raw2_to_fp8x2 __hip_cvt_bfloat16raw2_to_fp8x2
#define __nv_tanhf tanhf
