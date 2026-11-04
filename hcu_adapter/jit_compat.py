# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-to-HIP forwarding header definitions for adapter-owned JIT inputs.

jit_sources materializes these headers in its immutable compiler bundle and
adds that directory to the original compiler request. Neither community
sources nor the installed DTK include tree are written. The aliases retain
the existing DTK conversion contract; actual numeric validation is separate.
"""

import logging

LOG = logging.getLogger("hcu.adapter")

# CUDA header name -> lines forwarding to the HIP/DTK equivalent
# (double-include safe: HIP headers have their own guards; repeated
# identical `using` aliases are likewise legal C++).
_FORWARDS = {
    "cuda_bf16.h": [
        "#pragma once",
        "#include <hip/hip_bf16.h>",
        # csrc written for CUDA gets __nv_bfloat16 only via this header,
        # so the alias belongs here (same argument as the fp8 set below).
        "using __nv_bfloat16 = __hip_bfloat16;",
    ],
    "cuda_fp16.h": [
        "#pragma once",
        "#include <hip/hip_fp16.h>",
    ],
    "cuda_fp8.h": [
        "#pragma once",
        "#include <hip/hip_fp8.h>",
        # __nv_* token aliases (empirically-required set, box16 2026-10-25;
        # note the _fnuz suffix on the HIP fp8 type name).
        "using __nv_fp8x2_storage_t = __hip_fp8x2_storage_t;",
        "using __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3_fnuz;",
        "#define __NV_E4M3 __HIP_E4M3",
        "#define __NV_SATFINITE __HIP_SATFINITE",
        "#define __nv_cvt_float2_to_fp8x2 __hip_cvt_float2_to_fp8x2",
        "#define __nv_cvt_fp8x2_to_halfraw2 __hip_cvt_fp8x2_to_halfraw2",
    ],
    "cuda.h": [
        "#pragma once",
        "#include <hip/hip_runtime.h>",
    ],
    "cuda_runtime.h": [
        "#pragma once",
        "#include <hip/hip_runtime.h>",
    ],
    "vector_types.h": [
        "#pragma once",
        "#include <hip/hip_runtime.h>",
    ],
    "cub/block/block_load.cuh": [
        "#pragma once",
        "#include <hipcub/block/block_load.hpp>",
        "namespace cub = hipcub;",
    ],
    "cub/block/block_store.cuh": [
        "#pragma once",
        "#include <hipcub/block/block_store.hpp>",
        "namespace cub = hipcub;",
    ],
    "cub/block/block_scan.cuh": [
        "#pragma once",
        "#include <hipcub/block/block_scan.hpp>",
        "namespace cub = hipcub;",
    ],
}

_MARK = "// HCU adapter cuda->hip forwarder"


def install():
    """Compatibility entry: forwarders now live in the compiler input cache."""
    from .jit_sources import install as install_inputs

    return install_inputs()
