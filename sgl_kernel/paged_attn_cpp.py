# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Paged attention decode — HIP C++ kernel (fa3-facing, tier-④ cpp-first).

torch cpp_extension JIT (hipcc), same build cache pattern as the JIT rope
patches. See csrc/paged_decode_hip.h for the structural bets vs the triton
kernel (one workgroup per (b, kv-h) covering all G QA heads; lane->token
score mapping with a single full-wave max per page; page-granularity
online-softmax checkpoint).

Gates: numerics vs the fp32 CPU reference at the same 2e-2 tolerance as
the triton kernel's gate; perf A/B against paged_attention_decode_triton
on the same fixture (decode bench rows).
"""

import os

import torch
from torch.utils.cpp_extension import load

_DECL = ""  # header-only kernel + single TU cuda source; no extra decls
_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "paged_decode_hip.h"

using namespace sgl_kernel_hcu;

template <typename T>
void dispatch_decode(
    const T* q, const T* k, const T* v,
    const int32_t* pt, const int32_t* seqlens,
    T* o, float* lse, float scale,
    int64_t B, int64_t H, int64_t HKV, int64_t PT_W) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)B, (unsigned)HKV);
  paged_decode_w64<T, 128, 64><<<grid, 64, 64 * sizeof(float), stream>>>(
      q, k, v, pt, seqlens, o, lse, scale, (int)H, (int)HKV, (int)PT_W);
}

void paged_decode_hip(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor pt, torch::Tensor seqlens,
    torch::Tensor o, torch::Tensor lse, double scale) {
  const int64_t B = q.size(0), H = q.size(1), D = q.size(2);
  const int64_t HKV = k.size(2), PT_W = pt.size(1);
  TORCH_CHECK(D == 128 && k.size(1) == 64, "fast path is D=128/PAGE=64");
  if (q.scalar_type() == torch::kBFloat16) {
    dispatch_decode<hip_bfloat16>(
        reinterpret_cast<const hip_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const hip_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const hip_bfloat16*>(v.data_ptr()),
        pt.data_ptr<int32_t>(), seqlens.data_ptr<int32_t>(),
        reinterpret_cast<hip_bfloat16*>(o.data_ptr()),
        lse.data_ptr<float>(), (float)scale, B, H, HKV, PT_W);
  } else {
    dispatch_decode<__half>(
        reinterpret_cast<const __half*>(q.data_ptr()),
        reinterpret_cast<const __half*>(k.data_ptr()),
        reinterpret_cast<const __half*>(v.data_ptr()),
        pt.data_ptr<int32_t>(), seqlens.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(o.data_ptr()),
        lse.data_ptr<float>(), (float)scale, B, H, HKV, PT_W);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("paged_decode_hip", &paged_decode_hip); }
"""


def _get_ext():
    global _EXT
    try:
        return _EXT
    except NameError:
        pass
    header_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc"
    )
    # old torch cpp_extension.load only takes source FILES; stage the TU once
    build_dir = os.path.join(header_dir, "_build")
    os.makedirs(build_dir, exist_ok=True)
    cu_file = os.path.join(build_dir, "paged_decode_hip_tu.cu")
    with open(cu_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(_SRC)
    _EXT = load(
        name="sgl_paged_decode_hip",
        sources=[cu_file],
        with_cuda=True,
        extra_include_paths=[header_dir],
        # dcc (DTK hipcc) rejects NV --use-fast_math; -O3 suffices (the exp
        # hot path uses __expf intrinsics regardless)
        extra_cuda_cflags=["-O3", "--offload-arch=gfx936"],
        verbose=False,
        build_directory=build_dir,
    )
    return _EXT


def paged_attention_decode_hip(
    q,              # [B, 1, H, D]  bf16/fp16
    k_cache,        # [P, PAGE, HKV, D]
    v_cache,
    page_table,     # [B, PT_W] int32
    cache_seqlens,  # [B] int32
    softmax_scale=None,
):
    B, L, H, D = q.shape
    assert L == 1, "decode face only"
    _, PAGE, HKV, _ = k_cache.shape
    assert D == 128 and PAGE == 64, "C++ fast path: D=128, PAGE=64"
    scale = softmax_scale if softmax_scale is not None else D**-0.5
    seqlens = cache_seqlens.to(torch.int32)
    o = torch.empty(B, 1, H, D, dtype=q.dtype, device=q.device)
    lse = torch.empty(B, 1, H, dtype=torch.float32, device=q.device)
    _get_ext().paged_decode_hip(
        q.reshape(B, H, D).contiguous(), k_cache.contiguous(),
        v_cache.contiguous(), page_table.contiguous(), seqlens.contiguous(),
        o.reshape(B, H, D), lse.reshape(B, H), float(scale),
    )
    return o, lse
