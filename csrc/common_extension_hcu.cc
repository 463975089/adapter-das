// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// torch.library registration for the sgl_kernel native (hipcc) extension.
// Schemas are copied verbatim from upstream sgl-kernel
// csrc/common_extension.cc / common_extension_rocm.cc (v0.5.22) so that
// `torch.ops.sgl_kernel.<op>` resolves identically to the official wheel.
// Only the ops ported in this repository are registered; the python
// shadow package falls back to its triton/torch tiers when this module
// is absent (sgl_kernel/_native.py).

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DispatchKey.h>
#include <torch/library.h>

void concat_mla_k(at::Tensor k, at::Tensor k_nope, at::Tensor k_rope);
void concat_mla_absorb_q(at::Tensor a, at::Tensor b, at::Tensor out);
void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor& output);
void merge_state_v2(
    at::Tensor v_a,
    at::Tensor s_a,
    at::Tensor v_b,
    at::Tensor s_b,
    at::Tensor v_merged,
    at::Tensor s_merged);
void sgl_per_token_group_quant_8bit(
    at::Tensor input,
    at::Tensor output_q,
    at::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0);
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
    const std::optional<at::Tensor>& masked_m);

// FRAGMENT (not the singleton TORCH_LIBRARY): another sgl_kernel
// native library may legitimately coexist in exotic dev containers
// (e.g. an official sglang-das wheel); fragments with identical
// schemas merge. The python probe (sgl_kernel/_native.py) never
// imports this module when the namespace is already registered, so
// duplicate (op, key) impls cannot happen in supported flows.
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("concat_mla_k(Tensor! k, Tensor k_nope, Tensor k_rope) -> ()");
  m.impl("concat_mla_k", c10::DispatchKey::CUDA, &concat_mla_k);

  m.def("concat_mla_absorb_q(Tensor a, Tensor b, Tensor! out) -> ()");
  m.impl("concat_mla_absorb_q", c10::DispatchKey::CUDA, &concat_mla_absorb_q);

  m.def("copy_to_gpu_no_ce(Tensor input, Tensor! output) -> ()");
  m.impl("copy_to_gpu_no_ce", c10::DispatchKey::CUDA, &copy_to_gpu_no_ce);

  // schema identical to the upstream aot python wrapper face (aot
  // csrc/common_extension_rocm.cc:70): destination tensors are explicit
  m.def(
      "merge_state_v2(Tensor v_a, Tensor s_a, Tensor v_b, Tensor s_b, Tensor! v_merged, Tensor! s_merged) -> ()");
  m.impl("merge_state_v2", c10::DispatchKey::CUDA, &merge_state_v2);

  m.def(
      "sgl_per_token_group_quant_8bit(Tensor input, Tensor! output_q, Tensor! output_s, int group_size,"
      " float eps, float fp8_min, float fp8_max, bool scale_ue8m0) -> ()");
  m.impl("sgl_per_token_group_quant_8bit", c10::DispatchKey::CUDA, &sgl_per_token_group_quant_8bit);

  m.def(
      "sgl_per_token_group_quant_8bit_v2(Tensor input, Tensor! output_q, Tensor! output_s, int group_size,"
      " float eps, float fp8_min, float fp8_max, bool scale_ue8m0, bool fuse_silu_and_mul, Tensor? masked_m) -> ()");
  m.impl("sgl_per_token_group_quant_8bit_v2", c10::DispatchKey::CUDA, &sgl_per_token_group_quant_8bit_v2);
}
