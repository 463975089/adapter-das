# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""sgl_kernel.quantization — GGUF kernel surface (import closure, gap-listed).

Upstream `sglang.srt.layers.quantization.gguf` imports these five symbols
from sgl_kernel.quantization **unconditionally on the is_cuda branch** (§0.5
route). The CUDA AOT kernels they name have no triton/vendor port yet, so the
symbols exist and raise loudly at call time; GGUF models are outside the
current validation matrix. When a GGUF model needs to run, implement
ggml_dequantize (block-format tables) + the GEMMs as dequantize+matmul torch
reference, then follow the three-tier routing for optimization.
"""

import hcu_adapter.registry as _reg


def _stub(name):
    def _missing(*args, **kwargs):
        raise NotImplementedError(
            f"sgl_kernel.quantization.{name}: GGUF kernels have no HCU port "
            "yet (gap list). Need ggml block-format dequantize + torch "
            "reference GEMM before GGUF models can run."
        )

    _missing.__name__ = name
    return _missing


ggml_dequantize = _stub("ggml_dequantize")
ggml_moe_a8 = _stub("ggml_moe_a8")
ggml_moe_a8_vec = _stub("ggml_moe_a8_vec")
ggml_moe_get_block_size = _stub("ggml_moe_get_block_size")
ggml_mul_mat_a8 = _stub("ggml_mul_mat_a8")
ggml_mul_mat_vec_a8 = _stub("ggml_mul_mat_vec_a8")

_reg.register(
    "platform_detect",
    _stub("gguf-surface"),
    target="sgl_kernel.quantization:ggml_*",
    mechanism=_reg.MECH_SEAM,
    detail="import closure only; loud stubs (gap list)",
)
