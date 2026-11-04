# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Host-side DeepGEMM/TileKernels runtime knobs with the pinned vendor contract.

No numerics. Anchors: DeepGEMM csrc/runtime/runtime.hpp:78-87 (num SMs),
csrc/jit_kernels/heuristics/runtime.hpp:12-63 (MK alignment; non-SM100
branch), csrc/utils/math.hpp:23-27 (TMA size); TileKernels 2.1.0
tile_kernels/config.py:5-93. The vendor config imports TileLang only for
clamp constants, so the state is mirrored here instead. ``num_sms`` is the
HIP multiprocessor (CU) count and is consumed by HCU kernels only.
"""

import functools

import torch

_LEGACY_MK_ALIGNMENT = 128
_TMA_ALIGNMENT_BYTES = 16
_DTYPE_TO_STR = {
    torch.float32: "fp32",
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "e4m3",
    torch.int8: "e2m1",
}


def dtype_to_str(dtype):
    """Pinned TileKernels dtype names, including int8 packed-E2M1 storage."""
    if dtype not in _DTYPE_TO_STR:
        raise ValueError(f"Unsupported TileKernels dtype: {dtype}")
    return _DTYPE_TO_STR[dtype]


@functools.lru_cache(maxsize=None)
def get_device_num_sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


class _NumSMs:
    def __init__(self):
        self.value = 0

    def set(self, num_sms):
        if not 0 < num_sms <= get_device_num_sms():
            raise AssertionError(f"num_sms must be in (0, {get_device_num_sms()}], got {num_sms}")
        self.value = num_sms

    def get(self):
        return self.value or get_device_num_sms()


_deep_gemm_sms = _NumSMs()
_tile_kernels_sms = _NumSMs()
_mk_alignment = _LEGACY_MK_ALIGNMENT
_token_alignment = 128


def deep_gemm_set_num_sms(num_sms):
    _deep_gemm_sms.set(num_sms)


def deep_gemm_get_num_sms():
    return _deep_gemm_sms.get()


def set_mk_alignment_for_contiguous_layout(new_value):
    global _mk_alignment
    _mk_alignment = new_value


def get_mk_alignment_for_contiguous_layout():
    return _mk_alignment


def get_theoretical_mk_alignment_for_contiguous_layout(expected_m=None):
    # Vendor returns 256 only for arch major 10 (UMMA_N=256); others get 128.
    return _LEGACY_MK_ALIGNMENT


def get_tma_aligned_size(x, element_size):
    if _TMA_ALIGNMENT_BYTES % element_size != 0:
        raise AssertionError(f"element_size {element_size} does not divide {_TMA_ALIGNMENT_BYTES}")
    alignment = _TMA_ALIGNMENT_BYTES // element_size
    return (x + alignment - 1) // alignment * alignment


def tile_kernels_set_num_sms(num_sms):
    _tile_kernels_sms.set(num_sms)


def tile_kernels_get_num_sms():
    return _tile_kernels_sms.get()


def set_token_alignment(token_alignment):
    global _token_alignment
    _token_alignment = token_alignment


def get_token_alignment():
    return _token_alignment
