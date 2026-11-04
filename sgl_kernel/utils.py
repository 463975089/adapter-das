# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""sgl_kernel.utils — architecture helper surface (§0.5 is_cuda route).

Upstream modules import small arch predicates from the CUDA sgl-kernel
wheel's `sgl_kernel.utils` on the is_cuda branch. On HCU the honest answers:

- `is_arch_support_pdl`: False — Programmatic Dependent Launch is an NVIDIA
  sm90+ feature; nothing on the HIP stack implements it, and enabling it
  would corrupt stream ordering.
"""

import torch


def is_arch_support_pdl() -> bool:
    # PDL is an NVIDIA sm90+ stream feature; the hardware truth on HCU
    # (torch.version.hip is set) is "not supported" regardless of what the
    # CUDA-semantic capability bridge reports.
    if getattr(torch.version, "hip", None) is not None:
        return False
    try:
        major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    except Exception:
        return False
    return major >= 9
