# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""kernel-selector CUDA capability check HIP acceptance (AOT sgl_kernel)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging


def install(_he=None) -> None:
    # Kernel-selector capability gap: ops like gemm.fp8_scaled_mm declare
    # AOT-capability {CapabilityRequirement.CUDA}, but PlatformInfo.detect()
    # reports device_type="hip" on HCU, so HIP fails `device != platform.device`
    # and select_kernel raises "no backend usable on device 'hip'" — even
    # though the AOT backend (our sgl_kernel) runs fine here. Bridge: relax
    # the CUDA check to also accept HIP platforms (the backward-compatible
    # reading of "cuda" in CapabilityRequirement docs, and what every other
    # `_is_cuda = torch.cuda.is_available()` call site already assumes).
    try:
        from sglang.kernels.spec import CapabilityRequirement as _CR
        from sglang.kernels.spec import DeviceType as _DT

        _orig_sat = _CR.is_satisfied_by

        def _hcu_is_satisfied_by(self, platform):
            if (
                self.device == _DT.CUDA
                and platform.device == _DT.HIP
            ):
                # arch-bounded CUDA windows (e.g. SM90-only TORCH backends)
                # still must NOT match: HIP has no meaningful SM number.
                if self.min_cuda_arch is None and self.max_cuda_arch is None:
                    return True
                return False
            return _orig_sat(self, platform)

        _CR.is_satisfied_by = _hcu_is_satisfied_by

        # Scope the relaxation away from kernel *dispatch*. The fused-op
        # priority lists JIT before AOT, and its capability gate must keep
        # upstream's HIP truth: inheriting the CUDA->HIP relaxation above
        # makes a CUDA-only JIT backend auto-select on HIP even when AOT
        # (our sgl_kernel) is registered for HIP — the 33s
        # sgl_kernel_jit_activation build observed in the L1 kernels
        # segment. The relaxation exists for the fixed-path selector
        # (select_kernel, below), not for choosing a provenance; rebind
        # the capability predicate imported into fused_op so only that
        # module keeps the original check. HIP-native backends (AOT =
        # CUDA+HIP, AITER = HIP) stay eligible, CUDA-only JIT does not.
        import sglang.kernels.fused_op as _fop

        def _hcu_fused_capabilities_satisfied(capabilities, platform):
            if isinstance(capabilities, _CR):
                capabilities = (capabilities,)
            return (not capabilities) or any(
                _orig_sat(c, platform) for c in capabilities
            )

        _fop.capabilities_satisfied = _hcu_fused_capabilities_satisfied

        # The relaxation above can make upstream CUDA-declared specs eligible
        # alongside a spec that natively declares HIP — e.g. moe.topk_softmax
        # registers AOT(sgl_kernel, _HIP) + JIT(_CUDA): on real ROCm only AOT
        # is eligible, but with the bridge both pass and select_kernel raises
        # "multiple backends usable". (moe_align_block_size, same shape, is
        # explicitly pinned to AOT upstream.) Disambiguate the way the platform
        # intends: when our forcing created the ambiguity, prefer the specs
        # whose capability requirement is satisfied WITHOUT the relaxation.
        import sglang.kernels.selector as _sel
        from sglang.kernels.registry import registry as _kreg

        _orig_select_kernel = _sel.select_kernel

        def _hcu_select_kernel(op, backend=None):
            try:
                return _orig_select_kernel(op, backend=backend)
            except ValueError as e:
                if "multiple backends usable" not in str(e):
                    raise
                platform = _sel._platform()
                native = [
                    s
                    for s in _kreg.get(op)
                    if s.is_available(platform)
                    and all(_orig_sat(c, platform) for c in s.capabilities)
                ]
                if len(native) == 1:
                    return native[0]
                raise

        _sel.select_kernel = _hcu_select_kernel
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: capability HIP bridge not installed", exc_info=True
        )
