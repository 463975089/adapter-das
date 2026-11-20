# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""DSV4/GLM swiglu-clamp MoE activation triton route (fused_experts_impl assert relax)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
import os


def install(_he=None) -> None:
    # DSV4/GLM swiglu-clamp MoE activation: upstream's fused
    # silu_and_mul_clamp ships only a CUDA-JIT kernel (XPU takes it from
    # sgl_kernel), and fused_experts_impl hard-asserts `_is_cuda or
    # _is_xpu` before calling it -- any swiglu_limit MoE (GLM-5.3-Flash,
    # DeepSeek V4) dies at first forward on HIP. The adapter's triton
    # sgl_kernel.silu_and_mul_clamp has the glm5_next.swiglu_clamped
    # semantics (gate clamps max only, up both sides); route the dsv4 op
    # to it and relax that single assert by recompiling
    # fused_experts_impl's source in place (no sglang file is touched).
    # A blanket `_is_xpu` rebind is NOT safe: fused_moe.py has runtime
    # XPU-only branches that would misroute.
    if os.environ.get("HCU_SWIGLU_CLAMP") != "0":
        try:
            import inspect

            import sgl_kernel.elementwise as _ew

            # 替代档: sgl_kernel.elementwise.silu_and_mul_clamp（本处 replug）
            import sglang.kernels.ops.moe.dsv4 as _dsv4

            _dsv4.silu_and_mul_clamp = _ew.silu_and_mul_clamp
            import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as _fm

            # The swiglu-clamp assert lives in _fused_moe_kernel_sequence
            # (the shared kernel-orchestration helper; the call chain is
            # fused_experts -> [torch.ops] inplace_fused_experts ->
            # fused_experts_impl -> _fused_moe_kernel_sequence, all by
            # module-global lookup). Locate the assert across candidate
            # helpers, strip decorator lines (re-running
            # @register_custom_op would double-register the torch op),
            # recompile in place, and verify the relaxed assert landed.
            _old = "assert _is_cuda or _is_xpu, ("
            _patched = False
            for _name in (
                "_fused_moe_kernel_sequence",
                "fused_experts_impl",
                "fused_experts",
            ):
                _fn = getattr(_fm, _name, None)
                if _fn is None:
                    continue
                _src = inspect.getsource(_fn)
                if _old not in _src:
                    continue
                _lines = _src.split("\n")
                while _lines and _lines[0].lstrip().startswith("@"):
                    _lines.pop(0)
                _src = "\n".join(_lines).replace(
                    _old, "assert _is_cuda or _is_xpu or _is_hip, (", 1
                )
                exec(compile(_src, _fm.__file__, "exec"), _fm.__dict__)
                if _old in inspect.getsource(getattr(_fm, _name)):
                    raise RuntimeError(
                        "swiglu-clamp assert relax did not land"
                    )
                _patched = True
                break
            if not _patched:
                raise RuntimeError(
                    "upstream fused_moe swiglu-clamp assert not found in"
                    " any candidate helper (source drift?)"
                )
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: DSV4 swiglu-clamp MoE activation routed to"
                " adapter triton (silu_and_mul_clamp)"
            )
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: swiglu-clamp bridge not installed",
                exc_info=True,
            )
