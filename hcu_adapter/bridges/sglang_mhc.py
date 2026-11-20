# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash MHC hyper-connection emu (2026-10-16)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
import torch
from hcu_adapter.infra import hcu_env


def install(_he=None) -> None:
    # 13. GLM-5.3-Flash MHC hyper-connection emu (2026-10-16): official
    #     v0.5.21 gates the fused mix/combine fast kernels only on shape and
    #     is_cuda — both hold on HIP — so every layer boundary (2x45 calls
    #     per forward) runs fused_hc_mix (persistent CUDA kernel with
    #     device-scope atomics) and the hc_combine .cuh tvm_ffi JIT, neither
    #     ever validated on gfx938. A subtly wrong combine pollutes the
    #     residual stream gradually, matching the word-soup signature
    #     (stream rms 1.4 -> 5.4, max -> 194, flat logits). HCU_MHC_EMU=1
    #     routes both to the pure-torch reference math transcribed verbatim
    #     from hyperconnection.py's _mix_compute/_combine_compute (fp32).
    if hcu_env.enabled("HCU_MHC_EMU"):
        try:
            import torch.nn.functional as _Fm

            # JIT 唯一实现（hc_combine 无替代档；供应商预编译提报线）
            import sglang.kernels.ops.elementwise.hc_combine as _hcc
            import sglang.srt.layers.hyperconnection as _hy

            def _hcu_hc_combine(*a, **kw):
                block_output, residual, normed_residual, w_inject = a[:4]
                hc, hs = int(a[4]), int(a[5])
                bo = block_output.float()
                r = residual.float().unflatten(-1, (hc, hs))
                gate = 2.0 * torch.sigmoid(
                    _Fm.linear(normed_residual.float(), w_inject.float()) / hc
                )
                inj = bo.unsqueeze(-2) * gate.unsqueeze(-1)
                out = (r + inj).flatten(-2)
                return out.to(block_output.dtype)

            def _hcu_fused_hc_mix(x, w_down, w_up, hc, hs):
                xf = x.float()
                m = _Fm.silu(_Fm.linear(xf, w_down.float()) / hc)
                m = _Fm.linear(m, w_up.float())
                m = torch.sigmoid(m).unflatten(-1, (hc, hs))
                out = (m * xf.unflatten(-1, (hc, hs))).mean(dim=-2)
                return out.to(x.dtype)

            _hy.fused_hc_mix = _hcu_fused_hc_mix
            _hcc.hc_combine = _hcu_hc_combine
            _hcc.hc_combine_split = _hcu_hc_combine
            print(
                "[HCU mhcemu] fused_hc_mix + hc_combine routed to torch emu",
                flush=True,
            )
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: mhc emu not installed", exc_info=True
            )
