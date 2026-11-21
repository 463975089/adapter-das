# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""fa3 attention backend torch-emulation route (SGLANG_USE_SGL_FA3_KERNEL)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
import os


def install(_he=None) -> None:
    # fa3 attention backend: the ops layer (flash_attention_v3.py) hard-gates
    # on NVIDIA — `_is_fa3_supported` returns False whenever torch.version.cuda
    # is None (always on HIP), making flash_attn_with_kvcache raise and
    # flash_attn_varlen_func fall back to the flash_attn wheel (numerically
    # broken on this stack). The adapter ships a torch emulation in
    # sgl_kernel.flash_attn; SGLANG_USE_SGL_FA3_KERNEL=1 routes _load_fa3_kernels
    # to it directly (also skipping the kernels-community download attempt).
    try:
        from hcu_adapter.infra import hcu_env as _henv

        if _henv.enabled("HCU_FA3_EMU"):
            os.environ.setdefault("SGLANG_USE_SGL_FA3_KERNEL", "1")
            # 替代档: fa3 emu 门翻转挂载面（HCU_FA3_EMU；包面无该门出口）
            import sglang.kernels.ops.attention.flash_attention_v3 as _fa3v

            _fa3v._is_fa3_supported = lambda device=None: True
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: fa3 emu routing bridge not installed", exc_info=True
        )
