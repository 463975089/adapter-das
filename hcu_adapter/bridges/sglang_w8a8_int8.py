# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CT + explicit w8a8_int8 consumers GEMM import backfill

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging


def install(_he=None) -> None:
    # Both CT and explicit w8a8_int8 consumers can miss the CUDA-gated
    # GEMM import. Bind the existing implementation in their own globals.
    try:
        from hcu_adapter.features.ct_w4a8 import int8_linear

        int8_linear.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: int8_scaled_mm bridge not installed", exc_info=True
        )

    try:
        from hcu_adapter.bridges._sglang import mla_kv

        mla_kv.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: packed MLA KV writer bridge not installed", exc_info=True
        )
