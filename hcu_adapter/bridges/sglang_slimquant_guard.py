# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""slimquant_w4a8 checkpoint auto-rule (HCU_SLIMQUANT gate)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
from hcu_adapter.infra import hcu_env


def install(_he=None) -> None:
    # slimquant_w4a8 checkpoints (GLM-5-W4A8_V2_6 family): expert int4 unpack
    # + register the quant method. Default ON (the method only activates for
    # checkpoints whose quantization_config names slimquant_w4a8 — inert
    # otherwise); HCU_SLIMQUANT=0 opts out (debug/isolation).
    if hcu_env.enabled("HCU_SLIMQUANT"):
        try:
            from hcu_adapter.features.slimquant import slimquant as _slimquant

            _slimquant.activate()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: slimquant bridge not installed", exc_info=True
            )
