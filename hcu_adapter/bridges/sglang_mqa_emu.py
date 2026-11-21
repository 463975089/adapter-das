# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""gfx936 MQA-logits triton bridge (explicit-enable; torch emu retired from runtime)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
from hcu_adapter.infra import hcu_env


def install(_he=None) -> None:
    # gfx936-class HCUs have no fp8 matrix unit and the DTK aiter MQA-logits
    # kernels hard-gate on gfx942/950: this bridge swaps in the triton
    # paged/ragged arms (strict dispatch, governance 3b-8). 2026-11-19: the
    # auto rule no longer enables the bridge (3b-9) -- explicit
    # HCU_MQA_EMU=1 required; default/auto leaves stock aiter whose arch
    # gate then errors loudly (zero fallback).
    if hcu_env.enabled("HCU_MQA_EMU"):
        try:
            import hcu_adapter.features.dsa.mqa_logits_emu as _mqa_emu

            _mqa_emu.install()
        except ModuleNotFoundError:
            pass  # no sglang importable (adapter-only tooling) — emu irrelevant
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: MQA-logits emu not installed", exc_info=True
            )
