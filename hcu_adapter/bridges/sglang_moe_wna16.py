# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""moe_wna16 quant config non-linear layer contract restore

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging


def install(_he=None) -> None:
    # moe_wna16 quant config: upstream get_quant_method runs the
    # modules_to_not_convert check BEFORE any isinstance guard, so a
    # RadixAttention layer whose prefix substring-matches a skipped module
    # (AWQ checkpoints list "self_attn", which also catches
    # "...self_attn.attn") gets an UnquantizedLinearMethod — and RadixAttention
    # then calls create_weights(layer) with one arg against its 6-arg
    # signature. Every other config returns None for layers it doesn't
    # handle; restore that contract for non-linear layers.
    try:
        import sglang.srt.layers.quantization.moe_wna16 as _wna

        _hcu_orig_wna_gqm = _wna.MoeWNA16Config.get_quant_method

        def _hcu_wna_gqm(self, layer, prefix):
            from sglang.srt.layers.linear import LinearBase
            from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

            if not isinstance(layer, (LinearBase, FusedMoE)):
                return None
            return _hcu_orig_wna_gqm(self, layer, prefix)

        _wna.MoeWNA16Config.get_quant_method = _hcu_wna_gqm
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: moe_wna16 RadixAttention guard not installed",
            exc_info=True,
        )
