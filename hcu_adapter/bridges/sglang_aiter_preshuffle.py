# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""DSA aiter preshuffle paged-MQA gate force-on (page/table/tile decisions only)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
import sys
from hcu_adapter.infra import hcu_env


def install(_he=None) -> None:
    # Force the DSA "aiter preshuffle paged-MQA" gate on. The DTK wheel ships
    # working triton preshuffle MQA-logits kernels (pa_mqa_logits etc.), but
    # upstream's gate keys on SGLANG_USE_AITER — setting that env would also
    # flip unrelated fp8 weight-shuffle/gemm routes. Rebinding the predicate
    # in its home module before any consumer's byname import copies it makes
    # only the DSA page-size/table/preshuffle-tile decisions follow the
    # preshuffle path (page_size=64*index_kpool, 16-aligned cache tiles).
    try:
        import sglang.srt.layers.attention.dsa.utils as _dsa_utils
        from sglang.srt.utils import common as _dsa_common

        _dsa_utils.is_hip = _dsa_common.is_hip
        _dsa_utils.is_cuda = _dsa_common.is_cuda

        _hcu_orig_gate = _dsa_utils.aiter_can_use_preshuffle_paged_mqa
        if hcu_env.resolve("HCU_DSA_PRESHUFFLE")[0] == "0":
            # Legacy page_size=1 / KVBlockSize=1 DSA path (paired with the
            # HCU_MQA_EMU torch emulation on gfx936-class HCUs).
            def _hcu_gate():
                return False

            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: DSA aiter-preshuffle gate forced OFF (legacy"
                " page_size=1 path)"
            )
        else:

            def _hcu_gate():
                return True

            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: DSA aiter-preshuffle gate forced on"
                " (DTK wheel kernels)"
            )
        # The `import ...utils` above initializes part of the sglang.srt
        # tree, and consumers bind this predicate byname DURING that import
        # chain (mem_cache/memory_pool.py) or cache it into a module-level
        # `_use_aiter_preshuffle` constant (dsa/dsa_indexer.py,
        # kernels/.../index_buf_accessor.py) -- all before the rebind line
        # below runs. Sweep every already-imported module and rebind both
        # the function name and those cached constants; modules imported
        # later copy the patched attribute and need no fixup.
        _dsa_utils.aiter_can_use_preshuffle_paged_mqa = _hcu_gate
        for _m in list(sys.modules.values()):
            _d = getattr(_m, "__dict__", None)
            if not _d:
                continue
            if _d.get("aiter_can_use_preshuffle_paged_mqa") is _hcu_orig_gate:
                _d["aiter_can_use_preshuffle_paged_mqa"] = _hcu_gate
            if "_use_aiter_preshuffle" in _d:
                _d["_use_aiter_preshuffle"] = _hcu_gate()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: preshuffle gate patch not installed", exc_info=True
        )
