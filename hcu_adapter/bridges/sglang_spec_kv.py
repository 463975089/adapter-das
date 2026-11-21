# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Spec-decode KV-loc safety (2026-10-17)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
import torch
from hcu_adapter.infra import hcu_env


def _install_idle_verify_metadata(verify_type):
    from hcu_adapter.infra import registry
    from hcu_adapter.infra.kill_switch import op_disabled

    if op_disabled("nextn_idle_metadata") or hasattr(verify_type, "kv_indptr"):
        return False
    registry.register(
        "platform_detect", "spec-idle-kv-metadata",
        target="sglang.srt.speculative.eagle_info.EagleVerifyInput.kv_indptr",
        mechanism=registry.MECH_REBIND,
        detail="missing optional decode metadata defaults to None",
    )
    verify_type.kv_indptr = None
    return True


def install(_he=None) -> None:
    # 14. Spec-decode KV-loc safety (2026-10-17): stock v0.5.21's
    #     get_last_loc dispatcher routes is_cuda-semantics stacks to
    #     get_last_loc_triton -- the kernel whose own in-tree comment
    #     documents a Triton HIP miscompile (mixed-width int32->int64
    #     store) producing OUT-OF-RANGE last_loc values under EAGLE +
    #     page_size>1. GLM-5.3 MTP serving on this stack showed exactly
    #     that signature (accept len ~1.0 + target hidden drift +
    #     run-to-run nondeterminism; bisected green without MTP).
    #     sglang-das fixes it for hcu via their hcu_get_last_loc; here we
    #     rebind to the pure-torch variant (same math, page-agnostic).
    #     HCU_SPEC_LOC_SAFE=0 restores stock dispatch.
    if hcu_env.resolve("HCU_SPEC_LOC_SAFE")[0] == "1" and torch.version.hip:
        try:
            import sys as _sys2
            from sglang.srt.mem_cache import allocation as _alloc

            _alloc.get_last_loc = _alloc.get_last_loc_torch
            print(
                "[HCU specloc] get_last_loc -> torch (EAGLE+page>1"
                " out-of-range last_loc guard)",
                file=_sys2.stderr, flush=True,
            )
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: spec-loc safe routing not installed",
                exc_info=True,
            )

    # DSV4 CT-int8 q/kv scales must use the existing fused loader arm.
    # Extend its endswith arguments at runtime after operator/vendor setup.
    try:
        from hcu_adapter.features.dsv4 import dsv4_wqkv_scale_patch

        dsv4_wqkv_scale_patch.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: DSV4 CT scale loader not installed", exc_info=True,
        )

    # HYV4 load_weights renames every .weight_scale to fp8 _inv; CT int8
    # keeps .weight_scale. Class-local loader wrapper, no source writes.
    try:
        from hcu_adapter.features.dsv4 import hyv4_ct_scale_patch

        hyv4_ct_scale_patch.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: HYV4 CT scale naming not installed", exc_info=True,
        )

    # Import quantization only after operator/vendor setup. The original
    # loader receives a private cast interface; community files stay intact.
    try:
        from hcu_adapter.features.dsv4 import fp4_expert_load

        fp4_expert_load.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: FP4 expert runtime loader not installed", exc_info=True,
        )

    # EP token padding needs a reserved request row in V4.1 metadata.
    try:
        from hcu_adapter.features.dsv41 import dsv41_request_indices

        dsv41_request_indices.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: V4.1 padded request indices not installed",
            exc_info=True,
        )

    # Import speculative scheduling only after operator/vendor setup.
    # Private metadata allocation preserves community source and aliases.
    try:
        from hcu_adapter import eagle_hcu_sampling

        eagle_hcu_sampling.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: native HCU MTP sampling policy not installed", exc_info=True,
        )

    # Idle verify objects carry no kv_indptr on this stack; default it to
    # None so the attention backend builds the ordinary KV index instead
    # of raising AttributeError (#149 nextn fix, 2026-11-19).
    try:
        if torch.version.hip is not None:
            from sglang.srt.speculative.eagle_info import EagleVerifyInput

            if _install_idle_verify_metadata(EagleVerifyInput):
                logging.getLogger("hcu.adapter").warning(
                    "HCU adapter: absent idle verify KV metadata defaults to None"
                )
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: idle verify metadata not installed", exc_info=True,
        )
