# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer rotate_activation hadamard + prefill/decode backend validator

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
import sys


def install(_he=None) -> None:
    # DSA indexer rotate_activation: the upstream hadamard op is CUDA-only JIT
    # (warp-shuffle intrinsics the DTK toolchain rejects); the HIP arm of
    # the caller imports fast_hadamard_transform, which this wheel set
    # lacks. Triton radix-2 butterfly by default (HCU_HADAMARD_IMPL),
    # torch emu kept as the =emu reference arm.
    try:
        from hcu_adapter.features.hadamard import hadamard_triton

        hadamard_triton.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: hadamard triton bridge not installed", exc_info=True
        )

    # DSA prefill/decode backend validator: the upstream constraint check
    # refuses the triton DSA backends off-HIP ("ROCm/HIP only"), but it
    # receives hip=get_platform().is_hip — which the platform seam flips to
    # False. The triton DSA kernels are precisely what runs on HCU hardware
    # (a ROCm-class card): validate the constraint with the hardware truth
    # (hip=True) while every other judgement keeps the CUDA semantics.
    # Governance §0.5: judgement layers translated, hardware stays truthful.
    try:
        import hcu_adapter.infra.registry as _hreg
        import sglang.srt.arg_groups.overrides as _dsa_ov

        def _hcu_dsa_constraint_hook(module):
            _orig = module._check_dsa_backend_constraints

            def _hcu_check(*args, **kwargs):
                # v0.5.22 grew dcp_size (kwarg); stay pass-through so later
                # upstream signature drift cannot break the seam again.
                kwargs["hip"] = True
                return _orig(*args, **kwargs)

            _hreg.register(
                "platform_detect",
                _hcu_check,
                target="sglang.srt.arg_groups.overrides:_check_dsa_backend_constraints",
                mechanism=_hreg.MECH_SEAM,
                detail="dsa-backend-validator hip=hardware-truth",
            )
            return _hcu_check

        _fn = _hcu_dsa_constraint_hook(_dsa_ov)
        _dsa_ov._check_dsa_backend_constraints = _fn
        for _m in list(sys.modules.values()):
            _d = getattr(_m, "__dict__", None)
            if _d and _d.get("_check_dsa_backend_constraints") is not _fn and (
                getattr(_d.get("_check_dsa_backend_constraints", None), "__module__", "")
                == "sglang.srt.arg_groups.overrides"
            ):
                _d["_check_dsa_backend_constraints"] = _fn
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: DSA backend validator seam installed "
            "(hip=hardware truth inside cuda semantics)"
        )
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: DSA validator seam not installed", exc_info=True
        )
