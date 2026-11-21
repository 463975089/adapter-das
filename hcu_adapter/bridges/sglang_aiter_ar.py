# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""aiter all-reduce gate (custom AR retired; auto-rule + kill-switch precedence)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging


def install(_he=None) -> None:
    # Custom all-reduce RETIRED (2026-10-19, R-13#1 disposition): the
    # homegrown triton custom AR (sgl_kernel/allreduce shim ->
    # hcu_adapter/triton_ar.py + ar_ext.cu) was convicted in R-01
    # (device-scope rendezvous does not order host-shm payload stores;
    # skip-think 48/80) and the planned rendezvous fix is moot — the new
    # aiter.dist GraphSafe AR passed the full stress matrix on the
    # current image (4-rank and 8-rank, eager + graph-replay, 15/15 per
    # topology) and has been the production landing all along. Files
    # deleted; git history has the protocol archaeology. The aiter auto
    # rule below is the sole AR gate now.

    # AiterCustomAllreduce silently corrupts data when the CUDA graph it was
    # captured into replays on gfx936-class HCUs (eager all-reduce is fine).
    # sglang selects it as the AMD default because the C++ sgl_kernel
    # allreduce extension has no HIP build; the symptom is decode tokens
    # diverging from the first graph-replayed step while eager decode and
    # prefill stay correct. Refuse the dispatch on this arch — the exception
    # is caught in parallel_state's init and the group falls back to NCCL,
    # which is graph-correct.
    #
    # Kill-switch precedence (kill_switch.py): HCU_DISABLE_OPS=aiter_ar
    # refuses on ANY arch (triage); arch auto-rule via hcu_env (old-aiter
    # layout on gfx93* refuses — graph-replay corruption, 1eb038d; the new
    # aiter.dist GraphSafe class passed the full stress suite 2026-10-10);
    # explicit HCU_AITER_AR=1 overrides the auto-rule only.
    try:
        from hcu_adapter.infra import kill_switch as _ks

        _ks.log_state()
    except Exception:
        _ks = None
    _ar_val, _ar_src = (
        _he.resolve("HCU_AITER_AR") if _he else ("0", "default")
    )
    _ar_refuse = bool(_ks and _ks.op_disabled("aiter_ar")) or \
        str(_ar_val).startswith("refused")
    _ar_reason = (
        "disabled via HCU_DISABLE_OPS (triage)"
        if _ks and _ks.op_disabled("aiter_ar")
        else f"{_ar_val} [{_ar_src}]"
    )
    if _ar_refuse:
        try:
            import sglang.srt.distributed.device_communicators.\
                custom_all_reduce as _car

            def _hcu_no_aiter_ar(group, device):
                raise RuntimeError(
                    f"HCU adapter: AiterCustomAllreduce disabled"
                    f" ({_ar_reason}; falling back to NCCL)."
                    " Set HCU_AITER_AR=1 to re-enable (auto-rule only) or"
                    " remove aiter_ar from HCU_DISABLE_OPS."
                )

            _car.dispatch_custom_allreduce = _hcu_no_aiter_ar
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: aiter custom all-reduce refused (%s);"
                " NCCL all-reduce instead", _ar_reason
            )
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: aiter AR gate not installed", exc_info=True
            )

    elif _ar_val == "1" or str(_ar_val).startswith("allowed"):
        try:
            from hcu_adapter.bridges import sglang_allreduce

            sglang_allreduce.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: SGLang AITER all-reduce bridge not installed",
                exc_info=True,
            )
