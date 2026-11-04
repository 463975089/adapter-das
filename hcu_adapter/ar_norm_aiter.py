# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""R-09 (1): aiter fused all-reduce + RMSNorm seam for sglang layernorm.

Upstream v0.5.21 carries the whole fused AR+RMSNorm machinery, but its
candidate gate is architecturally closed on HCU:

  layer_boundary/fusions/allreduce.complete_attention_input /
  complete_ffn_input try the fused kernel only when
  ``apply_aiter_all_reduce_fusion`` (needs SGLANG_USE_AITER -- the global
  gate that would flip every aiter path upstream-wide) or
  ``apply_flashinfer_allreduce_fusion`` (needs sm90/sm100) is true.
  Neither ever holds on HIP, so ``plan.norm.forward_with_allreduce_fusion``
  -- and the aiter GraphSafe fused kernels the tp coordinator already
  owns -- are unreachable without opening the global gate.

Two symbols, both surgical:

* ARM: rebind ``apply_flashinfer_allreduce_fusion`` (definition site
  ``layer_boundary/residual/add_norm.py`` plus the from-import copies in
  ``layer_boundary/exit.py`` and ``layer_boundary/fusions/allreduce.py``)
  with the HCU eligibility -- arms the candidate try AND the producer-side
  reduction deferral in ``exit._can_defer_ffn_reduction``.
* DISPATCH: rebind ``flashinfer_allreduce_residual_rmsnorm`` -- the entry
  the layernorm else-branch calls once the candidate is armed -- at
  ``GroupCoordinator.fused_allreduce_rmsnorm`` (parallel_state.py), which
  dispatches to the aiter class's ``custom_fused_ar_rms`` / ``fused_ar_rms``.

8-rank probe evidence (nmz26 BW1102, tmp_fetch/probe_r09_arnorm.py,
2026-10-19): parity vs fp64-exact all-reduce is 1 ulp (norm) / 0 ulp
(residual) -- *better* than the rccl reference (3-8 ulps; the IPC kernel
sums exactly where rccl accumulates in ring order); fused vs
custom-AR+separate-norm wins 1.38-2.11x across t=1..4096 at hidden=4096;
CUDA-graph replay with fresh data is exact (R-06 old-class corruption
history cleared for the GraphSafe fused ops).

Group correctness gate: upstream's aiter branch reduces over the FULL TP
group regardless of ``use_attn_tp_group`` -- wrong under dp-attention /
attn-tp splits. The arm declines whenever dp-attention is on; the
dispatch declines per-call unless the caller's target group IS the full
TP group; every other shape falls to the stock entry (-> (None, None) ->
plain AR + norm).

Degradation: any failure declines per-call to the saved stock functions.
``HCU_AR_NORM_IMPL=stock`` keeps upstream behavior entirely.
"""

import logging

from . import hcu_env
from .registry import MECH_REBIND, register

LOG = logging.getLogger("hcu.adapter")

_ONCE = {"served": False}


def _eligible_target_world(use_attn_tp_group):
    """World size of the group the caller wants reduced; None unless it is
    exactly the full TP group (the only group the fused kernel may serve)."""
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.runtime_context import get_parallel

    tp_world = get_tp_group().world_size
    p = get_parallel()
    want = p.attn_tp_size if use_attn_tp_group else (
        p.moe_ep_size if p.moe_ep_size > 1 else p.moe_tp_size)
    return tp_world if want == tp_world and tp_world > 1 else None


def install():
    impl = hcu_env.resolve("HCU_AR_NORM_IMPL")[0]
    if impl == "stock":
        LOG.warning("[HCU ar+norm] HCU_AR_NORM_IMPL=stock"
                    " -- upstream path stays")
        return

    try:
        import torch

        if not torch.version.hip:
            return
        from aiter.dist.device_communicators.custom_all_reduce import (  # noqa: F401
            CustomAllreduce,
        )
        from sglang.srt.layers import flashinfer_comm_fusion as _fcf
        from sglang.srt.layers.layer_boundary import exit as _lb_exit
        from sglang.srt.layers.layer_boundary.fusions import (
            allreduce as _lb_fus,
        )
        from sglang.srt.layers.layer_boundary.residual import (
            add_norm as _an,
        )
    except Exception:
        LOG.warning("[HCU ar+norm] aiter fused AR+RMSNorm tier not "
                    "installed (impl=%s)", impl, exc_info=True)
        return

    # ---- ARM: apply_flashinfer_allreduce_fusion copies ------------------
    # v0.5.22 rework: the candidate-gate function is GONE from add_norm --
    # aiter AR+norm fusion is a first-class seam there now
    # (aiter_ar_fusion_enabled_for + norm.forward_with_allreduce_fusion
    # capability probing in layer_boundary/fusions/allreduce.py). On such
    # trees arm nothing and fall through to the dispatch-only install
    # below; on <=v0.5.21 trees keep the original ARM behavior.
    arm_fn = None
    stock_apply = getattr(_an, "apply_flashinfer_allreduce_fusion", None)
    if stock_apply is not None:
        if getattr(stock_apply, "_hcu_ar_norm_arm", False):
            arm_fn = stock_apply
        else:
            max_bs = getattr(_an, "FUSE_ALLREDUCE_MAX_BATCH_SIZE", 128)

            def _apply_ar_fusion(batch_size):
                try:
                    if not (0 < batch_size <= max_bs):
                        return False
                    if _an.is_dp_attention_enabled():
                        return False
                    from sglang.srt.distributed import get_tp_group
                    from sglang.srt.runtime_context import get_parallel

                    tp = get_tp_group()
                    if tp.world_size <= 1:
                        return False
                    # Pure-TP topologies only. Under EP (moe_ep>1) or a MoE-TP
                    # split, keep deployments on the stock eager path.
                    p = get_parallel()
                    if p.moe_ep_size > 1 or p.moe_tp_size != tp.world_size:
                        return False
                    return True
                except Exception:
                    return False

            _apply_ar_fusion._hcu_ar_norm_arm = True
            _apply_ar_fusion._hcu_stock_apply = stock_apply
            arm_fn = _apply_ar_fusion
            for mod in (_an, _lb_exit, _lb_fus):
                if hasattr(mod, "apply_flashinfer_allreduce_fusion"):
                    mod.apply_flashinfer_allreduce_fusion = arm_fn
    else:
        LOG.warning(
            "[HCU ar+norm] upstream candidate gate absent (v0.5.22 "
            "plan/norm rework): ARM retired, dispatch-only install"
        )

    # ---- DISPATCH: the flashinfer entry the armed candidate calls -------
    stock = _fcf.flashinfer_allreduce_residual_rmsnorm
    if getattr(stock, "_hcu_ar_norm_aiter", False):
        register(
            "allreduce", stock,
            target="sglang.srt.layers.flashinfer_comm_fusion."
                   "flashinfer_allreduce_residual_rmsnorm",
            mechanism=MECH_REBIND,
            detail=(
                "aiter fused AR+RMSNorm already installed; "
                + ("legacy candidate arm retained" if arm_fn else "v0.5.22 dispatch-only seam")
            ),
        )
        LOG.warning("[HCU ar+norm] aiter fused AR+RMSNorm tier installed "
                    "(HCU_AR_NORM_IMPL=%s)", impl)
        return

    decline_state = {"kernel": False, "error": False}

    def _fused_ar_norm(input_tensor, residual, weight, eps=1e-6,
                       max_token_num=2048, use_oneshot=None,
                       trigger_completion_at_end=False, fp32_acc=True,
                       use_attn_tp_group=True, *args, **kwargs):
        try:
            from sglang.srt.distributed import get_tp_group

            tp = get_tp_group()
            ca = getattr(tp, "ca_comm", None)
            if ca is not None and not getattr(ca, "disabled", False) and (
                    hasattr(ca, "custom_fused_ar_rms")
                    or hasattr(ca, "fused_ar_rms")):
                if _eligible_target_world(use_attn_tp_group) is not None:
                    out = tp.fused_allreduce_rmsnorm(
                        input_tensor, residual, weight, eps)
                    if out is not None:
                        if not _ONCE["served"]:
                            _ONCE["served"] = True
                            LOG.warning(
                                "[HCU ar+norm] fused AR+RMSNorm serving"
                                " (aiter GraphSafe, R-09(1))")
                        return out
                    if not decline_state["kernel"]:
                        decline_state["kernel"] = True
                        LOG.warning("[HCU ar+norm] fused kernel declined; using stock")
        except Exception:
            if not decline_state["error"]:
                decline_state["error"] = True
                LOG.warning("[HCU ar+norm] fused attempt failed; declining"
                            " to stock", exc_info=True)
        return stock(input_tensor, residual, weight, eps=eps,
                     max_token_num=max_token_num,
                     use_oneshot=use_oneshot,
                     trigger_completion_at_end=trigger_completion_at_end,
                     fp32_acc=fp32_acc,
                     use_attn_tp_group=use_attn_tp_group, *args, **kwargs)

    _fused_ar_norm._hcu_ar_norm_aiter = True
    _fcf.flashinfer_allreduce_residual_rmsnorm = _fused_ar_norm
    register(
        "allreduce", _fused_ar_norm,
        target="sglang.srt.layers.flashinfer_comm_fusion."
               "flashinfer_allreduce_residual_rmsnorm",
        mechanism=MECH_REBIND,
        detail="aiter fused AR+RMSNorm tier (HCU_AR_NORM_IMPL=%s): arm "
               "apply_flashinfer_allreduce_fusion (def site + exit.py + "
               "fusions/allreduce.py copies) + dispatch entry at "
               "GroupCoordinator.fused_allreduce_rmsnorm -> aiter "
               "GraphSafe fused kernels; full-TP-group-only gate, "
               "per-call decline to stock (R-09(1))" % impl,
    )
    LOG.warning("[HCU ar+norm] aiter fused AR+RMSNorm tier installed "
                "(HCU_AR_NORM_IMPL=%s)", impl)
