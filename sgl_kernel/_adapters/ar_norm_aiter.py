"""R-09 (1): aiter fused all-reduce + RMSNorm seam for sglang layernorm.

Upstream v0.5.21 already carries the whole consumer chain dormant:

  layernorm._forward_with_allreduce_fusion
    -- ``if _use_aiter:`` branch (SGLANG_USE_AITER; closed on this stack)
    else-branch -> flashinfer_allreduce_residual_rmsnorm (no flashinfer
       here -> (None, None) -> plain AR + norm)
  GroupCoordinator.fused_allreduce_rmsnorm (parallel_state.py:814)
    -> ca_comm.custom_fused_ar_rms / fused_ar_rms   [aiter GraphSafe class]

The aiter communicator class supplies ``custom_fused_ar_rms`` /
``fused_ar_rms``, so re-pointing the *flashinfer* entry the else-branch
already calls is the one-symbol surgical seam -- no SGLANG_USE_AITER
(which would flip every aiter gate upstream-wide).

8-rank probe evidence (nmz26 BW1102, tmp_fetch/probe_r09_arnorm.py,
2026-10-19): parity vs fp64-exact all-reduce is 1 ulp (norm) / 0 ulp
(residual) -- *better* than the rccl reference (3-8 ulps; the IPC kernel
sums exactly where rccl accumulates in ring order); fused vs
custom-AR+separate-norm wins 1.38-2.11x across t=1..4096 at hidden=4096;
CUDA-graph replay with fresh data is exact (R-06 old-class corruption
history cleared for the GraphSafe fused ops).

Group correctness gate: upstream's aiter branch reduces over the FULL TP
group regardless of ``use_attn_tp_group`` -- wrong under dp-attention /
attn-tp splits. This bridge only fuses when the caller's target group IS
the full TP group; every other shape declines to the stock flashinfer
entry (which returns (None, None) here, landing on plain AR + norm).

Degradation: any failure declines per-call to the saved stock function.
``HCU_AR_NORM_IMPL=stock`` keeps upstream behavior entirely.
"""

import logging

from . import hcu_env
from .registry import MECH_REBIND, register

LOG = logging.getLogger("hcu.adapter")

_FUSED = {"done": False}


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
    except Exception:
        LOG.warning("[HCU ar+norm] aiter fused AR+RMSNorm tier not "
                    "installed (impl=%s)", impl, exc_info=True)
        return

    stock = _fcf.flashinfer_allreduce_residual_rmsnorm
    if getattr(stock, "_hcu_ar_norm_aiter", False):
        return  # already installed (sitecustomize + explicit install)

    declined = []

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
                        if not _FUSED["done"]:
                            _FUSED["done"] = True
                            LOG.warning("[HCU ar+norm] fused AR+RMSNorm serving"
                                     " (aiter GraphSafe, R-09(1))")
                        return out
                    declined.append("kernel-declined")
        except Exception:
            if not declined:
                declined.append("error")
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
        detail="aiter fused AR+RMSNorm tier (HCU_AR_NORM_IMPL=%s): "
               "flashinfer seam re-pointed at the dormant upstream "
               "GroupCoordinator.fused_allreduce_rmsnorm chain; full-TP-"
               "group-only gate, per-call decline to stock (R-09(1))"
               % impl,
    )
    LOG.warning("[HCU ar+norm] aiter fused AR+RMSNorm tier installed "
                 "(HCU_AR_NORM_IMPL=%s)", impl)
