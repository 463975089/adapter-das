# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""deep_ep API bridge (EP validation 2026-10-11).

Extracted verbatim from bridges/sglang.py install() section 8 (layout
governance 2026-11-18; zero logic change). sglang v0.5.21 calls upstream
deep_ep `low_latency_dispatch(x, topk_idx, nmdtpr, num_experts,
use_fp8=..., ...)` and passes routing weights only at combine, while the
DTK deep_ep fork (1.1.0+dtk) wants `topk_weight` as the 3rd positional
and configures quantization via quant_type (0=bf16, 2=E4M3FNUZ,
3=UE8M0) + quant_group_size/round_scale. Translate the call shape so
DeepEP MoE runs zero-modification (HCU_DEEPEP_BRIDGE, env-vars.md).
"""
from __future__ import annotations

import logging
import torch


def install(_he=None) -> None:
    # 8. deep_ep API bridge (EP validation 2026-10-11): sglang v0.5.21 calls
    # upstream deep_ep `low_latency_dispatch(x, topk_idx, nmdtpr, num_experts,
    # use_fp8=..., ...)` and passes routing weights only at combine, while the
    # DTK deep_ep fork (1.1.0+dtk) wants `topk_weight` as the 3rd positional
    # and configures quantization via quant_type (0=bf16, 2=E4M3FNUZ,
    # 3=UE8M0) + quant_group_size/round_scale. Translate the call shape so
    # DeepEP MoE runs zero-modification (HCU_DEEPEP_BRIDGE, env-vars.md).
    try:
        if _he is None or _he.enabled("HCU_DEEPEP_BRIDGE"):
            import inspect as _insp

            import deep_ep as _dep

            _ll_orig = _dep.Buffer.low_latency_dispatch
            _ll_params = set(_insp.signature(_ll_orig).parameters)
            if "use_fp8" not in _ll_params and "quant_type" in _ll_params:
                import sys as _sys

                def _ll_dispatch_upstream(
                    self,
                    x,
                    topk_idx,
                    num_max_dispatch_tokens_per_rank=None,
                    num_experts=None,
                    *,
                    topk_weights=None,
                    use_fp8=False,
                    use_nvfp4=False,
                    use_mxfp4=False,
                    use_mxfp8=False,
                    x_global_scale=None,
                    x_scales=None,
                    round_scale=False,
                    use_ue8m0=False,
                    quant_group_size=128,
                    async_finish=False,
                    return_recv_hook=False,
                    **unmapped,
                ):
                    # sglang keeps the routing weights in the caller frame
                    # (_dispatch_core); the DTK fork needs them at dispatch.
                    if topk_weights is None:
                        _cw = _sys._getframe(1).f_locals.get("topk_weights")
                        if not isinstance(_cw, torch.Tensor):
                            raise TypeError(
                                "HCU deep_ep bridge: topk_weights "
                                "unavailable (absent from caller frame); "
                                "cannot translate to DTK fork call shape"
                            )
                        topk_weights = _cw
                    if unmapped:
                        logging.getLogger("hcu.deepep_bridge").warning(
                            "dropped unmapped kwargs: %s", sorted(unmapped)
                        )
                    if use_nvfp4 or use_mxfp4 or use_mxfp8:
                        raise NotImplementedError(
                            "HCU deep_ep bridge: nvfp4/mxfp4/mxfp8 "
                            "dispatch has no DTK-fork equivalent"
                        )
                    if use_fp8:
                        return _ll_orig(
                            self,
                            x,
                            topk_idx,
                            topk_weights,
                            num_max_dispatch_tokens_per_rank,
                            num_experts,
                            quant_type=3 if use_ue8m0 else 2,
                            quant_group_size=quant_group_size,
                            fp8_round_scale=bool(round_scale),
                            async_finish=async_finish,
                            return_recv_hook=return_recv_hook,
                            x_scales=(
                                x_scales
                                if x_scales is not None
                                else x_global_scale
                            ),
                        )
                    if x_global_scale is not None and x_scales is None:
                        logging.getLogger("hcu.deepep_bridge").warning(
                            "x_global_scale dropped on bf16 dispatch path"
                        )
                    return _ll_orig(
                        self,
                        x,
                        topk_idx,
                        topk_weights,
                        num_max_dispatch_tokens_per_rank,
                        num_experts,
                        quant_type=0,
                        async_finish=async_finish,
                        return_recv_hook=return_recv_hook,
                        **({"x_scales": x_scales} if x_scales is not None else {}),
                    )

                def _ll_dispatch_v0521(self, x, topk_idx, *args, **kwargs):
                    if "quant_type" in kwargs or "topk_weight" in kwargs or (
                        args and isinstance(args[0], torch.Tensor)
                    ):
                        return _ll_orig(self, x, topk_idx, *args, **kwargs)
                    if kwargs.get("topk_weights") is None:
                        kwargs["topk_weights"] = _sys._getframe(1).f_locals.get(
                            "topk_weights"
                        )
                    return _ll_dispatch_upstream(self, x, topk_idx, *args, **kwargs)

                _ll_dispatch_v0521._hcu_accepts_quant_type = True
                # HY3 INT8 dispatch (#149): the compat wrapper also keeps the
                # vendor quant_type entry so Plugin INT8 dispatch retains its
                # dtype and per-token scales.
                _ll_dispatch_v0521._hcu_quant_type_dispatch = _ll_orig
                _dep.Buffer.low_latency_dispatch = _ll_dispatch_v0521
                logging.getLogger("hcu.adapter").info(
                    "HCU deep_ep bridge: v0.5.21 low_latency_dispatch call "
                    "shape -> DTK fork (quant_type) translated"
                )
    except ImportError:
        pass
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: deep_ep bridge not installed", exc_info=True
        )
