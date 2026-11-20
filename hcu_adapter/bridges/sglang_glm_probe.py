# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash layer-boundary precision probe (2026-10-16)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
import os
import torch
from hcu_adapter.infra import hcu_env


def install(_he=None) -> None:
    # 12. GLM-5.3-Flash layer-boundary precision probe (2026-10-16):
    #     HCU_DUMP_LAYERS=1 wraps every Glm5Next decoder layer plus its
    #     self_attn/mlp submodules on rank 0 and prints per-block stats
    #     (in/out rms + max abs + nan/inf counts + last-token delta rms)
    #     for small prefill passes. glm53f is mhc=True (4-way
    #     hyper-connection residual) so the inter-layer stream is not a
    #     plain residual and norm hooks would see partial transforms —
    #     the layer boundary is the only clean seam. HCU_DUMP_LAYERS_T
    #     pins the token count (e.g. 5 for the Berlin anchor) so warmup
    #     passes stay out of the log; HCU_DUMP_LAYERS_FIRES caps prints
    #     per hook (default 2). CUDA-graph capture passes through
    #     untouched (a sync inside capture would abort the capture).
    if hcu_env.enabled("HCU_DUMP_LAYERS"):
        try:
            import sglang.srt.models.glm5_next as _g5n

            _lp = {
                "rank": None,
                "fires": {},
                "max": int(os.environ.get("HCU_DUMP_LAYERS_FIRES", "2")),
                "want_t": int(os.environ.get("HCU_DUMP_LAYERS_T", "0")),
                "save": {
                    int(x)
                    for x in os.environ.get(
                        "HCU_DUMP_LAYERS_SAVE", ""
                    ).split(",")
                    if x.strip().isdigit()
                },
                "saved": set(),
            }

            def _lp_rank():
                if _lp["rank"] is None:
                    _lp["rank"] = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized()
                        else 0
                    )
                return _lp["rank"]

            def _lp_pick(cands):
                for t in cands:
                    if (
                        isinstance(t, torch.Tensor)
                        and t.dim() >= 2
                        and t.is_floating_point()
                        and 1 < t.shape[0] <= 64
                        and (
                            not _lp["want_t"] or t.shape[0] == _lp["want_t"]
                        )
                    ):
                        return t
                return None

            def _lp_stats(x):
                f = x.detach().float()
                return (
                    float(f.pow(2).mean().sqrt()),
                    float(f.abs().max()),
                    int(torch.isnan(f).sum()) + int(torch.isinf(f).sum()),
                )

            def _lp_wrap(mod, idx, tag, is_lin):
                key = (idx, tag)
                orig = mod.forward

                def fwd(*a, **kw):
                    # The MoE runner is inplace (writes into hidden_states and
                    # returns the same tensor), so the input must be snapshotted
                    # BEFORE the call or the wrapper is blind.
                    snap = None
                    try:
                        if (
                            not torch.cuda.is_current_stream_capturing()
                            and _lp_rank() == 0
                            and _lp["fires"].get(key, 0) < _lp["max"]
                        ):
                            tin0 = _lp_pick(a)
                            if tin0 is None:
                                tin0 = _lp_pick(list(kw.values()))
                            if tin0 is not None:
                                snap = (
                                    tin0.detach().float().clone(),
                                    tin0.shape[0],
                                )
                    except Exception:
                        snap = None
                    out = orig(*a, **kw)
                    try:
                        if snap is None:
                            return out
                        if torch.cuda.is_current_stream_capturing():
                            return out
                        _lp["fires"][key] = _lp["fires"].get(key, 0) + 1
                        i_f, T0 = snap
                        outs = out if isinstance(out, (tuple, list)) else (out,)
                        tout = _lp_pick(outs)
                        if tout is None or tout.shape[0] != T0:
                            return out
                        i_rms, i_mx, i_bad = _lp_stats(i_f)
                        o_rms, o_mx, o_bad = _lp_stats(tout)
                        row_out = tout.detach().float().flatten(1)[-1]
                        d = float((row_out - i_f.flatten(1)[-1]).pow(2).mean().sqrt())
                        am = int(row_out.abs().argmax())
                        print(
                            f"[HCU L45] i={idx:02d} "
                            f"{'K' if is_lin else 'F'} {tag} T={T0}"
                            f" in(rms={i_rms:.4g},max={i_mx:.4g})"
                            f" out(rms={o_rms:.4g},max={o_mx:.4g})"
                            f" d={d:.4g} bad={i_bad}+{o_bad} am={am}",
                            flush=True,
                        )
                        once = os.environ.get("HCU_DUMP_LAYERS_ONCE", "1") == "1"
                        if idx in _lp["save"] and (
                            not once or (idx, tag) not in _lp["saved"]
                        ):
                            _lp["saved"].add((idx, tag))
                            torch.save(
                                {
                                    "in": i_f.cpu(),
                                    "out": tout.detach().float().cpu(),
                                },
                                f"/tmp/hcu_sv_{tag}_{idx:02d}.pt",
                            )
                            print(
                                f"[HCU L45] saved /tmp/hcu_sv_{tag}_{idx:02d}.pt",
                                flush=True,
                            )
                    except Exception as _e:
                        if _lp["fires"].get(key, 0) < 1:
                            print(
                                f"[HCU L45] i={idx} {tag} err {_e}", flush=True
                            )
                            _lp["fires"][key] = _lp["fires"].get(key, 0) + 1
                    return out

                mod.forward = fwd
                return orig

            def _lp_init(_orig_init):
                def init(self, *a, **k):
                    _orig_init(self, *a, **k)
                    m = getattr(self, "model", self)
                    layers = getattr(m, "layers", None)
                    if layers is None:
                        for attr in ("language_model", "text_model"):
                            mm = getattr(self, attr, None)
                            if mm is not None:
                                layers = getattr(mm, "layers", None)
                                if layers is not None:
                                    break
                    if layers is None:
                        print("[HCU L45] no layers found", flush=True)
                        return
                    for i, lyr in enumerate(layers):
                        is_lin = bool(getattr(lyr, "is_linear_attn", False))
                        _lp_wrap(lyr, i, "LYR", is_lin)
                        if getattr(lyr, "self_attn", None) is not None:
                            _lp_wrap(lyr.self_attn, i, "ATT", is_lin)
                        if getattr(lyr, "mlp", None) is not None:
                            _lp_wrap(lyr.mlp, i, "MLP", is_lin)
                    # final norm (post hc_contract collapse): its input is the
                    # (T, hidden) final hidden feeding lm_head — the cleanest
                    # tail anchor for lm_head/logit debugging.
                    _mnorm = getattr(self.model, "norm", None)
                    if _mnorm is not None:
                        _lp_wrap(_mnorm, 99, "NRM", False)
                    print(
                        f"[HCU L45] probe on {len(layers)} layers"
                        f" (want_t={_lp['want_t'] or 'any'})",
                        flush=True,
                    )

                return init

            _g5n.Glm5NextForConditionalGeneration.__init__ = _lp_init(
                _g5n.Glm5NextForConditionalGeneration.__init__
            )
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: glm5 layer probe not installed", exc_info=True
            )
