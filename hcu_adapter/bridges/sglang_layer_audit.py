# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HCU_LAYER_AUDIT per-module output magnitude audit (triage tooling)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
import os
import torch


def install(_he=None) -> None:
    # Layer-output audit (triage tooling): HCU_LAYER_AUDIT=1 wraps
    # nn.Module.__call__ and appends per-module output magnitudes for
    # decoder-layer / linear-attn / DSA / MoE classes to
    # HCU_LAYER_AUDIT_PATH.<pid>. Zero overhead when the env is unset.
    # Used to find the first layer whose activations blow up.
    if os.environ.get("HCU_LAYER_AUDIT") == "1":
        try:
            import torch.nn as _tnn

            _audit_cls = tuple(
                s.strip()
                for s in os.environ.get(
                    "HCU_LAYER_AUDIT_CLASSES",
                    "DecoderLayer,LinearAttention,GatedDeltaNet,"
                    "SparseAttention,MoeBlock,MLP,Glm5NextModel",
                ).split(",")
                if s.strip()
            )
            _audit_fh = open(
                os.environ.get("HCU_LAYER_AUDIT_PATH", "/root/logs/layer_audit.log")
                + "."
                + str(os.getpid()),
                "a",
                buffering=1,
            )
            _audit_step = [0]
            _orig_module_call = _tnn.Module.__call__

            def _hcu_audit_call(self, *args, **kwargs):
                _is_stack = type(self).__name__ == "Glm5NextModel"
                if _is_stack:
                    _audit_step[0] += 1
                out = _orig_module_call(self, *args, **kwargs)
                name = type(self).__name__
                if _is_stack or name.endswith(_audit_cls):
                    # .item() D2H-syncs, which stalls/hangs stream capture;
                    # capture inputs are zeros anyway, so skip mid-capture.
                    try:
                        if torch.cuda.is_current_stream_capturing():
                            return out
                    except Exception:
                        pass
                    lid = getattr(self, "layer_id", None)
                    t = out[0] if isinstance(out, tuple) else out
                    if torch.is_tensor(t) and t.numel():
                        f = t.detach().float()
                        _audit_fh.write(
                            "%s lid=%s fw=%d meanabs=%.4e maxabs=%.4e nan=%d\n"
                            % (
                                name,
                                lid,
                                _audit_step[0],
                                f.abs().mean().item(),
                                f.abs().max().item(),
                                int(torch.isnan(f).sum().item()),
                            )
                        )
                return out

            _tnn.Module.__call__ = _hcu_audit_call
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: layer audit installed -> %s.* (classes %s)",
                os.environ.get("HCU_LAYER_AUDIT_PATH", "/root/logs/layer_audit.log"),
                ",".join(_audit_cls),
            )
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: layer audit not installed", exc_info=True
            )
