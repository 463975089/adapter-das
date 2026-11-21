# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Prefill context-parallel bridge (2026-10-12)

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
    # 11. Prefill context-parallel bridge (2026-10-12): stock v0.5.21
    #     validate_prefill_cp_platform rejects HIP outright; HCU_PREFILL_CP=1
    #     (opt-in) opens the gate so the upstream CP machinery (zigzag/
    #     interleave, fa3 convention) can run on HCU.
    try:
        from hcu_adapter import prefill_cp as _pcp

        _pcp.install()
    except ImportError:
        pass
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: prefill-CP bridge not installed", exc_info=True
        )

    # 11c. GLM-5.3-Flash fused mhc_post_pre bypass (2026-10-16): the cross-
    #      layer fusion gate (deepseek_v4_fused_mhc.is_cross_layer_mhc_
    #      fusion_enabled) checks only SGLANG_OPT_FUSE_MHC_POST_PRE /
    #      USE_TILELANG_MHC_{PRE,POST} — all default True — and never
    #      is_hip, so on gfx938 every decoder layer's attn->ffn boundary
    #      runs the aiter triton fused mhc_post_pre kernel. Runtime dump
    #      /tmp/hcu_mhcpost_003.pt (HCU_DUMP_MHC=1) proves it computes
    #      wrong FFN-stage h_post/h_res: identical inputs give
    #      2*sigmoid(z)=[0.881, 0.616, 0.001, 1.020] from the fused kernel
    #      vs [0.856, 0.006, 0.001, 0.014] from the pure-torch reference
    #      (kernels/ops/layernorm/mhc.py _mhc_pre_torch) — z shifted +0.5..+5
    #      per stream. The corrupted post/comb propagate into every layer's
    #      residual streams (word soup; all modules — KDA/DSA/MoE/conv/
    #      router — verify clean against checkpoint references). HCU_MHC_
    #      UNFUSE=1 (default on HIP) forces the unfused chain: hc_post +
    #      hc_ffn_pre, both _mhc_*_torch on HIP (aiter needs gfx950,
    #      tilelang is hip-gated off).
    if hcu_env.resolve("HCU_MHC_UNFUSE")[0] == "1" and torch.version.hip:
        os.environ["SGLANG_OPT_FUSE_MHC_POST_PRE"] = "false"
        # stdout stays protocol-clean for pytest-xdist/execnet
        # workers; loud notices belong on stderr.
        import sys as _sys

        print(
            "[HCU mhcunfuse] SGLANG_OPT_FUSE_MHC_POST_PRE forced false"
            " (aiter triton mhc_post_pre wrong on gfx938)",
            file=_sys.stderr, flush=True,
        )

    # 11b. GLM-5.3-Flash mHC boundary dump (2026-10-16): HCU_DUMP_MHC=1 wraps
    #      kernels/ops/layernorm/mhc.{hc_pre,hc_post} (the torch path HCU
    #      actually takes — aiter needs gfx950, tilelang is hip-gated off)
    #      and saves every T==HCU_DUMP_LAYERS_T call's inputs/outputs on
    #      rank 0. Must run BEFORE the layer-probe block imports
    #      sglang.srt.models.glm5_next, whose module-level
    #      `from ...mhc import hc_pre` would otherwise bind the original.
    if os.environ.get("HCU_DUMP_MHC") == "1":
        try:
            # 替代档: mhc dump 探针挂载面（HCU_DUMP_MHC；包面无该出口）
            import sglang.kernels.ops.layernorm.mhc as _mhcmod

            _mhcd = {
                "want_t": int(os.environ.get("HCU_DUMP_LAYERS_T", "0") or 0),
                "max": int(os.environ.get("HCU_DUMP_MHC_MAX", "200")),
                "pre": 0,
                "post": 0,
                "rank": None,
            }

            def _mhcd_rank():
                if _mhcd["rank"] is None:
                    _mhcd["rank"] = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized()
                        else 0
                    )
                return _mhcd["rank"]

            def _mhcd_ok(t):
                return (
                    isinstance(t, torch.Tensor)
                    and t.dim() == 2
                    and t.is_floating_point()
                    and t.shape[0] > 1
                    and (
                        not _mhcd["want_t"] or t.shape[0] == _mhcd["want_t"]
                    )
                    and not torch.cuda.is_current_stream_capturing()
                    and _mhcd_rank() == 0
                )

            _mhcd_orig_pre = _mhcmod.hc_pre
            _mhcd_orig_post = _mhcmod.hc_post

            def _hcu_hc_pre_dump(x, *a, **kw):
                res = _mhcd_orig_pre(x, *a, **kw)
                try:
                    if _mhcd_ok(x) and _mhcd["pre"] < _mhcd["max"]:
                        torch.save(
                            {
                                "x": x.detach().float().cpu(),
                                "layer_input": res[0].detach().float().cpu(),
                                "comb": res[1].detach().float().cpu(),
                                "post": res[2].detach().float().cpu(),
                            },
                            f"/tmp/hcu_mhcpre_{_mhcd['pre']:03d}.pt",
                        )
                        _mhcd["pre"] += 1
                except Exception:
                    pass
                return res

            def _hcu_hc_post_dump(x, residual, h_post, h_res, *a, **kw):
                out = _mhcd_orig_post(x, residual, h_post, h_res, *a, **kw)
                try:
                    if _mhcd_ok(x) and _mhcd["post"] < _mhcd["max"]:
                        torch.save(
                            {
                                "x": x.detach().float().cpu(),
                                "residual": residual.detach().float().cpu(),
                                "h_post": h_post.detach().float().cpu(),
                                "h_res": h_res.detach().float().cpu(),
                                "out": out.detach().float().cpu(),
                            },
                            f"/tmp/hcu_mhcpost_{_mhcd['post']:03d}.pt",
                        )
                        _mhcd["post"] += 1
                except Exception:
                    pass
                return out

            _mhcmod.hc_pre = _hcu_hc_pre_dump
            _mhcmod.hc_post = _hcu_hc_post_dump
            print(
                "[HCU mhcdump] hc_pre/hc_post dump wrappers installed",
                flush=True,
            )
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: mhc dump wrappers not installed", exc_info=True
            )
