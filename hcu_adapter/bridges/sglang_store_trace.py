# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HCU store-trace + DSA topk torch-arm probe (debug capture)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
import torch
from hcu_adapter.infra import hcu_env


def install(_he=None) -> None:
    # HCU store-trace (2026-10-15): which branch wrote the index-K cache.
    try:
        from sglang.srt.layers.attention.dsa import dsa_indexer as _di_mod

        if hcu_env.enabled("HCU_DUMP_TOPK") and not getattr(
            _di_mod, "_hcu_store_trace", False
        ):
            dsa_indexer = _di_mod
            dsa_indexer._hcu_store_trace = True
            _cls = dsa_indexer.Indexer
            _orig_store = _cls._store_index_k_cache

            def _traced_store(self, *a, **kw):
                pool = dsa_indexer.get_token_to_kv_pool()
                rid = self.layer_id if hasattr(self, "layer_id") else -1
                owned = (
                    pool._is_layer_owned(rid)
                    if hasattr(pool, "_is_layer_owned")
                    else "n/a"
                )
                buf = pool.get_index_k_with_scale_buffer(layer_id=rid)
                s0 = float(buf.float().abs().sum()) if buf.numel() < 3_000_000 else float(buf[: 1_000_000].float().abs().sum())
                out = _orig_store(self, *a, **kw)
                s1 = float(buf.float().abs().sum()) if buf.numel() < 3_000_000 else float(buf[: 1_000_000].float().abs().sum())
                print(
                    f"[HCU store] layer={rid} owned={owned} buf={tuple(buf.shape)} "
                    f"abs before={s0:.3f} after={s1:.3f}",
                    flush=True,
                )
                return out

            _cls._store_index_k_cache = _traced_store
            print("[HCU store] trace installed", flush=True)
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: store trace not installed", exc_info=True
        )

    # DSA topk torch-arm probe (2026-10-15): capture prefill RAGGED topk in/out.
    try:
        if hcu_env.enabled("HCU_DUMP_TOPK"):
            from sglang.srt.layers.attention.dsa import dsa_topk_backend as _dtb
            _tpk = {"n": 0}
            _orig_unf = _dtb._topk_unfused

            def _probe_unf(score, lengths, topk, row_starts=None, topk_op=None, topk_op_kwargs=None):
                out = _orig_unf(score, lengths, topk, row_starts=row_starts,
                                topk_op=topk_op, topk_op_kwargs=topk_op_kwargs)
                if _tpk["n"] < 2:
                    sf = score.float()
                    valid = torch.isfinite(sf)
                    try:
                        print("[HCU topkidx] call", _tpk["n"],
                              "score", tuple(score.shape), "finite%", round(valid.float().mean().item(), 4),
                              "scoremax", round(sf[valid].max().item(), 3) if valid.any() else 0,
                              "out[-1]pct", round((out == -1).float().mean().item(), 4),
                              "out_uniq", out.unique().numel(), flush=True)
                        torch.save({"score": score[:2].float().cpu(), "lens": lengths[:4].cpu(),
                                    "out": out[:2].cpu()}, f"/tmp/hcu_topkidx_{_tpk['n']}.pt")
                        _tpk["n"] += 1
                    except Exception as e:
                        print("[HCU topkidx] probe err", e, flush=True)
                return out

            _dtb._topk_unfused = _probe_unf
            print("[HCU topkidx] probe installed", flush=True)
    except Exception:
        logging.getLogger("hcu.adapter").warning("HCU adapter: topkidx probe not installed", exc_info=True)
