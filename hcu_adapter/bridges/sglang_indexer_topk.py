# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer top-k sgl-kernel->torch bridge

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
import torch
from hcu_adapter.infra import hcu_env


def install(_he=None) -> None:
    # DSA indexer top-k: the sgl-kernel backend ops have no HIP build; bridge
    # to torch equivalents (see hcu_adapter.features.dsa.dsa_topk).
    try:
        import sgl_kernel as _sk
        import hcu_adapter.features.dsa.dsa_topk as _dsa_topk

        _dsa_topk.install()
        if hcu_env.enabled("HCU_DUMP_TOPK"):
            _tpk = {"n": 0}

            def _hcu_log_topk(fn_name, fn):
                def wrapped(*a, **k):
                    out = fn(*a, **k)
                    if _tpk["n"] < 3:
                        score = a[0]
                        print(
                            f"[HCU topk] {fn_name} score{tuple(score.shape)}"
                            f" dtype={score.dtype} lens={a[1].tolist()[:4]}"
                            f" topk={a[3] if len(a) > 3 else k.get('topk')}",
                            flush=True,
                        )
                        torch.save(
                            {
                                "score": score[:2].float().cpu(),
                                "lens": a[1][:4].cpu(),
                                "out": out[:2].cpu(),
                            },
                            f"/tmp/hcu_topk_{_tpk['n']}.pt",
                        )
                        _tpk["n"] += 1
                    return out

                return wrapped

            for _nm in ("fast_topk_transform_fused",
                        "fast_topk_transform_ragged_fused",
                        "fast_topk_v2"):
                _fn = getattr(_sk, _nm, None)
                if _fn is not None:
                    setattr(_sk, _nm, _hcu_log_topk(_nm, _fn))
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: dsa topk bridge not installed", exc_info=True
        )
