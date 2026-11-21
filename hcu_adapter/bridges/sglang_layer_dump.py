# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HCU_DUMP_LAYERS per-layer debug capture (2026-10-16)

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
    # Debug facility: HCU_DUMP_LAYERS=1 captures, per decoder layer, the
    # post-reduction residual stream entering `input_layernorm` (TP8 uses
    # delayed partial reductions, so module hooks deeper than this see
    # one rank's partial sum). Entry i of the saved list = residual stream
    # before layer i (embed output = entry 0), matching the pure torch
    # reference (tests/tools/_ref_forward_h.py). With HCU_DUMP_INNER=1 also
    # saves xa/xb (norm outputs) of layers 0-1 to /tmp/hcu_inner.pt.
    # Prefill-sized passes only (1 < T <= 64), rank 0, file
    # HCU_DUMP_LAYERS_PATH (default /tmp/hcu_layers.pt).
    if hcu_env.enabled("HCU_DUMP_LAYERS"):
        # NOTE: importing sglang model modules HERE fails — their deep chains
        # (glm4_moe -> deepseek_v2 -> dsa_backend -> `import deep_gemm`)
        # depend on bridges installed further down this file. Instead the
        # registration below wraps importlib.import_module and patches the
        # model class when sglang's model loader imports it naturally (the
        # one moment where the whole stack is guaranteed importable).
        try:
            import sglang.srt.models.glm4_moe as _g4m  # may fail; see note
        except Exception:
            _g4m = None

            _dump = {
                "cur": {},
                "rank": None,
                "path": os.environ.get(
                    "HCU_DUMP_LAYERS_PATH", "/tmp/hcu_layers.pt"
                ),
                "nlayers": int(os.environ.get("HCU_DUMP_LAYERS_TOTAL", "0")),
                "want_t": int(os.environ.get("HCU_DUMP_LAYERS_T", "0")),
                "inner": {},
            }

            def _hcu_row(t):
                if not isinstance(t, torch.Tensor) or t.dim() != 2:
                    return None
                if not 1 < t.shape[0] <= 64:
                    return None
                return t[t.shape[0] - 1].detach().float().cpu()

            def _hcu_layernorm_pre(layer_idx):
                def hook(mod, args, kwargs):
                    try:
                        x = kwargs.get("x", args[0] if args else None)
                        r = kwargs.get(
                            "residual", args[1] if len(args) > 1 else None)
                        if _dump["rank"] is None:
                            _dump["rank"] = (
                                torch.distributed.get_rank()
                                if torch.distributed.is_initialized()
                                else 0
                            )
                        if _dump["rank"] != 0:
                            return
                        if r is None:
                            hs = x
                        elif isinstance(r, torch.Tensor):
                            hs = x.float() + r.float()
                        else:
                            return
                        row = _hcu_row(hs)
                        if row is not None and (
                            not _dump["want_t"]
                            or hs.shape[0] == _dump["want_t"]
                        ):
                            cur = _dump["cur"]
                            if layer_idx == 0 and cur:
                                _dump["cur"] = cur = {}
                            cur[layer_idx] = (hs.shape[0], row)
                            nl = _dump["nlayers"]
                            if nl and len(cur) == nl:
                                rows = [
                                    (i, cur[i][0], cur[i][1])
                                    for i in sorted(cur)
                                ]
                                torch.save(rows, _dump["path"])
                                print(
                                    f"[HCU dump] saved {len(rows)} rows"
                                    f" (T={rows[0][1]}) to {_dump['path']}",
                                    flush=True,
                                )
                                _dump["cur"] = {}
                    except Exception:
                        logging.getLogger("hcu.adapter").warning(
                            "HCU adapter: layer dump capture failed",
                            exc_info=True,
                        )

                return hook

            def _hcu_norm_out(tag):
                def hook(mod, args, output):
                    if not hcu_env.enabled("HCU_DUMP_INNER"):
                        return
                    if _dump["rank"] not in (None, 0):
                        return
                    out = output[0] if isinstance(output, tuple) else output
                    row = _hcu_row(out)
                    if row is not None:
                        _dump["inner"][tag] = row
                        if len(_dump["inner"]) >= 4:
                            torch.save(dict(_dump["inner"]), "/tmp/hcu_inner.pt")
                            print(
                                f"[HCU inner] saved keys="
                                f"{sorted(_dump['inner'])}",
                                flush=True,
                            )
                            _dump["inner"] = {}

                return hook

            def _hcu_dump_init(_orig_init):
                def init(self, *a, **k):
                    _orig_init(self, *a, **k)
                    layers = getattr(
                        getattr(self, "model", None), "layers", None
                    )
                    if layers is None:
                        return
                    for i, lyr in enumerate(layers):
                        lyr.input_layernorm.register_forward_pre_hook(
                            _hcu_layernorm_pre(i), with_kwargs=True
                        )
                        if i < 2 and hcu_env.enabled("HCU_DUMP_INNER"):
                            lyr.input_layernorm.register_forward_hook(
                                _hcu_norm_out(f"xa{i}")
                            )
                            lyr.post_attention_layernorm.register_forward_hook(
                                _hcu_norm_out(f"xb{i}")
                            )
                    print(
                        f"[HCU dump] pre-hooks on {len(layers)}"
                        " input_layernorms",
                        flush=True,
                    )

                return init

            _HCU_DUMP_TARGETS = {
                "sglang.srt.models.glm4_moe": ("GlmMoeDsaForCausalLM",),
                # GLM-5.3-Flash: text-only -> self.model = Glm5NextModel
                # with .layers; hook machinery is class-agnostic (same norm
                # attr names).
                "sglang.srt.models.glm5_next": (
                    "Glm5NextForConditionalGeneration",
                    "Glm5NextForCausalLM",
                ),
            }

            def _hcu_patch_model_module(mod):
                targets = _HCU_DUMP_TARGETS.get(getattr(mod, "__name__", ""))
                if not targets or getattr(mod, "_hcu_dump_patched", False):
                    return
                mod._hcu_dump_patched = True
                for cls_name in targets:
                    cls = getattr(mod, cls_name, None)
                    if cls is not None:
                        cls.__init__ = _hcu_dump_init(cls.__init__)

            import importlib as _imp

            _orig_import_module = _imp.import_module

            def _hcu_import_module(name, *a, **k):
                mod = _orig_import_module(name, *a, **k)
                try:
                    _hcu_patch_model_module(mod)
                except Exception:
                    logging.getLogger("hcu.adapter").warning(
                        "HCU adapter: layer dump model patch failed",
                        exc_info=True,
                    )
                return mod

            _imp.import_module = _hcu_import_module
            if _g4m is not None:
                _hcu_patch_model_module(_g4m)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: layer dump bridge not installed", exc_info=True
            )
