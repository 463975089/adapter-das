# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HCU_DUMP_INT8 serve-time W8A8Int8 apply audit (torch recompute + cos/maxerr log)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
import os
import torch


def install(_he=None) -> None:
    # Serve-time INT8 apply audit: HCU_DUMP_INT8=1 wraps
    # CompressedTensorsW8A8Int8.apply_weights and, for the first _N calls,
    # recomputes each output with plain torch (per-token-dequant x @
    # channel-dequant weight) and logs cos/maxerr to /tmp/int8_apply_dump.log.
    # Standalone GEMM probes all pass but the served model garbles — this
    # catches the real in-process call chain (fused loaders, scheme state).
    if os.environ.get("HCU_DUMP_INT8") == "1":
        try:
            import functools

            import sglang.srt.layers.quantization.compressed_tensors.schemes.\
                compressed_tensors_w8a8_int8 as _cti8d

            _Scheme = _cti8d.CompressedTensorsW8A8Int8
            _i8_orig_apply = _Scheme.apply_weights
            _i8_log = open("/tmp/int8_apply_dump.log", "w", buffering=1)
            _i8_budget = 96

            def _i8_audited(self, layer, x, bias):
                out = _i8_orig_apply(self, layer, x, bias)
                global _i8_budget
                if _i8_budget > 0:
                    _i8_budget -= 1
                    try:
                        import torch.nn.functional as F

                        # 替代档: 无（审计参考；包面未导出 per_token_quant_int8）
                        from sglang.kernels.ops.quantization.int8_kernel import (
                            per_token_quant_int8,
                        )

                        x_q, x_s = per_token_quant_int8(x)
                        deq_a = (x_q.float() * x_s.float()).to(x.dtype)
                        deq_b = (
                            layer.weight.t().float() * layer.weight_scale.float()
                        ).to(x.dtype)
                        ref = deq_a @ deq_b.t()
                        if bias is not None:
                            ref = ref + bias.float()
                        cos = F.cosine_similarity(
                            out.float().flatten(), ref.float().flatten(), dim=0
                        ).item()
                        maxerr = (out.float() - ref.float()).abs().max().item()
                        _i8_log.write(
                            f"M={x.shape[0]} x{x.shape}"
                            f" xamax={x.float().abs().max().item():.6f}"
                            f" wamax={layer.weight.float().abs().max().item():.4f}"
                            f" wsamax={layer.weight_scale.float().abs().max().item():.6f}"
                            f" cos={cos:.6f} maxerr={maxerr:.5f}"
                            f" out3={out.flatten()[:3].float().tolist()}"
                            f" x3={x.flatten()[:3].float().tolist()}\n"
                        )
                    except Exception as e:
                        _i8_log.write(f"AUDIT-FAIL {e!r}\n")
                return out

            _Scheme.apply_weights = _i8_audited

            # When did the weights die: dump amax before/after
            # process_weights_after_loading (per Linear module).
            try:
                _i8_pw_orig = _Scheme.process_weights_after_loading

                def _pw_audited(self, layer):
                    try:
                        w = layer.weight
                        _i8_log.write(
                            f"PW-BEFORE strat={self.strategy}"
                            f" static={self.is_static_input_scheme}"
                            f" w{list(w.shape)} {w.dtype} type={type(w).__name__}"
                            f" ptr={w.data_ptr()}"
                            f" wamax={w.float().abs().max().item():.4f}\n"
                        )
                    except Exception as e:
                        _i8_log.write(f"PW-BEFORE-FAIL {e!r}\n")
                    _i8_pw_orig(self, layer)
                    try:
                        w = layer.weight
                        ws = layer.weight_scale
                        _i8_log.write(
                            f"PW-AFTER w{list(w.shape)} {w.dtype}"
                            f" type={type(w).__name__} ptr={w.data_ptr()}"
                            f" wamax={w.float().abs().max().item():.4f}"
                            f" ws{list(ws.shape) if ws is not None else None}"
                            f" wsamax={0 if ws is None else ws.float().abs().max().item():.6f}\n"
                        )
                    except Exception as e:
                        _i8_log.write(f"PW-AFTER-FAIL {e!r}\n")

                _Scheme.process_weights_after_loading = _pw_audited
            except Exception as e:
                _i8_log.write(f"PW-WRAP-SKIP {e!r}\n")

            # Post-load census: after the model's load_weights returns, count
            # int8 params and sample amaxes (GDN + full-attention layers).
            try:
                import sglang.srt.models.qwen3_5 as _q35

                _lw_orig = _q35.Qwen3_5ForConditionalGeneration.load_weights

                def _lw_audited(self, weights):
                    # Peek the incoming stream: record the shape/dtype/amax of
                    # every layer-1/layer-3 tensor we see, and count names, so
                    # we can tell "weights never arrived" from "loader wrote
                    # nowhere". Then run the real load with a generator.
                    import itertools

                    seen = {}

                    def _tracing():
                        for name, tensor in weights:
                            if (
                                "layers.1.linear_attn.in_proj" in name
                                or "layers.3." in name
                            ):
                                am = 0.0
                                if tensor.numel():
                                    am = tensor.float().abs().max().item()
                                seen[name] = (
                                    list(tensor.shape),
                                    str(tensor.dtype),
                                    round(am, 3),
                                )
                            yield name, tensor

                    out = _lw_orig(self, _tracing())
                    _i8_log.write(f"LW-STREAM unique_seen={len(seen)}\n")
                    for n in sorted(seen)[:14]:
                        _i8_log.write(f"LW-STREAM {n} {seen[n]}\n")
                    try:
                        _i8_log.write(
                            f"LW-LOADED set_size={len(out)}"
                            f" l1qkvz={'model.layers.1.linear_attn.in_proj_qkvz.weight' in out}"
                            f" l3qkv={'model.layers.3.qkv_proj.weight' in out}\n"
                        )
                    except Exception as e:
                        _i8_log.write(f"LW-LOADED-FAIL {e!r}\n")
                    try:
                        i8_total = i8_zero = 0
                        samples = []
                        for n, p in self.named_parameters():
                            if p.dtype == torch.int8 and p.dim() >= 1:
                                i8_total += 1
                                am = p.float().abs().max().item()
                                if am == 0:
                                    i8_zero += 1
                                if n.endswith(
                                    "layers.1.linear_attn.in_proj_qkvz.weight"
                                ) or n.endswith("layers.3.qkv_proj.weight"):
                                    samples.append(f"{n} amax={am:.2f}")
                        _i8_log.write(
                            f"LOAD-CENSUS int8_total={i8_total} zero={i8_zero}\n"
                        )
                        for s in samples:
                            _i8_log.write(f"LOAD-CENSUS {s}\n")
                    except Exception as e:
                        _i8_log.write(f"CENSUS-FAIL {e!r}\n")
                    return out

                _q35.Qwen3_5ForConditionalGeneration.load_weights = _lw_audited
            except Exception as e:
                _i8_log.write(f"CENSUS-WRAP-SKIP {e!r}\n")

            # Which early stage kills the activations: embedding or RMSNorm.
            try:
                import sglang.srt.layers.layernorm as _lnm

                _RMS = _lnm.RMSNorm
                _rms_fwd = _RMS.forward

                def _rms_audited(self, x, residual=None):
                    out = _rms_fwd(self, x, residual)
                    if not getattr(self, "_hcu_audited", False):
                        self._hcu_audited = True
                        try:
                            w = getattr(self, "weight", None)
                            o = out[0] if isinstance(out, tuple) else out
                            _i8_log.write(
                                f"RMS xamax={x.float().abs().max().item():.6f}"
                                f" wamax={0 if w is None else w.float().abs().max().item():.6f}"
                                f" oamax={o.float().abs().max().item():.6f}\n"
                            )
                        except Exception as e:
                            _i8_log.write(f"RMS-AUDIT-FAIL {e!r}\n")
                    return out

                _RMS.forward = _rms_audited
            except Exception as e:
                _i8_log.write(f"RMS-WRAP-SKIP {e!r}\n")

            try:
                import sglang.srt.layers.vocab_parallel as _vp

                _Emb = _vp.VocabParallelEmbedding
                # capture the UN-patched forward first (review P2): reading
                # _Emb.forward after the assignment below re-enters this
                # wrapper -- the self-referencing partial recursed forever on
                # the first embedding forward. Same shape as _rms_fwd above.
                _emb_orig_forward = _Emb.forward

                def _emb_audited(self, x):
                    orig = getattr(self, "_hcu_orig_fwd", None)
                    if orig is None:
                        orig = self._hcu_orig_fwd = functools.partial(
                            _emb_orig_forward, self
                        )
                    out = orig(x)
                    if not getattr(self, "_hcu_audited", False):
                        self._hcu_audited = True
                        try:
                            w = getattr(self, "weight", None)
                            _i8_log.write(
                                f"EMB xamax={x.float().abs().max().item() if x.numel() else 0:.6f}"
                                f" wamax={0 if w is None else w.float().abs().max().item():.6f}"
                                f" oamax={out.float().abs().max().item():.6f}\n"
                            )
                        except Exception as e:
                            _i8_log.write(f"EMB-AUDIT-FAIL {e!r}\n")
                    return out

                _Emb.forward = _emb_audited
            except Exception as e:
                _i8_log.write(f"EMB-WRAP-SKIP {e!r}\n")
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: int8 apply audit not installed", exc_info=True
            )
