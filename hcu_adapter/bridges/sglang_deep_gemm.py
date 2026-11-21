# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""DeepGEMM DTK bridge (EP, 2026-10-11)

Extracted verbatim from bridges/sglang.py install() (9. DeepGEMM DTK bridge (EP, 2026-10-11): opt-in HCU_DEEPGEMM_BRIDGE=1.)
under layout governance 2026-11-18; zero logic change. Section notes:
9. DeepGEMM DTK bridge (EP, 2026-10-11): opt-in HCU_DEEPGEMM_BRIDGE=1.
   The deepep_ll/deepep_normal MoE formats have no triton runner; the
   only reachable compute backends are DeepGEMM (CUDA) / aiter / npu,
   and the wrapper's platform gate hard-disables DeepGEMM on HIP even
   though the DTK deepgemm wheel carries upstream-named grouped GEMMs
   (incl. m_grouped_bf16_gemm_nt_masked/_contiguous). Two pieces:
   (a) hcu_adapter.features.deep_gemm.deep_gemm_dtk aliases DTK `deepgemm` as
       `deep_gemm` in sys.modules;
   (b) the overlay finder flips configurer.ENABLE_JIT_DEEPGEMM to True
       after its (platform-gated) computation, which routes DeepEP MoE
       to the modern FusedMoE runner path. fp8 DeepEP stays unverified:
       scale-layout transforms raise loudly in the shim until validated.
the finder only patches loads that happen AFTER registration;
early adapter imports (e.g. the MQA emu installer) may already
have cached these modules behind our back -- flip them too.
entrypoint/compile_utils bind `deep_gemm` under a module
level `if ENABLE_JIT_DEEPGEMM:` that already ran with the
pre-flip False -- rebind the name or their functions hit
NameError.
The retro-flip above fixes the FLAG but cannot materialize the
definitions inside entrypoint's module-level `if ENABLE_JIT_-
DEEPGEMM:` (executed earlier with the pre-bridge False): the
fp8 arms then die on deep_gemm_wrapper.get_mn_major_tma_-
aligned_tensor (DSV4F fp8 dp8ep8, validation-host-16 2026-10-21; audit:
cached modules carry FLIPPED=True yet lack the fn; purge +
reimport defines it against the shim + identity layout). Drop
the stale modules AFTER the flip so every later import
(including the _dgr import below) re-executes under True.
Re-import NOW and verify; if the re-exec still lands False
(import-order dependent, scheduler spawn included), synthesize
the identity outright -- the triton fp8 kernels consume the
LOGICAL scale layout, so identity is the exact contract either
way and the fp8 arms stop depending on this import dance.
DeepEP-normal (bf16 contiguous) MoE runner: upstream computes
`_legacy_silu_and_mul = silu_and_mul` only on CUDA, so HIP
runs carry None and the prefill path dies on call. The helper
imports the PACKAGE face (fused-op dispatch -> AOT on HIP);
see its docstring for why the module body is forbidden.
HCU kpool write tier: the stock triton writer family is the default
(2026-10-15 初判 int64 scalar-load div-4 bug was later traced to a
probe-side buf layout error; the kernels are healthy on the real 2-D
(pages, page_size*(D+S)) layout — see
tests/kernels/test_kpool_write_gfx938.py). HCU_KPOOL_WRITE_EMU=1
keeps the bit-equal torch emu as an explicit A/B fallback / DTK
regression watch.
HCU kpool read tiers: aiter's cp_gather_indexer_k_quant_cache
(GetKAndS.aiter preshuffle arm) illegal-addresses / returns zeros on
gfx93x. Default on gfx93x is the preshuffle-aware triton gather
(kpool_read_triton.py, one launch, same 16-tile map as the writer);
HCU_KPOOL_READ_EMU=1 keeps the torch reference as an explicit A/B
fallback. Non-gfx93 platforms keep the upstream aiter kernel.
HCU read-scope probe (2026-10-15): GetKAndS never fires on this stack —
watch the cache-level read hub instead and dump both branches' output.
HCU ragged-MQA input probe (2026-10-15): prefill topk score arrives all
zero at _topk_unfused; the score producer is the plain-Indexer
fp8_mqa_logits arm (aiter alias -> torch emu). Pin q/k magnitudes at the
emu boundary to decide whether its K source cache is ever written.
HCU kpool plan-trap (2026-10-15, wave3): why does the extend-side pooled
HCU kpool store-trace (2026-10-15): which pooled-write entrypoint actually
runs. Must live AFTER the DeepGEMM bridge install: importing
dsa_indexer_kpool earlier trips its module-level
`try: import deep_gemm except ImportError as e: deep_gemm = e` while
the shim is absent, permanently binding the name to the exception
(serve died in IndexerKPool.__init__ on deep_gemm.get_num_sms()).
"""
from __future__ import annotations

import logging
import torch
from hcu_adapter.infra import hcu_env


def install(_he=None) -> None:
    from hcu_adapter.bridges.sglang import _bind_deepgemm_legacy_silu  # noqa: F401
    try:
        if _he is not None and _he.enabled("HCU_DEEPGEMM_BRIDGE"):
            from hcu_adapter.features.deep_gemm import deep_gemm_dtk as _dgd

            _dgd.install()

            from hcu_adapter._base import (
                LibStubFinder as _LSF,
                register_finder as _reg,
            )

            _reg(
                _LSF(
                    "sglang.srt.layers.deep_gemm_wrapper",
                    allowlist={
                        "sglang.srt.layers.deep_gemm_wrapper.configurer": {
                            "ENABLE_JIT_DEEPGEMM": True
                        }
                    },
                )
            )
            import sys as _sys

            for _mn in (
                "sglang.srt.layers.deep_gemm_wrapper.configurer",
                "sglang.srt.layers.deep_gemm_wrapper.entrypoint",
                "sglang.srt.layers.deep_gemm_wrapper.compile_utils",
                "sglang.srt.layers.deep_gemm_wrapper",
            ):
                _m = _sys.modules.get(_mn)
                if _m is None:
                    continue
                if hasattr(_m, "ENABLE_JIT_DEEPGEMM"):
                    _m.ENABLE_JIT_DEEPGEMM = True
                    _m.HCU_DEEPGEMM_BRIDGE_FLIPPED = True
                if not hasattr(_m, "deep_gemm"):
                    _m.deep_gemm = _sys.modules.get("deep_gemm")

            for _mn in (
                "sglang.srt.layers.deep_gemm_wrapper",
                "sglang.srt.layers.deep_gemm_wrapper.configurer",
                "sglang.srt.layers.deep_gemm_wrapper.entrypoint",
                "sglang.srt.layers.deep_gemm_wrapper.compile_utils",
            ):
                _sys.modules.pop(_mn, None)

            import sglang.srt.layers.deep_gemm_wrapper as _dw

            if not hasattr(_dw, "get_mn_major_tma_aligned_tensor"):
                _dw.get_mn_major_tma_aligned_tensor = (
                    lambda t, *a, **k: t
                )
                _ep = _sys.modules.get(
                    "sglang.srt.layers.deep_gemm_wrapper.entrypoint"
                )
                if _ep is not None:
                    _ep.get_mn_major_tma_aligned_tensor = (
                        _dw.get_mn_major_tma_aligned_tensor
                    )
                logging.getLogger("hcu.adapter").warning(
                    "HCU DeepGEMM bridge: wrapper re-import still lacked "
                    "get_mn_major_tma_aligned_tensor; synthesized identity."
                )

            _bind_deepgemm_legacy_silu()

            logging.getLogger("hcu.adapter").info(
                "HCU DeepGEMM bridge: deep_gemm -> DTK deepgemm alias + "
                "ENABLE_JIT_DEEPGEMM flipped (EP opt-in)"
            )
    except ImportError:
        pass
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: deepgemm bridge not installed", exc_info=True
        )

    try:
        from hcu_adapter.infra import hcu_env as _kpw_env

        if _kpw_env.enabled("HCU_KPOOL_WRITE_EMU"):
            from hcu_adapter.features.dsa import kpool_write_emu

            kpool_write_emu.install_patch()
            logging.getLogger("hcu.adapter").info(
                "HCU adapter: kpool index-K write kernels routed to torch emu "
                "(explicit fallback; stock triton is the default)"
            )
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: kpool write emu not installed", exc_info=True
        )
    try:
        from hcu_adapter.infra import hcu_env as _kpr_env

        if _kpr_env.enabled("HCU_KPOOL_READ_EMU"):
            mode = "emu"
        elif _kpr_env.arch_name().startswith("gfx93"):
            mode = "triton"
        else:
            mode = "aiter"
        if mode != "aiter":
            from hcu_adapter.features.dsa import kpool_write_emu as _kwe

            _kwe.install_read_patch(mode)
            logging.getLogger("hcu.adapter").info(
                "HCU adapter: index-K cache gather (GetKAndS.aiter) routed "
                "to %s tier (aiter kernel dead on gfx93x)", mode
            )
        else:
            logging.getLogger("hcu.adapter").info(
                "HCU adapter: index-K cache gather keeps upstream aiter "
                "(non-gfx93 platform)"
            )
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: kpool read tier not installed", exc_info=True
        )
    try:
        if hcu_env.enabled("HCU_DUMP_TOPK"):
            import sglang.srt.mem_cache.index_key_cache as _ikc

            if not getattr(_ikc, "_hcu_gks_trap", False):
                _ikc._hcu_gks_trap = True
                _gk = {"n": 0}
                _gks_orig = _ikc.IndexKeyCache.get_k_and_scale

                def _gks_t(self, layer_id, seq_len_tensor, page_indices, seq_len_sum, max_seq_len):
                    buf = self.get_buffer(layer_id)
                    out = _gks_orig(self, layer_id, seq_len_tensor, page_indices, seq_len_sum, max_seq_len)
                    if _gk["n"] < 3:
                        try:
                            print(
                                f"[HCU kgks] L{layer_id} pages={tuple(page_indices.shape)} "
                                f"sum={seq_len_sum} buf_abs={float(buf.float().abs().sum()):.1f} "
                                f"k_abs={float(out[0].float().abs().sum()):.1f} "
                                f"s_abs={float(out[1].float().abs().sum()):.3f}",
                                flush=True,
                            )
                            _gk["n"] += 1
                        except Exception as _e:
                            print(f"[HCU kgks] err {_e}", flush=True)
                            _gk["n"] += 1
                    return out

                _ikc.IndexKeyCache.get_k_and_scale = _gks_t
                print("[HCU kgks] installed", flush=True)
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: gks trap not installed", exc_info=True
        )

    try:

        if hcu_env.enabled("HCU_DUMP_TOPK"):
            from hcu_adapter.features.dsa import mqa_logits_emu as _mle

            if not getattr(_mle, "_hcu_trap", False):
                _mle._hcu_trap = True
                _qt = {"n": 0}
                _orig_rml = _mle._ragged_fp8_mqa_logits

                def _rml_t(*a, **kw):
                    out = _orig_rml(*a, **kw)
                    if _qt["n"] < 4:
                        try:
                            import torch as _th

                            parts = []
                            for i, t in enumerate(a[:6]):
                                if isinstance(t, _th.Tensor):
                                    parts.append(
                                        f"a{i}={tuple(t.shape)} abs={float(t.float().abs().sum()):.2f}"
                                    )
                                else:
                                    parts.append(f"a{i}={t}")
                            if out is not None:
                                of = out.float()
                                parts.append(
                                    f"out_abs={float(of.abs().sum()):.2f} "
                                    f"fin={float(_th.isfinite(of).float().mean()):.3f} "
                                    f"nz%={float((of != 0).float().mean()):.4f}"
                                )
                            print(f"[HCU mqr] {' '.join(parts)}", flush=True)
                            _qt["n"] += 1
                        except Exception as _e:
                            print(f"[HCU mqr] err {_e}", flush=True)
                            _qt["n"] += 1
                    return out

                _mle._ragged_fp8_mqa_logits = _rml_t
                try:
                    import aiter.ops.triton.fp8_mqa_logits as _am

                    if getattr(_am, "fp8_mqa_logits", None) is _orig_rml:
                        _am.fp8_mqa_logits = _rml_t
                except Exception:
                    pass
                print("[HCU mqr] installed", flush=True)
    except Exception as _e:
        print(f"[HCU mqr] install-fail {_e}", flush=True)

    try:
        if hcu_env.enabled("HCU_DUMP_TOPK"):
            import traceback as _tbk

            import sglang.srt.layers.attention.dsa.dsa_indexer_metadata as _dim

            if not getattr(_dim, "_hcu_tk_trap", False):
                _dim._hcu_tk_trap = True
                _tn = {"n": 0}
                _tt_orig = _dim.DSAIndexerMetadata.topk_transform

                def _tt_t(self, *a, **kw):
                    out = _tt_orig(self, *a, **kw)
                    if _tn["n"] < 2:
                        try:
                            logits = kw.get("logits") or (a[0] if a else None)
                            msg = "logits=?"
                            if isinstance(logits, torch.Tensor):  # noqa
                                lf = logits.float()
                                msg = (
                                    f"logits={tuple(logits.shape)} "
                                    f"abs={float(lf.abs().sum()):.2f} "
                                    f"nz%={float((lf != 0).float().mean()):.4f} "
                                    f"max={float(lf.max()):.4f}"
                                )
                            print(f"[HCU tkt] {msg}", flush=True)
                            print(
                                "".join(_tbk.format_stack()[-8:-2]),
                                flush=True,
                            )
                            _tn["n"] += 1
                        except Exception as _e:
                            print(f"[HCU tkt] err {_e}", flush=True)
                            _tn["n"] += 1
                    return out

                _dim.DSAIndexerMetadata.topk_transform = _tt_t
                print("[HCU tkt] installed", flush=True)
    except Exception as _e:
        print(f"[HCU tkt] install-fail {_e}", flush=True)

    try:
        from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as _dkp
        from sglang.srt.layers.attention.dsa import kpool_plan as _kpm

        if hcu_env.enabled("HCU_DUMP_TOPK") and not getattr(
            _kpm, "_hcu_plan_trap", False
        ):
            _kpm._hcu_plan_trap = True
            _pt = {"n": 0}

            _pem = _kpm.init_kpool_extend_metadata

            def _pem_t(metadata, forward_batch, **kw):
                out = _pem(metadata, forward_batch, **kw)
                if _pt["n"] < 10:
                    plan = getattr(out, "kpool_extend_plan", None)
                    pool_size = kw.get("pool_size", -1)
                    rps = kw.get("real_page_size", -1)
                    print(
                        f"[HCU kpplan] extend mode={forward_batch.forward_mode} "
                        f"pool={pool_size} rps={rps} "
                        f"scpu={forward_batch.seq_lens_cpu is not None} "
                        f"ecpu={forward_batch.extend_seq_lens_cpu is not None} "
                        f"plan={'OK' if plan is not None else 'NONE'}",
                        flush=True,
                    )
                    _pt["n"] += 1
                return out

            _kpm.init_kpool_extend_metadata = _pem_t

            _cwe = _dkp.IndexerKPool._compress_write_extend
            _ct3 = {"n": 0}

            def _cwe_t(self, key, gate_score, positions, forward_batch, layer_id, metadata, **kw):
                plan = getattr(
                    getattr(metadata, "attn_metadata", metadata),
                    "kpool_extend_plan",
                    None,
                )
                if _ct3["n"] < 8:
                    wr = getattr(plan, "writes", None)
                    tl_ = getattr(plan, "tails", None)
                    print(
                        f"[HCU kpcwe] L{layer_id} key={tuple(key.shape)} "
                        f"plan={'None' if plan is None else 'OK'} "
                        f"writes={'NA' if wr is None else wr.is_empty} "
                        f"tails={'NA' if tl_ is None else tl_.is_empty}",
                        flush=True,
                    )
                    _ct3["n"] += 1
                return _cwe(self, key, gate_score, positions, forward_batch, layer_id, metadata, **kw)

            _dkp.IndexerKPool._compress_write_extend = _cwe_t

            _cw = _dkp.IndexerKPool._compress_write
            _ct2 = {"n": 0}

            def _cw_t(self, *a, **kw):
                out = _cw(self, *a, **kw)
                if _ct2["n"] < 12:
                    fb = a[2] if len(a) > 2 else kw.get("forward_batch")
                    mode = getattr(fb, "forward_mode", "?")
                    layer = getattr(self, "layer_id", "?")
                    print(
                        f"[HCU kpcw] L{layer} mode={mode}", flush=True
                    )
                    _ct2["n"] += 1
                return out

            _dkp.IndexerKPool._compress_write = _cw_t
            print("[HCU kpplan] installed", flush=True)
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: kpool plan trap not installed", exc_info=True
        )

    try:
        from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as _dkp
        from sglang.srt.layers.attention.dsa import dsa_indexer as _dstm
        from sglang.srt.layers.attention.dsa import kpool_fp8_index as _kfi

        if hcu_env.enabled("HCU_DUMP_TOPK") and not getattr(
            _kfi, "_hcu_kpool_trap", False
        ):
            _kfi._hcu_kpool_trap = True
            _gt = {"n": 0}

            _gather_orig = _kfi.gather_index_k_scale_prefix_into

            def _gather_trap(pool, buf, page_indices, seq_len, k_out, scale_out, **kw):
                out = _gather_orig(
                    pool=pool, buf=buf, page_indices=page_indices,
                    seq_len=seq_len, k_out=k_out, scale_out=scale_out, **kw,
                )
                if _gt["n"] < 3:
                    try:
                        import torch as _th  # noqa

                        layer = kw.get("layer_id", _gt.get("layer", 0))
                        wr = (
                            pool.get_index_k_with_scale_buffer(layer_id=layer)
                            if hasattr(pool, "get_index_k_with_scale_buffer")
                            else None
                        )
                        msg = (
                            f"gather pagetbl={tuple(page_indices.shape)} seqlen={seq_len} "
                            f"read_buf ptr={buf.data_ptr()} abs={float(buf.float().abs().sum()):.1f}"
                        )
                        if wr is not None:
                            same = (
                                "SAME"
                                if wr.data_ptr() == buf.data_ptr()
                                else "DIFF"
                            )
                            msg += (
                                f" | write_pool ptr={wr.data_ptr()} {same} "
                                f"abs={float(wr.float().abs().sum()):.1f}"
                            )
                        msg += f" | k_out after abs={float(k_out.float().abs().sum()):.3f}"
                        print(f"[HCU kprb] {msg}", flush=True)
                        _gt["n"] += 1
                    except Exception as _e:
                        print(f"[HCU kprb] err {_e}", flush=True)
                        _gt["n"] += 1
                return out

            _kfi.gather_index_k_scale_prefix_into = _gather_trap
            print("[HCU kprb] gather trap installed", flush=True)

            _tkh = {"n": 0}

            def _arm_trap(name, fn):
                def w(self, forward_batch, layer_id, q_fp8, weights, metadata, **kw):
                    plan = getattr(
                        getattr(metadata, "attn_metadata", metadata) if metadata is not None else None,
                        "kpool_extend_plan",
                        None,
                    )
                    ku8 = getattr(plan, "ragged_k_u8", None)
                    out = fn(self, forward_batch, layer_id, q_fp8, weights, metadata, **kw)
                    if _tkh["n"] < 6:
                        try:
                            qa = float(q_fp8.float().abs().sum())
                            wa = float(weights.float().abs().sum()) if weights is not None else -1
                            ka = float(ku8.float().abs().sum()) if ku8 is not None else -1
                            oa = float(out.float().abs().sum()) if out is not None else -1
                            print(
                                f"[HCU kparm] {name} L{layer_id} q={qa:.1f} w={wa:.1f} "
                                f"ku8={ka:.1f} out={oa:.1f} shape={tuple(out.shape) if out is not None else None}",
                                flush=True,
                            )
                            _tkh["n"] += 1
                        except Exception as _e:
                            print(f"[HCU kparm] err {_e}", flush=True)
                            _tkh["n"] += 1
                    return out

                return w

            _dkp.IndexerKPool._get_topk_ragged_kpool_plan = _arm_trap(
                "PLAN", _dkp.IndexerKPool._get_topk_ragged_kpool_plan
            )
            _dkp.IndexerKPool._get_topk_ragged = _arm_trap(
                "RAGGED", _dkp.IndexerKPool._get_topk_ragged
            )

            _skh = {"n": 0}
            _sk_orig = _dkp.IndexerKPool._forward_cuda_skip_logits

            def _sk_t(self, *a, **kw):
                _names = ("x", "positions", "forward_batch", "layer_id", "act_quant", "metadata")
                _d = dict(zip(_names, a))
                x = kw.get("x", _d.get("x"))
                layer_id = kw.get("layer_id", _d.get("layer_id"))
                metadata = kw.get("metadata", _d.get("metadata"))
                plan = getattr(
                    getattr(metadata, "attn_metadata", metadata, ) if metadata is not None else None,
                    "kpool_extend_plan",
                    None,
                ) if metadata is not None else None
                ku8 = getattr(plan, "ragged_k_u8", None)
                if _skh["n"] < 4:
                    try:
                        msg = (
                            f"[HCU kpsl] L{layer_id} nargs={len(a)} "
                            f"x_abs={float(x.float().abs().sum()):.1f} "
                            f"plan={'None' if plan is None else 'OK'} "
                            f"total_k_rows={getattr(plan, 'ragged_total_k_rows', 'NA')}"
                        )
                        if ku8 is not None:
                            msg += f" ragged_k_u8 abs={float(ku8.float().abs().sum()):.1f}"
                        else:
                            msg += " ragged_k_u8=NONE"
                        print(msg, flush=True)
                        _skh["n"] += 1
                    except Exception as _e:
                        print(f"[HCU kpsl] err {_e}", flush=True)
                        _skh["n"] += 1
                return _sk_orig(self, *a, **kw)

            _dkp.IndexerKPool._forward_cuda_skip_logits = _sk_t

            _kt = {"n": 0}

            _asm = _kfi.kpool_assemble_softmax_rotate_write_cache

            def _asm_t(*a, **kw):
                out = _asm(*a, **kw)
                if _kt["n"] < 4:
                    try:
                        import torch as _th

                        buf = kw.get("buf")
                        if buf is None and len(a) > 1:
                            buf = a[1]
                        chunk_k = kw.get("chunk_k")
                        if chunk_k is None and len(a) > 2:
                            chunk_k = a[2]
                        loc = kw.get("loc")
                        if loc is None and len(a) > 11:
                            loc = a[11]
                        msg = f"asm npos={a[6].shape[0] if len(a) > 6 and hasattr(a[6], 'shape') else '?'}"
                        if buf is not None:
                            nb = buf.float().abs().sum()
                            msg += f" buf_abs_after={float(nb):.1f}"
                        if chunk_k is not None:
                            msg += f" ck={tuple(chunk_k.shape)} abs={float(chunk_k.float().abs().sum()):.1f}"
                        if loc is not None:
                            l0 = loc[:4].tolist()
                            msg += f" loc[:4]={l0}"
                        print(f"[HCU kpasm] {msg}", flush=True)
                        _kt["n"] += 1
                    except Exception as _e:
                        print(f"[HCU kpasm] err {_e}", flush=True)
                        _kt["n"] += 1
                return out

            _kfi.kpool_assemble_softmax_rotate_write_cache = _asm_t

            _sct = _kfi.scatter_kpool_tail_updates
            _sc = {"n": 0}

            def _sct_t(*a, **kw):
                out = _sct(*a, **kw)
                if _sc["n"] < 2:
                    print("[HCU kpsct] called", flush=True)
                    _sc["n"] += 1
                return out

            _kfi.scatter_kpool_tail_updates = _sct_t

            _kwc = _dkp.IndexerKPool._write_compressed_pooled_index_cache
            _kn = {"n": 0}

            def _wc_t(self, slot_k, slot_score, write_locs, forward_batch, layer_id, **kw):
                out = _kwc(self, slot_k, slot_score, write_locs, forward_batch, layer_id, **kw)
                if _kn["n"] < 3:
                    try:
                        sk = slot_k.float()
                        pool = _dstm.get_token_to_kv_pool()
                        buf = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
                        nb = float(buf.view(torch.float8_e4m3fn).float().abs().sum()) if buf.dtype == torch.uint8 else float(buf.float().abs().sum())
                        print(
                            f"[HCU kpoolwc] L{layer_id} slot_k {tuple(slot_k.shape)} "
                            f"abs={float(sk.abs().sum()):.3f} nz%={float((sk != 0).float().mean()):.3f} "
                            f"locs={'None' if write_locs is None else (int(write_locs.min()), int(write_locs.max()))} "
                            f"buf_abs_after={nb:.1f}",
                            flush=True,
                        )
                        _kn["n"] += 1
                    except Exception as _e:
                        print(f"[HCU kpoolwc] data-err {_e}", flush=True)
                        _kn["n"] += 1
                return out

            _dkp.IndexerKPool._write_compressed_pooled_index_cache = _wc_t
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: kpool trap not installed", exc_info=True
        )
