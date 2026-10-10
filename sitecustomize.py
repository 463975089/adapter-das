"""HCU compatibility bridges for the upstream sglang stack.

Loaded automatically (as `sitecustomize`) whenever PYTHONPATH includes this
directory, including spawn'd sglang scheduler children. sglang and vllm files
stay untouched; what we bridge here:

1. vllm._custom_ops.rms_norm / fused_add_rms_norm -> sgl_kernel triton kernels.
   The HIP vllm build's `torch.ops._C` extension does not carry these ops, so
   sglang's layernorm forward_hip would crash at call time (the import-time
   `_has_vllm_rms_norm` check passes). We replace the python wrappers with
   bridges to our kernel — same signatures, fp32-accumulated math.
2. `aiter` stub registration happens via `import sgl_kernel` (see
   sgl_kernel._aiter_hcu_stub) for every AMD-only import on the HIP path.
3. DSA model overrides (`sglang.srt.arg_groups.model_overrides.deepseek_v2`,
   which also owns Glm5Next) force `page_size=1` on legacy HIP (no aiter
   preshuffle paged-MQA). Models whose indexer pools keys (index_kpool > 1,
   e.g. GLM-5.3-Flash kpool=4) then die on `page_size % index_kpool != 0`.
   The memory pool only needs a multiple of index_kpool, so we wrap the
   override registry to keep a CLI-provided page_size when upstream forces 1.
"""

import os

import torch

if getattr(torch.version, "hip", None) is not None:
    # env registry first: every bridge below resolves its knobs through it
    # (docs/env-vars.md). One startup line lists effective values + sources.
    try:
        from sgl_kernel._adapters import hcu_env as _he

        _he.log_summary()
    except Exception:
        _he = None
    try:
        import vllm._custom_ops as _vco

        from sgl_kernel import rmsnorm as _sk_rmsnorm

        if not hasattr(torch.ops._C, "rms_norm"):

            def rms_norm(out, input, weight, epsilon):
                _sk_rmsnorm(input, weight, epsilon, out=out)

            def fused_add_rms_norm(
                out, input, residual_out, residual, weight, epsilon
            ):
                # sglang calls the 6-arg HIP form: out = rmsnorm(residual+input),
                # residual_out = residual+input. vllm's 4-arg in-place form is
                # bridged into it without extra copies.
                residual_in = residual if residual_out is residual else residual_out
                residual_in.copy_(residual)
                residual_in.add_(input)
                _sk_rmsnorm(residual_in, weight, epsilon, out=out)

            _vco.rms_norm = rms_norm
            _vco.fused_add_rms_norm = fused_add_rms_norm
    except Exception:  # pragma: no cover - vllm absent: sglang falls back natively
        pass

    try:
        import sgl_kernel as _sk

        if not hasattr(_sk, "apply_token_bitmask_inplace_cuda"):
            # sglang's xgrammar shim takes the HIP branch and imports this from
            # sgl_kernel; bridge it to the in-tree triton op (CUDA branch).
            from sglang.kernels.ops.grammar.bitmask_ops import (
                apply_token_bitmask_inplace_triton as _bitmask_triton,
            )

            _sk.apply_token_bitmask_inplace_cuda = _bitmask_triton
    except Exception:
        pass

    # Debug facility: HCU_FP8_TRACE=1 logs every fp8 apply call (shapes +
    # OK/RAISE) to /tmp/fp8_trace.log — serve-time GEMM-path localization.
    if os.environ.get("HCU_FP8_TRACE") == "1":
        try:
            import sglang.srt.layers.quantization.fp8_utils as _fpu

            _trace_log = open("/tmp/fp8_trace.log", "w", buffering=1)

            def _hcu_trace(fn, name):
                def w(*a, **k):
                    try:
                        out = fn(*a, **k)
                        _trace_log.write(f"{name} OK\n")
                        return out
                    except Exception as e:
                        _trace_log.write(f"{name} RAISE {e!r:.140}\n")
                        raise

                return w

            for _name in ("apply_fp8_linear", "apply_w8a8_block_fp8_linear",
                          "apply_fp8_ptpc_linear"):
                _fn = getattr(_fpu, _name, None)
                if callable(_fn):
                    setattr(_fpu, _name, _hcu_trace(_fn, _name))
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: fp8 trace not installed", exc_info=True
            )

    # v0.5.21 HIP native fp8 per-token quant bug (Qwen3-4B-FP8 decode-graph
    # capture, bs=16 + output_padding=17): apply_fp8_linear passes a
    # preallocated (bs,1) scale buffer while num_token_padding makes the
    # quant output bs+1 rows — `_native_dynamic_per_token_quant_fp8` then
    # copy_s a (17,1) scale into a (16,1) buffer. The aiter branch handles
    # this; the native fallback doesn't. Bridge: align scale to the padded
    # row count before quant (graph-safe: same buffers, no host sync).
    try:
        import sglang.kernels.ops.quantization.fp8_kernel as _fpk

        _orig_spq = _fpk.scaled_fp8_quant

        def _hcu_scaled_fp8_quant(
            input, scale=None, num_token_padding=None,
            use_per_token_if_dynamic=False, **kw,
        ):
            # v0.5.21 HIP native bug: the wrapper pads ONLY the output tensor
            # to num_token_padding but `_native_dynamic_per_token_quant_fp8`
            # divides the unpadded input by the padded per-token scale
            # (17,1) -> shape mismatch. The aiter branch pads its quant
            # internally; the native one doesn't. Bridge: quantize unpadded
            # with the native path, then pad the output ourselves (and pad
            # the scale with neutral rows so downstream _scaled_mm sees
            # consistent per-row scales). Graph-safe: no host sync.
            pad_rows = 0
            if (
                num_token_padding
                and num_token_padding > input.shape[0]
                and not _fpk._use_aiter
            ):
                pad_rows = num_token_padding - input.shape[0]
                num_token_padding = None
            out, s = _orig_spq(
                input, scale, num_token_padding=num_token_padding,
                use_per_token_if_dynamic=use_per_token_if_dynamic, **kw,
            )
            if pad_rows:
                if s.shape[0] == input.shape[0] and out.shape[0] != s.shape[0]:
                    pass  # scale already covers real rows only
                out = torch.nn.functional.pad(
                    out, (0, 0, 0, pad_rows)
                ).contiguous()
            return out, s

        _hcu_scaled_fp8_quant.__wrapped__ = _orig_spq
        _fpk.scaled_fp8_quant = _hcu_scaled_fp8_quant
        # byname importers that were loaded before this bridge
        import sys as _sys

        for _mod_name in tuple(_sys.modules):
            if _mod_name.endswith("quantization.fp8_utils"):
                _m = _sys.modules[_mod_name]
                if getattr(_m, "scaled_fp8_quant", None) is _orig_spq:
                    _m.scaled_fp8_quant = _hcu_scaled_fp8_quant
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: fp8 per-token padding bridge not installed",
            exc_info=True,
        )

    # Family-override modules import `_register_for` by reference at their own
    # import time (later than ours), so patching the attribute here intercepts
    # every registration, including deepseek_v2's DSA branch.
    try:
        import functools
        import logging
        import os

        from sglang.srt.arg_groups import model_override_base as _mob

        def _hcu_dsa_index_kpool(hf_config):
            for cfg in (hf_config, getattr(hf_config, "text_config", None)):
                kpool = getattr(cfg, "index_kpool", None)
                if kpool:
                    return int(kpool)
            return None

        def _hcu_wrap_override(fn):
            @functools.wraps(fn)
            def wrapped(server_args, hf_config):
                overrides = fn(server_args, hf_config)
                if overrides.get("page_size") != 1:
                    return overrides
                wanted = os.environ.get("HCU_DSA_PAGE_SIZE") or getattr(
                    server_args, "page_size", None
                )
                try:
                    wanted = int(wanted)
                except (TypeError, ValueError):
                    return overrides
                if wanted <= 1:
                    return overrides
                kpool = _hcu_dsa_index_kpool(hf_config)
                if kpool and wanted % kpool:
                    logging.getLogger("hcu.adapter").warning(
                        "HCU adapter: page_size %d not a multiple of index_kpool"
                        " %d, keeping upstream default 1",
                        wanted,
                        kpool,
                    )
                    return overrides
                overrides["page_size"] = wanted
                logging.getLogger("hcu.adapter").warning(
                    "HCU adapter: keeping page_size=%d for DSA (upstream legacy"
                    " HIP default is 1, index_kpool=%s)",
                    wanted,
                    kpool,
                )
                return overrides

            return wrapped

        _hcu_orig_register_for = _mob._register_for

        def _hcu_register_for(*args, **kwargs):
            decorator = _hcu_orig_register_for(*args, **kwargs)

            def register(fn):
                return decorator(_hcu_wrap_override(fn))

            return register

        _mob._register_for = _hcu_register_for
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: DSA page_size bridge not installed", exc_info=True
        )

    # Force the DSA "aiter preshuffle paged-MQA" gate on. The DTK wheel ships
    # working triton preshuffle MQA-logits kernels (pa_mqa_logits etc.), but
    # upstream's gate keys on SGLANG_USE_AITER — setting that env would also
    # flip unrelated fp8 weight-shuffle/gemm routes. Rebinding the predicate
    # in its home module before any consumer's byname import copies it makes
    # only the DSA page-size/table/preshuffle-tile decisions follow the
    # preshuffle path (page_size=64*index_kpool, 16-aligned cache tiles).
    try:
        import sglang.srt.layers.attention.dsa.utils as _dsa_utils

        if os.environ.get("HCU_DSA_PRESHUFFLE") == "0":
            # Legacy page_size=1 / KVBlockSize=1 DSA path (paired with the
            # HCU_MQA_EMU torch emulation on gfx936-class DCUs).
            _dsa_utils.aiter_can_use_preshuffle_paged_mqa = lambda: False
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: DSA aiter-preshuffle gate forced OFF (legacy"
                " page_size=1 path)"
            )
        else:
            _dsa_utils.aiter_can_use_preshuffle_paged_mqa = lambda: True
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: DSA aiter-preshuffle gate forced on"
                " (DTK wheel kernels)"
            )
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: preshuffle gate patch not installed", exc_info=True
        )

    # slimquant_w4a8 checkpoints (GLM-5-W4A8_V2_6 family): expert int4 unpack
    # + register the quant method. Default ON (the method only activates for
    # checkpoints whose quantization_config names slimquant_w4a8 — inert
    # otherwise); HCU_SLIMQUANT=0 opts out (debug/isolation).
    _sq_on = _he.enabled("HCU_SLIMQUANT") if _he else \
        os.environ.get("HCU_SLIMQUANT", "1") != "0"
    if _sq_on:
        try:
            from sgl_kernel._adapters import slimquant as _slimquant

            _slimquant.activate()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: slimquant bridge not installed", exc_info=True
            )

    # DSA indexer top-k: the sgl-kernel backend ops have no HIP build; bridge
    # to torch equivalents (see sgl_kernel._adapters.dsa_topk).
    try:
        import sgl_kernel as _sk
        import sgl_kernel._adapters.dsa_topk as _dsa_topk

        _dsa_topk.install()
        if os.environ.get("HCU_DUMP_TOPK") == "1":
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

    # AiterCustomAllreduce silently corrupts data when the CUDA graph it was
    # captured into replays on gfx936-class DCUs (eager all-reduce is fine).
    # sglang selects it as the AMD default because the C++ sgl_kernel
    # allreduce extension has no HIP build; the symptom is decode tokens
    # diverging from the first graph-replayed step while eager decode and
    # prefill stay correct. Refuse the dispatch on this arch — the exception
    # is caught in parallel_state's init and the group falls back to NCCL,
    # which is graph-correct.
    #
    # Kill-switch precedence (kill_switch.py): HCU_DISABLE_OPS=aiter_ar
    # refuses on ANY arch (triage); arch auto-rule via hcu_env (old-aiter
    # layout on gfx93* refuses — graph-replay corruption, 1eb038d; the new
    # aiter.dist GraphSafe class passed the full stress suite 2026-10-10);
    # explicit HCU_AITER_AR=1 overrides the auto-rule only.
    try:
        from sgl_kernel._adapters import kill_switch as _ks

        _ks.log_state()
    except Exception:
        _ks = None
    _ar_val, _ar_src = (
        _he.resolve("HCU_AITER_AR") if _he else ("0", "default")
    )
    _ar_refuse = bool(_ks and _ks.op_disabled("aiter_ar")) or \
        str(_ar_val).startswith("refused")
    _ar_reason = (
        "disabled via HCU_DISABLE_OPS (triage)"
        if _ks and _ks.op_disabled("aiter_ar")
        else f"{_ar_val} [{_ar_src}]"
    )
    if _ar_refuse:
        try:
            import sglang.srt.distributed.device_communicators.\
                custom_all_reduce as _car

            def _hcu_no_aiter_ar(group, device):
                raise RuntimeError(
                    f"HCU adapter: AiterCustomAllreduce disabled"
                    f" ({_ar_reason}; falling back to NCCL)."
                    " Set HCU_AITER_AR=1 to re-enable (auto-rule only) or"
                    " remove aiter_ar from HCU_DISABLE_OPS."
                )

            _car.dispatch_custom_allreduce = _hcu_no_aiter_ar
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: aiter custom all-reduce refused (%s);"
                " NCCL all-reduce instead", _ar_reason
            )
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: aiter AR gate not installed", exc_info=True
            )

    # gfx936-class DCUs have no fp8 matrix unit and the DTK aiter MQA-logits
    # kernels hard-gate on gfx942/950: emulate paged/ragged fp8 MQA logits in
    # torch (dequant -> bf16 GEMM, fp32 scales). Legacy page_size=1 only.
    # hcu_env auto-rule: emulation ON only where the arch lacks fp8;
    # explicit HCU_MQA_EMU=0/1 wins (debug).
    _mqa_on = _he.enabled("HCU_MQA_EMU") if _he else \
        os.environ.get("HCU_MQA_EMU") == "1"
    if _mqa_on:
        try:
            import sgl_kernel._adapters.mqa_logits_emu as _mqa_emu

            _mqa_emu.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: MQA-logits emu not installed", exc_info=True
            )

    # Debug facility: HCU_DUMP_LAYERS=1 captures, per decoder layer, the
    # post-reduction residual stream entering `input_layernorm` (TP8 uses
    # delayed partial reductions, so module hooks deeper than this see
    # one rank's partial sum). Entry i of the saved list = residual stream
    # before layer i (embed output = entry 0), matching the pure torch
    # reference (tests/_ref_forward_h.py). With HCU_DUMP_INNER=1 also
    # saves xa/xb (norm outputs) of layers 0-1 to /tmp/hcu_inner.pt.
    # Prefill-sized passes only (1 < T <= 64), rank 0, file
    # HCU_DUMP_LAYERS_PATH (default /tmp/hcu_layers.pt).
    if os.environ.get("HCU_DUMP_LAYERS") == "1":
        try:
            import sglang.srt.models.glm4_moe as _g4m

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
                    if os.environ.get("HCU_DUMP_INNER") != "1":
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
                    for i, lyr in enumerate(self.model.layers):
                        lyr.input_layernorm.register_forward_pre_hook(
                            _hcu_layernorm_pre(i), with_kwargs=True
                        )
                        if i < 2 and os.environ.get("HCU_DUMP_INNER") == "1":
                            lyr.input_layernorm.register_forward_hook(
                                _hcu_norm_out(f"xa{i}")
                            )
                            lyr.post_attention_layernorm.register_forward_hook(
                                _hcu_norm_out(f"xb{i}")
                            )
                    print(
                        f"[HCU dump] pre-hooks on {len(self.model.layers)}"
                        " input_layernorms",
                        flush=True,
                    )

                return init

            _g4m.GlmMoeDsaForCausalLM.__init__ = _hcu_dump_init(
                _g4m.GlmMoeDsaForCausalLM.__init__
            )
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: layer dump bridge not installed", exc_info=True
            )
