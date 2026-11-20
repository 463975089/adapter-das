# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Legacy sglang compatibility chain behind an explicit install boundary.

Bridge rule (see skills/hcu-adapter-dev): when rebinding a host operator that
has a tier in sglang's kernel catalog, import the PACKAGE face
(``sglang.kernels.ops.<domain>``) so the fused-op dispatch picks the backend.
Importing the module body (``sglang.kernels.ops.<domain>.<module>``) bypasses
that dispatch and pins one implementation; it is allowed only with an
inline ``JIT 唯一实现`` note (or a pointer to the alternative tier).
"""

_JIT_BODY = "sglang.kernels.ops.activation.activation"


def _bind_deepgemm_legacy_silu() -> bool:
    """Rebind the DeepGEMM runner's `_legacy_silu_and_mul` through the package face.

    Upstream binds ``_legacy_silu_and_mul`` at import time:

        if not (_is_npu or _is_hip) and _is_cuda:   # JIT module body
        elif _is_musa: ...
        else: _legacy_silu_and_mul = None            # HIP pre-seam

    The adapter's platform seam answers CUDA for sglang, so ``is_hip()`` is
    False by the time the runner is imported and upstream binds the JIT module
    body — that is the 33s ninja build (``sgl_kernel_jit_activation_*``)
    observed in the L1 kernels segment. Only a boot order that imports the
    runner before the seam leaves ``None``.

    The bridge therefore replaces **exactly the JIT module-body function** with
    the PACKAGE face (``sglang.kernels.ops.activation``). The fused-op dispatch
    then runs on the real platform: its JIT candidate is CUDA-only and the
    fused-op capability gate keeps upstream's HIP truth (see the capability
    bridge below), so the call resolves to AOT = our shadow ``sgl_kernel`` op
    and continues through HCU_SILU_MUL_IMPL (auto = lightop > aiter > triton;
    explicit lightop|aiter|triton|torch) like every other ``silu_and_mul``
    call. Any other host binding is left untouched.
    If upstream reorders the priority the package face follows automatically;
    the bridge never pins an implementation.

    Returns True when the runner carries a usable binding after the call,
    False when the host package is absent.
    """
    import logging

    log = logging.getLogger("hcu.adapter")
    try:
        import sglang.srt.layers.moe.moe_runner.deep_gemm as _dgr

        bound = getattr(_dgr, "_legacy_silu_and_mul", None)
        if bound is not None and getattr(bound, "__module__", "") != _JIT_BODY:
            return True
        from sglang.kernels.ops.activation import silu_and_mul as _hcu_silu
    except ImportError:
        return False
    except Exception:
        log.warning("HCU adapter: silu_and_mul binding failed", exc_info=True)
        return False

    _dgr._legacy_silu_and_mul = _hcu_silu
    log.info(
        "HCU DeepGEMM bridge: _legacy_silu_and_mul bound to "
        "sglang silu_and_mul (replaced %s)", "JIT body" if bound else "None"
    )
    return True


def hcu_diffusion_platform_plugin():
    from hcu_adapter.bridges.sglang_diffusion import platform_plugin

    return platform_plugin()


def __getattr__(name):
    if name == 'HcuDiffusionPlatform':
        from hcu_adapter.bridges.sglang_diffusion import get_platform_class

        return get_platform_class()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


_FAILED_BRIDGES = []


def _install_bridge(topic, _he=None):
    """Try-isolated install of one hcu_adapter.bridges.sglang_<topic>
    module (R-12 contract: single-bridge failure never kills the chain
    of ATTEMPTS).

    Final tightening (governance 3b-8, 2026-11-19): with
    HCU_STRICT_BRIDGES=1 (default) a failed bridge ABORTS the install
    -- serving with the topic's ops silently on the upstream path is a
    degradation, and "no degradation, ever" means refusing to serve.
    Stronger and simpler than per-op first-use poison wrappers (those
    would need a target table per topic): the refusal fires at startup,
    in the tier table's face. HCU_STRICT_BRIDGES=0 keeps the audit-only
    behavior (record + ERROR + DEGRADED summary) for triage sessions.
    """
    from hcu_adapter.infra import hcu_env

    strict = str(hcu_env.resolve("HCU_STRICT_BRIDGES")[0]).strip().lower() not in (
        "0", "off", "false", "none", "")
    try:
        import importlib

        importlib.import_module(
            "hcu_adapter.bridges.sglang_%s" % topic).install(_he)
    except Exception as exc:
        import logging

        _FAILED_BRIDGES.append(topic)
        logging.getLogger("hcu.adapter").error(
            "HCU adapter: sglang_%s bridge FAILED -- the ops it serves "
            "would run on the upstream path (degradation)", topic,
            exc_info=True)
        if strict:
            raise RuntimeError(
                "sglang_%s bridge failed and HCU_STRICT_BRIDGES=1: "
                "refusing to serve degraded (governance 3b-8 -- zero "
                "runtime fallback). Fix the bridge (see the ERROR above) "
                "or set HCU_STRICT_BRIDGES=0 for a triage session." % topic
            ) from exc


def install() -> None:
    """Install the existing sglang bridge sequence without changing its order."""
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

    import logging
    import os
    import sys

    import torch

    if getattr(torch.version, "hip", None) is not None:
        # is_cuda semantics hard-depends on vllm only for `_custom_ops` symbol
        # stubs (bound elsewhere). The vllm_hcu platform plugin, however,
        # hangs the process on this stack (apply_platform_patches under our
        # import order — observed live on validation-host-26 dtk26041: "Platform plugin hcu
        # is activated" then silence until watchdog). sglang resolves its own
        # platforms via the seam below; the plugin is dead weight here.
        # Gate value must be a NON-empty allowlist (vllm envs treats an unset
        # var as None = load-all, and "" would be racy against vllm envs being
        # imported before sitecustomize); "none" is a name no real plugin has,
        # so the filter keeps nothing regardless of import order. Operator can
        # still opt in by exporting VLLM_PLUGINS explicitly (setdefault).
        os.environ.setdefault("VLLM_PLUGINS", "none")
        # Common bootstrap already logged the registry and installed the
        # topology/tvm-ffi guards. Keep the resolver handle used below.
        try:
            from hcu_adapter.infra import hcu_env as _he
        except Exception:
            _he = None
        # Optional code-object repair must precede DeepGEMM/SGLang aliases.
        try:
            from hcu_adapter.features.deep_gemm import deepgemm_cov5

            deepgemm_cov5.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: requested DeepGEMM COV5 repair not installed",
                exc_info=True,
            )
        # deep_gemm name availability (DSV4 family): sglang.srt.models.deepseek_v4
        # (and _dspark) import deep_gemm at module level, and the FIRST registry
        # scan can happen as early as the sections below importing sglang.srt.
        # Without the alias the scanner logs "Ignore import error ... No module
        # named 'deep_gemm'" and caches the modules missing (the registry
        # backfill section further down heals the cache, but the ignore line
        # stays). Alias the DTK `deepgemm` wheel into sys.modules BEFORE any
        # sglang.srt import so the first-pass scan imports them cleanly.
        try:
            import deep_gemm  # noqa: F401
        except ImportError:
            try:
                from hcu_adapter.features.deep_gemm import deep_gemm_dtk as _dgd_early

                _dgd_early.install()
            except Exception:
                logging.getLogger("hcu.adapter").warning(
                    "HCU adapter: early deep_gemm alias not installed", exc_info=True
                )
        # platform semantics seam (governance §0.5 is_cuda route) MUST be the
        # first sglang-touching step: common.py freezes FP8_E4M3_MAX from
        # is_hip() at import time and every module's `from ... import is_hip`
        # binds constants then too. Everything below imports sglang.
        try:
            from hcu_adapter.infra import platform_seam

            platform_seam.install()
            from hcu_adapter.bridges._sglang import device_visibility

            device_visibility.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: platform semantics seam not installed", exc_info=True
            )
        # Bind the parent launch context after platform selection, before
        # downstream imports can cache its scheduler launch entry.
        try:
            from hcu_adapter.bridges import sglang_spawn_device

            sglang_spawn_device.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: scheduler device bridge not installed", exc_info=True
            )
        # runtime bootstrap fixes (TP broadcast, watchdog floor, kv strides, JIT cache, dsa_topk_v2, tvm-ffi alias, JIT progress, V41 archs alias) -> hcu_adapter/bridges/sglang_bootstrap_fixes.py
        _install_bridge("bootstrap_fixes", _he)


        # serve-time gates (fa4/flashinfer registry, breakable prefill graph, QKV-A eligibility, debug probes, model env, MoE tile, MLA bmm, R-09 aiter quant/AR-norm) -> hcu_adapter/bridges/sglang_serve_gates.py
        _install_bridge("serve_gates", _he)


        # serve-time AOT requirement: launch aborts without native sgl_kernel ops (ninja-JIT fallback not tolerated; HCU_REQUIRE_NATIVE=0 downgrades) -> hcu_adapter/bridges/sglang_native_ops_gate.py
        _install_bridge("native_ops_gate", _he)


        # vendor/triton routes (libdevice redirect, PDL strip, KDA router, fa3 vendor, GDN aiter, layernorm face) -> hcu_adapter/bridges/sglang_vendor_routes.py
        _install_bridge("vendor_routes", _he)


        # fp8 quant path bridges (HCU_FP8_TRACE probe, native per-token quant bug, group-quant dtype) -> hcu_adapter/bridges/sglang_fp8_quant.py
        _install_bridge("fp8_quant", _he)


        # kernel-selector CUDA capability check HIP acceptance (AOT sgl_kernel) -> hcu_adapter/bridges/sglang_kernel_selector.py
        _install_bridge("kernel_selector", _he)


        # CT + explicit w8a8_int8 consumers GEMM import backfill -> hcu_adapter/bridges/sglang_w8a8_int8.py
        _install_bridge("w8a8_int8", _he)


        # FlashMLA construct-only stubs + sgl-flash-mla wheel face skip contract -> hcu_adapter/bridges/sglang_flash_mla.py
        _install_bridge("flash_mla", _he)


        # fa3 attention backend torch-emulation route (SGLANG_USE_SGL_FA3_KERNEL) -> hcu_adapter/bridges/sglang_fa3_backend.py
        _install_bridge("fa3_backend", _he)


        # moe_wna16 quant config non-linear layer contract restore -> hcu_adapter/bridges/sglang_moe_wna16.py
        _install_bridge("moe_wna16", _he)


        # HCU_DUMP_INT8 serve-time W8A8Int8 apply audit (torch recompute + cos/maxerr log) -> hcu_adapter/bridges/sglang_int8_audit.py
        _install_bridge("int8_audit", _he)


        # family-override modules _register_for interception (incl. deepseek_v2 DSA branch) -> hcu_adapter/bridges/sglang_family_overrides.py
        _install_bridge("family_overrides", _he)


        # DSA aiter preshuffle paged-MQA gate force-on (page/table/tile decisions only) -> hcu_adapter/bridges/sglang_aiter_preshuffle.py
        _install_bridge("aiter_preshuffle", _he)


        # DSV4/GLM swiglu-clamp MoE activation triton route (fused_experts_impl assert relax) -> hcu_adapter/bridges/sglang_swiglu_clamp.py
        _install_bridge("swiglu_clamp", _he)


        # DSV4 quant/rope family bridges (silu-fused-quant, MHC stats, sparse rope/norm, q/k norm+rope) -> hcu_adapter/bridges/sglang_dsv4_quant.py
        _install_bridge("dsv4_quant", _he)


        # HCU_LAYER_AUDIT per-module output magnitude audit (triage tooling) -> hcu_adapter/bridges/sglang_layer_audit.py
        _install_bridge("layer_audit", _he)


        # slimquant_w4a8 checkpoint auto-rule (HCU_SLIMQUANT gate) -> hcu_adapter/bridges/sglang_slimquant_guard.py
        _install_bridge("slimquant_guard", _he)


        # DSA indexer top-k sgl-kernel->torch bridge -> hcu_adapter/bridges/sglang_indexer_topk.py
        _install_bridge("indexer_topk", _he)


        # HCU store-trace + DSA topk torch-arm probe (debug capture) -> hcu_adapter/bridges/sglang_store_trace.py
        _install_bridge("store_trace", _he)


        # DSA indexer rotate_activation hadamard + prefill/decode backend validator -> hcu_adapter/bridges/sglang_dsa_guards.py
        _install_bridge("dsa_guards", _he)


        # aiter all-reduce gate (custom AR retired; auto-rule + kill-switch precedence) -> hcu_adapter/bridges/sglang_aiter_ar.py
        _install_bridge("aiter_ar", _he)


        # gfx936 MQA-logits fp8 emu (paged/ragged torch fallback auto-rule) -> hcu_adapter/bridges/sglang_mqa_emu.py
        _install_bridge("mqa_emu", _he)


        # HCU_DUMP_LAYERS per-layer debug capture (2026-10-16) -> hcu_adapter/bridges/sglang_layer_dump.py
        _install_bridge("layer_dump", _he)


        # 8. deep_ep API bridge -> hcu_adapter/bridges/sglang_deep_ep.py
        _install_bridge("deep_ep", _he)



        # 9. DeepGEMM DTK bridge (EP, 2026-10-11) -> hcu_adapter/bridges/sglang_deep_gemm.py
        _install_bridge("deep_gemm", _he)



        # 10. DTK aiter fused-MoE bridge (2026-10-11) -> hcu_adapter/bridges/sglang_moe.py
        _install_bridge("moe", _he)


        # 11. Prefill context-parallel bridge (2026-10-12) -> hcu_adapter/bridges/sglang_cp.py
        _install_bridge("cp", _he)


        # 12. GLM-5.3-Flash layer-boundary precision probe (2026-10-16) -> hcu_adapter/bridges/sglang_glm_probe.py
        _install_bridge("glm_probe", _he)


        # 13. GLM-5.3-Flash MHC hyper-connection emu (2026-10-16) -> hcu_adapter/bridges/sglang_mhc.py
        _install_bridge("mhc", _he)


        # 14. Spec-decode KV-loc safety (2026-10-17) -> hcu_adapter/bridges/sglang_spec_kv.py
        _install_bridge("spec_kv", _he)


        try:
            from hcu_adapter.bridges import sglang_spec_h2d_pin as spec_h2d_pin

            spec_h2d_pin.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: speculative pinned metadata not installed", exc_info=True,
            )

        # Native routing needs stable expert slots before its weight sum.
        # Keep the compiled entry and cached aliases; never rewrite source.
        try:
            from hcu_adapter import moe_topk_order

            moe_topk_order.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: native grouped TopK order bridge not installed",
                exc_info=True,
            )

        # Install the model-method binding only after operator/vendor bridges
        # are ready. This imports the community class and retains existing
        # aliases of forward; no early model scan or disk patch is needed.
        try:
            from hcu_adapter import moe_dualstream_gate

            moe_dualstream_gate.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: dual-stream MoE capture gate not installed",
                exc_info=True,
            )

        try:
            from hcu_adapter.bridges import sglang_multimem_gate as multimem_gate

            multimem_gate.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: multimem probe gate not installed", exc_info=True,
            )

        try:
            from hcu_adapter import qsa_pool

            qsa_pool.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: QSA pool not installed", exc_info=True
            )

        try:
            from hcu_adapter.bridges._sglang import grouped_topk, sampling

            grouped_topk.install()
            sampling.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: grouped_topk/sampling bridge not installed", exc_info=True
            )

        # P7 (governance 3b-8, 2026-11-19): the tier resolution table is
        # the ONE thing serve startup prints loudly -- one line per op
        # (op -> impl [role/mechanism] via target) plus the catalog ops
        # still on the upstream path. Performance triage starts here:
        # diff this table between the fast day and the slow day.
        try:
            from hcu_adapter.infra import registry

            registry.log_summary()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: tier table print failed", exc_info=True
            )
        if _FAILED_BRIDGES:
            logging.getLogger("hcu.adapter").error(
                "HCU adapter: DEGRADED -- %d bridge(s) failed: %s; the "
                "ops they serve are on the upstream path (zero-fallback "
                "audit: this line is a degradation, not a fallback)",
                len(_FAILED_BRIDGES), ", ".join(_FAILED_BRIDGES),
            )
