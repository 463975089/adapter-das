# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""runtime bootstrap fixes (TP broadcast, watchdog floor, kv strides, JIT cache, dsa_topk_v2, tvm-ffi alias, JIT progress, V41 archs alias)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging


def install(_he=None) -> None:
    # TP init-broadcast payload fix (torch2.11 dlpack readonly BufferError);
    # runtime binding retains imported aliases without source writes;
    # needs sglang.srt.utils.common importable -- after the seam above.
    try:
        from hcu_adapter.infra import dist_bc

        dist_bc.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: dist broadcast bridge not installed", exc_info=True
        )
    # cold-JIT watchdog floor (box16 cold-boot fatality, 2026-10-25): cold
    # caches + stock 300s watchdog = SIGQUIT mid compile storm. Wraps the
    # watchdog factory BEFORE scheduler.py from-imports it; explicit
    # --watchdog-timeout always wins (watchdog_cold.py).
    try:
        from hcu_adapter.infra import watchdog_cold

        watchdog_cold.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: cold-JIT watchdog floor not installed", exc_info=True
        )
    # KV-pool stride tensor fix (torch2.11 numpy-scalar dtype infer); text
    # patch per governance §3 tier-3, allowlist entry "kv_pool_strides".
    try:
        from hcu_adapter.infra import kv_fix

        kv_fix.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: kv stride text patch not installed", exc_info=True
        )
    # Compile adapted csrc/include assets from an immutable adapter cache.
    # Community sources remain untouched; cached function aliases are retained.
    try:
        from hcu_adapter.features.jit import jit_sources

        jit_sources.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: immutable JIT compiler inputs not installed", exc_info=True
        )
    try:
        from hcu_adapter.bridges._sglang import jit_eh_norm

        jit_eh_norm.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: JIT EH norm launch bounds not installed", exc_info=True
        )
    try:
        from hcu_adapter.features.jit import small_copy_jit

        small_copy_jit.install()
    except ImportError:
        logging.getLogger("hcu.adapter").info(
            "HCU adapter: upstream small_copy is unavailable"
        )
    except Exception:
        # Vendor-ABI refusals raise TypeError/RuntimeError before any
        # registration (small_copy_jit.install); degrade to a warning so
        # the bootstrap fixes below still install, matching the jit_eh_norm
        # / dsa_topk_v2 step pattern. try_small_copy probes at call time,
        # so running without the binding is safe.
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: small_copy JIT fix not installed", exc_info=True
        )
    # DSA top-k v2 fusion (dsv41f decode): stride seam on the upstream JIT
    # topk_v2 kernel + the v41 full-topk fused decode arm. Replaces the
    # torch.topk fallbacks behind 57 mbtopk kernels/step (2026-10-22 trace).
    try:
        from hcu_adapter.features.dsa import dsa_topk_v2

        dsa_topk_v2.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: dsa topk v2 fusion not installed", exc_info=True
        )
    # tvm-ffi fp8 dtype aliasing for the DSV4 JIT MoE activation kernels
    # (silu_and_mul_{masked,contig}_post_quant): the kernels declare the
    # fp8 output as uint8; CUDA torch fp8 tensors present as uint8 through
    # DLPack, DTK ones present as float8_e4m3fn and the type match rejects
    # them. View fp8 args as uint8 before module.run (byte-identical; the
    # kernel writes raw fp8 patterns into the same storage).
    try:
        from hcu_adapter.features.jit import jit_fp8_ffi

        jit_fp8_ffi.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: jit fp8 ffi alias not installed", exc_info=True
        )
    # JIT build progress reporter: one stderr/journal line per real ninja
    # compile (compiling/built/FAILED) so a cold 35-module JIT storm is
    # tellable from a wedge in gate logs; the gate heartbeat tails the
    # journal (ci/l1_gpu_gate.sh run_seg). Observability only -- a missed
    # install costs a log line, never numerics.
    try:
        from hcu_adapter.features.jit import jit_build_log

        jit_build_log.install()
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: jit build progress reporter not installed",
            exc_info=True,
        )
    # DeepSeek-V4.1-Flash checkpoints carry architectures
    # ["DeepseekV41ForCausalLM"], a name no sglang tree registers (checked
    # v0.5.22 and upstream main); the V4 class serves V41 configs via its
    # model_type == "deepseek_v41" gates. Insert the alias directly into
    # ModelRegistry.models -- deliberately NOT via get_supported_archs(),
    # which would populate the registry cache before the DeepGEMM DTK
    # bridge (section 9) makes `deep_gemm` importable; deepseek_v4 would
    # then be cached-missing and even the V4 arch itself stops resolving.
    # A dict insert survives the later `import_model_classes(...) | self.models`.
    try:
        import sglang.srt.models.registry as _dreg

        if "DeepseekV41ForCausalLM" not in _dreg.ModelRegistry.models:
            try:
                import deep_gemm  # noqa: F401
            except ImportError:
                # deepseek_v4 imports deep_gemm at module level; the DTK
                # bridge (section 9) has not run yet at this point.
                from hcu_adapter.features.deep_gemm import deep_gemm_dtk as _dgdb

                _dgdb.install()
            import sglang.srt.models.deepseek_v4 as _dsv4

            _dreg.ModelRegistry.models.setdefault(
                "DeepseekV4ForCausalLM", _dsv4.DeepseekV4ForCausalLM
            )
            _dreg.ModelRegistry.models["DeepseekV41ForCausalLM"] = (
                _dsv4.DeepseekV4ForCausalLM
            )
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: DeepseekV41ForCausalLM -> DeepseekV4ForCausalLM "
                "registry alias installed (checkpoint arch ahead of upstream)"
            )
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: deepseek_v41 registry alias not installed",
            exc_info=True,
        )
