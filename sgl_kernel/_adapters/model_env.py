"""Model-aware platform env defaults (R-08 B-tier).

Some upstream sglang envs are per-model escape hatches for missing HIP arms
of fused kernels. They used to be hand-exported in every recipe (cookbook
triage: 18 vars across launch/test scripts); this module derives them from
the model config at ModelRunner init so a correct deployment needs zero
envs, while an explicit export always wins (setdefault only).

Landed here because the read sites are late-bound (``envs.X.get()`` at
backend init / pool build time — dsa_backend.py:484, dsa_indexer.py:277,
mamba1.py cache-dtype path), so setting them just before ModelRunner's own
init races nothing.

Rules (each logged once at apply time):
* has_indexer (hf config exposes index_kpool / indexer-head keys, e.g.
  GLM-5.x DSA, DeepSeek V3.2):
  - SGLANG_DSA_FUSE_TOPK=0        — fused topk transform has no HIP arm
  - SGLANG_DISABLE_DSA_INDEXER_FUSION=1 — rope_first/indexer fusion has no
    HIP arm (upstream's own escape hatch, default False)
* fp16 hybrid/GDN bottom model (dtype float16 + mamba/linear-attn keys in
  hf config, e.g. Qwen3.5 GDN fp16 checkpoints):
  - SGLANG_MAMBA_CONV_DTYPE=float16 — upstream conv pool hardcodes bf16
    regardless of model dtype; first forward of an fp16 model crashes
    without this (cookbook note, 10-12).

Deliberately NOT model-automated (triage verdicts, docs/env-vars.md §R-08):
SGLANG_DEEPEP_BF16_DISPATCH (workflow tier — the CLI flag
--deepep-dispatcher-output-dtype bf16 is the canonical spelling);
HCU_FA_IMPL / HCU_LINEAR_ATTN_GDN (already default ``auto`` = adaptive).

Gated by env HCU_MODEL_ENV (default on, =0 opts out).
"""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger(__name__)

_TARGET_MODULE = "sglang.srt.model_executor.model_runner"
_APPLIED = False

_DSA_HINT_KEYS = ("index_kpool", "indexer_n_heads", "index_head_dim")
_GDN_HINT_KEYS = ("mamba_num_heads", "mamba_head_dim", "linear_num_value_heads")


def _hf_config_like(model_config):
    """ModelConfig in sglang carries .hf_config (transformers config obj)."""
    for attr in ("hf_config", "config"):
        cfg = getattr(model_config, attr, None)
        if cfg is not None:
            return cfg
    return model_config  # tolerate a raw config object in tests


def _has_indexer(hf) -> bool:
    return any(getattr(hf, k, None) for k in _DSA_HINT_KEYS)


def _is_fp16_gdn(model_config, hf) -> bool:
    dtype = getattr(model_config, "dtype", None)
    dtype = getattr(dtype, "value", dtype)  # torch.dtype | str enum
    name = str(getattr(dtype, "name", dtype)).lower()
    if "bfloat16" in name or "bf16" in name:  # "float16" is a substring of
        return False  # "bfloat16" — exclude the bf16 family explicitly
    return (
        "float16" in name or "fp16" in name or "half" in name
    ) and any(getattr(hf, k, None) for k in _GDN_HINT_KEYS)


def apply_model_env(model_config) -> list[str]:
    """setdefault the per-model envs; returns the lines that were applied
    (empty list = nothing to do / everything already explicit)."""
    hf = _hf_config_like(model_config)
    applied = []
    if _has_indexer(hf):
        for k, v in (
            ("SGLANG_DSA_FUSE_TOPK", "0"),
            ("SGLANG_DISABLE_DSA_INDEXER_FUSION", "1"),
        ):
            if k not in os.environ:
                os.environ[k] = v
                applied.append(f"{k}={v}")
    if _is_fp16_gdn(model_config, hf):
        if "SGLANG_MAMBA_CONV_DTYPE" not in os.environ:
            os.environ["SGLANG_MAMBA_CONV_DTYPE"] = "float16"
            applied.append("SGLANG_MAMBA_CONV_DTYPE=float16")
    for line in applied:
        logger.info("HCU adapter: model-env default %s (model-derived)", line)
    return applied


def install() -> None:
    """Patch ModelRunner.__init__ (first arg is model_config) via the
    importlib.import_module wrapper — same lazy pattern as state_snapshot."""
    from . import hcu_env, registry

    mode, _src = hcu_env.resolve("HCU_MODEL_ENV")
    if str(mode).lower() in ("off", "0", "none", "false", ""):
        logger.info(
            "HCU adapter: model-env defaults off (%s [%s])", mode, _src
        )
        return
    registry.register(
        "model_env",
        "auto",
        target="sglang.srt.model_executor.model_runner:ModelRunner.__init__",
        mechanism=registry.MECH_SEAM,
        detail=f"mode={mode} (DSA fusion off / fp16-GDN conv dtype)",
    )

    orig_import = importlib.import_module
    patched = [False]

    def _import(name, *a, **k):
        mod = orig_import(name, *a, **k)
        global _APPLIED
        if name == _TARGET_MODULE and not patched[0]:
            patched[0] = True
            runner_cls = getattr(mod, "ModelRunner", None)
            if runner_cls is not None and not getattr(
                runner_cls, "_hcu_model_env_patched", False
            ):
                runner_cls._hcu_model_env_patched = True
                orig_init = runner_cls.__init__

                def _init(self, *args, **kwargs):
                    cfg = args[0] if args else kwargs.get("model_config")
                    try:
                        if cfg is not None:
                            apply_model_env(cfg)
                    except Exception:
                        logger.warning(
                            "hcu model-env: derive failed", exc_info=True
                        )
                    return orig_init(self, *args, **kwargs)

                runner_cls.__init__ = _init
                logger.info(
                    "HCU adapter: model-env defaults armed on ModelRunner"
                    " (DSA fusion off / fp16-GDN conv dtype; HCU_MODEL_ENV=0"
                    " opts out)"
                )
        return mod

    importlib.import_module = _import
