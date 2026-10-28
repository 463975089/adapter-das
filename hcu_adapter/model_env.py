"""Model-aware platform env defaults (R-08 B-tier).

Some upstream sglang envs are per-model escape hatches for missing HIP arms
of fused kernels. They used to be hand-exported in every recipe (cookbook
triage: 18 vars across launch/test scripts); this module derives them from
the model config so a correct deployment needs zero envs, while an explicit
export always wins (setdefault only).

Two arms, because the 2026-10-18 zero-env serving gate caught the original
single-arm design never firing in real launches:

* ``apply_from_argv`` (primary) — sitecustomize time, i.e. before ANY sglang
  import: sniff ``--model-path`` from argv, read the checkpoint's
  config.json, setdefault. Necessary because serving reads some envs during
  static imports / attention-backend init (SGLANG_DSA_FUSE_TOPK at
  dsa_backend init decides fused-vs-torch topk; a late write leaves fused
  on and boot dies with "Unsupported dsa_topk_backend ... TORCH").
* ModelRunner.__init__ hook (backstop) — covers API-style launches with no
  argv; the original importlib.import_module wrapper idea silently missed
  the launch path (static ``import`` statements never call it).

Rules (each logged once at apply time):
* has_indexer (config exposes index_kpool / indexer-head keys anywhere in
  the (possibly composite) config — GLM-5.3-Flash keeps them under
  text_config; DeepSeek V3.2 at top level):
  - SGLANG_DSA_FUSE_TOPK=0        — fused topk transform has no HIP arm
  - SGLANG_DISABLE_DSA_INDEXER_FUSION=1 — rope_first/indexer fusion has no
    HIP arm (upstream's own escape hatch, default False)
* fp16 hybrid/GDN bottom model (dtype float16 + mamba/linear-attn keys in
  the same config section, e.g. Qwen3.5 GDN fp16 checkpoints):
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
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

_TARGET_MODULE = "sglang.srt.model_executor.model_runner"
_APPLIED = False

_DSA_HINT_KEYS = ("index_kpool", "indexer_n_heads", "index_head_dim")
_GDN_HINT_KEYS = ("mamba_num_heads", "mamba_head_dim", "linear_num_value_heads")
_DTYPE_KEYS = ("torch_dtype", "dtype")


def _hf_config_like(model_config):
    """ModelConfig in sglang carries .hf_config (transformers config obj)."""
    for attr in ("hf_config", "config"):
        cfg = getattr(model_config, attr, None)
        if cfg is not None:
            return cfg
    return model_config  # tolerate a raw config object in tests


def _is_fp16_name(name: str) -> bool:
    n = str(name).lower()
    if "bfloat16" in n or "bf16" in n:  # "float16" is a substring of
        return False  # "bfloat16" — exclude the bf16 family explicitly
    return "float16" in n or "fp16" in n or "half" in n


def _has_indexer(hf) -> bool:
    return any(getattr(hf, k, None) for k in _DSA_HINT_KEYS)


def _is_fp16_gdn(model_config, hf) -> bool:
    dtype = getattr(model_config, "dtype", None)
    dtype = getattr(dtype, "value", dtype)  # torch.dtype | str enum
    name = str(getattr(dtype, "name", dtype))
    return _is_fp16_name(name) and any(
        getattr(hf, k, None) for k in _GDN_HINT_KEYS
    )


def _log_applied(lines, source):
    for line in lines:
        logger.warning(
            "HCU adapter: model-env default %s (model-derived, %s)", line, source
        )


def _setdefault(pairs):
    applied = []
    for k, v in pairs:
        if k not in os.environ:
            os.environ[k] = v
            applied.append(f"{k}={v}")
    return applied


def apply_model_env(model_config) -> list[str]:
    """setdefault the per-model envs; returns the lines that were applied
    (empty list = nothing to do / everything already explicit)."""
    hf = _hf_config_like(model_config)
    applied = []
    if _has_indexer(hf):
        applied += _setdefault(
            (
                ("SGLANG_DSA_FUSE_TOPK", "0"),
                ("SGLANG_DISABLE_DSA_INDEXER_FUSION", "1"),
            )
        )
    if _is_fp16_gdn(model_config, hf):
        applied += _setdefault((("SGLANG_MAMBA_CONV_DTYPE", "float16"),))
    for line in applied:
        logger.info("HCU adapter: model-env default %s (model-derived)", line)
    return applied


def _walk_sections(cfg: dict, depth: int = 0):
    """Yield cfg plus nested dict sections — composite HF configs keep the
    language-model keys under text_config (glm5_next et al.)."""
    if not isinstance(cfg, dict) or depth > 3:
        return
    yield cfg
    for v in cfg.values():
        if isinstance(v, dict):
            yield from _walk_sections(v, depth + 1)


def _model_path_from_argv(argv=None) -> str | None:
    argv = list(sys.argv if argv is None else argv)
    for i, a in enumerate(argv):
        if a == "--model-path" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--model-path="):
            return a.split("=", 1)[1]
    return None


def apply_from_argv(argv=None) -> list[str]:
    """Primary arm: derive the same defaults at sitecustomize time from
    --model-path + config.json, before any sglang import can snapshot them."""
    path = _model_path_from_argv(argv)
    if not path:
        return []
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(
            "HCU adapter: model-env argv sniff skipped (%s: %s)", path, e
        )
        return []
    applied = []
    sections = list(_walk_sections(cfg))
    if any(any(sec.get(k) for k in _DSA_HINT_KEYS) for sec in sections):
        applied += _setdefault(
            (
                ("SGLANG_DSA_FUSE_TOPK", "0"),
                ("SGLANG_DISABLE_DSA_INDEXER_FUSION", "1"),
            )
        )
    for sec in sections:
        if not any(sec.get(k) for k in _GDN_HINT_KEYS):
            continue
        if any(_is_fp16_name(sec[k]) for k in _DTYPE_KEYS if sec.get(k)):
            applied += _setdefault((("SGLANG_MAMBA_CONV_DTYPE", "float16"),))
            break
    _log_applied(applied, f"argv sniff {path}")
    return applied


def install() -> None:
    """sitecustomize entry: argv sniff first (pre-import), then arm the
    ModelRunner hook as an API-launch backstop."""
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
        detail="mode={} (argv sniff at boot + ModelRunner backstop; DSA"
        " fusion off / fp16-GDN conv dtype)".format(mode),
    )

    try:
        apply_from_argv()
    except Exception:
        logger.warning("hcu model-env: argv sniff failed", exc_info=True)

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
                    "HCU adapter: model-env ModelRunner backstop armed"
                    " (HCU_MODEL_ENV=0 opts out)"
                )
        return mod

    importlib.import_module = _import

