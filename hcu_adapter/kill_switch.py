# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Operator kill switches for fast precision triage (HCU adapter).

High-risk vendor ops (custom all-reduce, flash attention, ...) fail in ways
that depend on message size / seqlen / graph capture, so "it worked in the
vendor's unit test" does not transfer. When a run shows precision symptoms
(garbage decode, divergence vs eager, wrong top-1 tokens), the fastest
localization is: disable the suspect op via one env var, rerun, and see
whether the symptom disappears — then run the matching stress test in
tests/vendor/ (test_allreduce.py / test_flash_attn.py).

Usage
-----
    export HCU_DISABLE_OPS=aiter_ar          # one op
    export HCU_DISABLE_OPS=aiter_ar,aiter_fa # several, comma/space separated

Registered names (see sitecustomize.py + hcu_adapter/aiter/__init__.py):
    aiter_ar   AiterCustomAllreduce (sglang dispatch refused -> NCCL fallback)
    aiter_fa   aiter flash-attention entry points (call-time raise)

Precedence: HCU_DISABLE_OPS > arch auto-rules (e.g. gfx93* auto-refuses
aiter_ar because CUDA-graph replay corrupts data) > per-op opt-in envs
(e.g. HCU_AITER_AR=1). An explicit disable always wins.
"""

import logging

from hcu_adapter import hcu_env as hcu_environment

LOG = logging.getLogger("hcu.adapter")

# canonical name -> (what gets blocked, opt-in env that re-enables it)
SWITCHES = {
    "minimax_h3_rotary_embedding": ("H3 candidate RoPE to torch reference", None),
    "w4a8_contiguous_gemm": ("Optional DTK W4A8 HIPC COV5 ABI repair (startup only)", None),
    "vllm_sparse_indexer": ("vLLM C4 indexer public OOT compatibility; fresh-process stock fallback", "HCU_VLLM_SPARSE_INDEXER_IMPL"),
    "vllm_mhc": ("vLLM official Torch mHC compatibility registration (startup only)", None),    "deep_ep": ("DeepInfer DeepEP compatibility surface", None),
    "worker.runtime": ("DeepInfer worker lifecycle adaptation", None),
    "deep_gemm": ("DeepInfer GEMM/MQA/mHC compatibility surface", None),
    "flash_attn.cute.interface._flash_attn_fwd": ("DeepInfer ViT attention", None),
    "aiter_ar": (
        "AiterCustomAllreduce dispatch (group falls back to NCCL)",
        "HCU_AITER_AR",
    ),
    "aiter_fa": (
        "aiter flash-attention entry points (use --attention-backend triton)",
        None,
    ),
}

_SEP = str.maketrans(",; ", ",,,")


def disabled_ops():
    """Names disabled via HCU_DISABLE_OPS (frozenset, lowercase)."""
    raw = hcu_environment.resolve("HCU_DISABLE_OPS")[0]
    return frozenset(
        s.strip().lower() for s in raw.translate(_SEP).split(",") if s.strip()
    )


def op_disabled(name):
    """True when `name` is explicitly disabled via HCU_DISABLE_OPS."""
    return name.lower() in disabled_ops()


def opt_in(name):
    """True when the op's per-opt-in env is set to 1 (arch auto-rules only)."""
    env = SWITCHES.get(name, (None, None))[1]
    if not env:
        return False
    value, source = hcu_environment.resolve(env)
    return source == "explicit" and value == "1"


def guard(fn, name, detail=""):
    """Wrap `fn` so calls raise while op `name` is disabled; passthrough else.

    Used by the meta-path overlay to guard REAL vendor-wheel attributes
    (missing ones already raise via _base.Missing).
    """
    if not op_disabled(name):
        return fn

    what, _ = SWITCHES.get(name, (name, None))

    def _refused(*args, **kwargs):
        raise NotImplementedError(
            f"HCU adapter: op '{name}' ({what}) disabled via"
            f" HCU_DISABLE_OPS for triage{'; ' + detail if detail else ''}."
            f" Remove it from HCU_DISABLE_OPS to re-enable."
        )

    return _refused


def log_state():
    """One startup line summarizing explicit disables (for debuggability)."""
    off = sorted(disabled_ops())
    if off:
        unknown = [n for n in off if n not in SWITCHES]
        if unknown:
            LOG.warning(
                "HCU adapter: HCU_DISABLE_OPS has unknown names %s"
                " (registered: %s)", unknown, sorted(SWITCHES)
            )
        LOG.warning(
            "HCU adapter: kill switches ACTIVE: %s", ", ".join(off)
        )
