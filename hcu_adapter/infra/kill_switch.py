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

from hcu_adapter.infra import hcu_env as hcu_environment

LOG = logging.getLogger("hcu.adapter")

# canonical name -> (what gets blocked, opt-in env that re-enables it)
SWITCHES = {
    "vllm_hisparse": ("Explicit Core033 HiSparse provider; startup refusal without fallback", None),
    "hy4_mooncake_grouping_reuse": ("Plugin invocation-local Mooncake grouping; native builder fallback", None),
    "hy4_prefill_workspace": ("Plugin PCP KV workspace coalescing; startup-only native builder fallback", None),

    "vllm_paged_mqa_logits": ("Explicit non-MX vLLM/plugin paged MQA provider; startup refusal", None),
    "nextn_idle_metadata": ("Missing NEXTN idle KV metadata default (startup only)", None),
    "prefill_graph_driver": ("Breakable graph HIP stream query repair (startup only)", None),
    "vllm_inverse_rope": ("Plugin inverse RoPE; fresh-process stock fallback", None),
    "vendor_fa": ("DTK FlashAttention; FA3 bridge falls back to Triton, explicit diffusion FA2 is refused", "HCU_FA_IMPL"),
    "vllm_flashmla_decode": ("vLLM V4.1 native packed sparse decode; fresh-process stock fallback", None),
    "vllm_flashmla_prefill": ("vLLM V4.1 native sparse prefill; fresh-process stock fallback", None),
    "vllm_fp4_indexer": ("V41 FP4 model/cache/producer/reader; fresh-process FP8 rollback", "HCU_VLLM_FP4_INDEXER_IMPL"),
    "vllm_fp4_packer": ("Plugin E2M1 lowering; fresh-process original-binding rollback", "HCU_VLLM_FP4_PACKER_IMPL"),
    "vllm_scaled_mm": ("vLLM LightOp FP8/INT8 execution bridge; startup Triton fallback", None),
    "w4a8_contiguous_gemm": ("Optional DTK W4A8 HIPC COV5 ABI repair (startup only)", None),
    "vllm_dsv41_channel_fp8": ("vLLM V4.1 channel-FP8 config registration; fresh-process stock fallback", "HCU_VLLM_DSV41_CHANNEL_FP8_IMPL"),
    "vllm_ct_w4a8": ("vLLM V4.1 CT W4A8 config registration; fresh-process stock fallback", "HCU_VLLM_CT_W4A8_IMPL"),
    "vllm_pcp_q_rope": ("V4.1 PCP prefill Q-only RoPE; fresh-process stock fallback", None),
    "vllm_pcp": ("vLLM V4.1 PCP attention ownership; fresh-process stock fallback", "HCU_VLLM_PCP_IMPL"),

    "gdn_verify": ("SGLang GDN AITER verify route (startup-only upstream fallback)",
                   "HCU_LINEAR_ATTN_GDN_VERIFY"),
    "mamba_state_scatter": (
        "SGLang flattened Mamba state scatter (startup-only upstream fallback)",
        None,
    ),

    "qsa_mqa": ("QSA MQA BoltOPs dispatch (falls back to Torch reference)", None),
    "vllm_minimax_prefill": ("MiniMax Plugin prefill fusion; startup veto retains the stock composition", None),
    "vllm_minimax_qknorm_abi": ("vLLM MiniMax fused native ABI bridge; disabling rejects old ABI", None),
    "vllm_sparse_indexer": ("vLLM C4 indexer public OOT compatibility; fresh-process stock fallback", "HCU_VLLM_SPARSE_INDEXER_IMPL"),
    "vllm_mhc": ("vLLM official Torch/AITER mHC compatibility registration (startup only)", "HCU_VLLM_MHC_IMPL"),
    "deep_ep": ("DeepInfer DeepEP compatibility surface", None),
    "sglang_tree_sampling": ("SGLang target-only stochastic tree sampling; disabled calls fail explicitly", None),
    "sglang_grouped_topk": ("SGLang grouped router; ops.sglang_grouped_topk=torch restores current dispatch", None),
    "int8_scaled_mm": ("SGLang INT8 dense GEMM provider; ops.int8_scaled_mm=triton restores reference route", None),
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
