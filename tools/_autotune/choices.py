"""Per-op tier judge map + ladder-descent semantics for --autotune.

This module is the tool's OPINION layer: for every catalog op it declares

* which backends are measurable candidates (state gate comes from the
  catalog itself, not duplicated here);
* the correctness judge per backend -- an existing repo test, run either
  as a pytest file or standalone (`mode: std`), with optional env the test
  needs to actually route to the backend under judgment;
* the ladder order (index in LADDER) used to descend when a tier fails.

An op absent from OPS keeps its catalog default and is reported as
"no-judge fallback" -- honest downgrade, never a fake green. Adding
coverage = adding entries here; nothing else changes.

Judge env values are TEMPORARY overrides scoped to the child pytest
process only (explicit-env wins the adapter's裁决链), so autotune never
mutates the serving configuration it is measuring.
"""

import os

# ladder by op: ordered preference when measured evidence is absent for
# the top pick, or when a tier fails its judges. Highest performance
# expectation first, emu/torch safety net last. Tiers NOT in a ladder are
# never auto-selected (they stay reachable via hand-written config).
_DEFAULT_LADDER = ["emu", "triton", "vendor"]

OPS = {
    # ------------------------------------------------------------ silu/act
    "silu_and_mul": {
        "ladder": ["lightop", "triton", "torch"],
        "judges": {
            "triton": {"tests": ["tests/kernels/test_elementwise.py"],
                       "mode": "pytest"},
            "lightop": {"tests": ["tests/kernels/test_elementwise.py"],
                        "mode": "pytest",
                        "env": {"HCU_SILU_MUL_IMPL": "lightop"}},
            "torch": {"tests": ["tests/kernels/test_elementwise.py"],
                      "mode": "pytest",
                      "env": {"HCU_SILU_MUL_IMPL": "torch"}},
        },
        "bench": "act_and_mul",
    },
    # ------------------------------------------------------------- rmsnorm
    "rms_norm": {
        "ladder": ["aiter", "triton", "torch"],
        "judges": {
            "triton": {"tests": ["tests/kernels/test_layernorm.py"],
                       "mode": "pytest"},
            "aiter": {"tests": ["tests/kernels/test_layernorm.py"],
                      "mode": "pytest",
                      "env": {"HCU_RMS_NORM_IMPL": "aiter"}},
        },
        "bench": "rms_norm",
    },
    # ---------------------------------------------------------------- rope
    "rotary_embedding": {
        "ladder": ["triton"],
        "judges": {
            "triton": {"tests": ["tests/kernels/test_rope.py"],
                       "mode": "pytest"},
        },
        "bench": "rope",
    },
    # ------------------------------------------------------------ moe gate
    "moe_gate": {
        "ladder": ["triton"],
        "judges": {
            "triton": {"tests": ["tests/kernels/test_moe_gate.py"],
                       "mode": "pytest"},
        },
        "bench": "moe_gate",
    },
    # ---------------------------------------------------------- quant fp8
    "per_token_quant": {
        "ladder": ["aiter", "lightop", "torch"],
        "judges": {
            "aiter": {"tests": ["tests/kernels/test_quant_aiter_gfx938.py"],
                      "mode": "pytest",
                      "env": {"HCU_QUANT_IMPL": "aiter"}},
            "torch": {"tests": ["tests/kernels/test_quant_aiter_gfx938.py"],
                      "mode": "pytest",   # loud-fail sentinel: stock must raise
                      "expect_fail": True},
        },
        "bench": "per_token_group_quant",
    },
    # ------------------------------------------------------- paged MQA DSA
    "paged_mqa_logits": {
        "ladder": ["triton", "emu"],
        "judges": {
            "triton": {"tests": ["tests/kernels/test_mqa_triton_gfx938.py"],
                       "mode": "std"},   # standalone: exit code is the judge
            "emu": {"tests": ["tests/kernels/test_mqa_triton_gfx938.py"],
                    "mode": "std",
                    "env": {"HCU_MQA_LOGITS_IMPL": "emu"}},
        },
        "bench": "paged_mqa",
    },
    # ------------------------------------------------------- flash attention
    "flash_attention": {
        "ladder": ["triton", "emu"],
        "judges": {
            "emu": {"tests": ["tests/kernels/test_flash_attn3.py"],
                    "mode": "pytest"},
        },
        "bench": "fa3",
    },
    # ------------------------------------------------------------- conv1d
    "conv1d": {
        "ladder": ["fork", "emu"],
        "judges": {
            "fork": {"tests": ["tests/kernels/test_conv1d_fn_varlen.py"],
                     "mode": "pytest"},
            "emu": {"tests": ["tests/kernels/test_conv1d_fn_varlen.py"],
                    "mode": "pytest",
                    "env": {"HCU_CONV1D_IMPL": "stock"}},
        },
        "bench": None,   # conv shapes too varlen-specific for a microbench v1
    },
    # ---------------------------------------------------------- kda prefill
    "kda_prefill": {
        "ladder": ["fork", "emu"],
        "judges": {
            "fork": {"tests": ["tests/kernels/test_kda_chunk_gfx938.py"],
                     "mode": "pytest"},
        },
        "bench": None,
    },
    # ---------------------------------------------------------- GDN linear
    "linear_attn_gdn": {
        "ladder": ["aiter", "triton"],
        "judges": {
            "aiter": {"tests": ["tests/kernels/test_fla_hcu_fork_gfx938.py"],
                      "mode": "pytest",
                      "env": {"HCU_LINEAR_ATTN_GDN": "aiter"}},
            "triton": {"tests": ["tests/kernels/test_fla_hcu_fork_gfx938.py"],
                       "mode": "pytest",
                       "env": {"HCU_LINEAR_ATTN_GDN": "triton"}},
        },
        "bench": None,   # _bench_gdn_aiter needs dual-rig wiring, v2
    },
}

# ops never measured (platform/env semantics, not compute tiers): keep
# catalog default, zero ceremony.
SKIP_OPS = {
    "platform_detect", "model_env", "kv_pool_strides", "dist_broadcast",
    "triton_pdl", "triton_libdevice", "allreduce", "sampling",
    "fused_moe", "w4a16_moe",
    # fused_moe/w4a16: bench wiring needs tuned-csv + deepgemm shapes; v2
    # allreduce: needs 2-rig world for a meaningful judge; v2
}


def tier_order(op):
    return OPS.get(op, {}).get("ladder", [])


def judge_for(op, backend):
    return OPS.get(op, {}).get("judges", {}).get(backend)


def bench_kind(op):
    return OPS.get(op, {}).get("bench")


def measured_ops():
    return [op for op in hcu_catalog_ops() if op in OPS and op not in SKIP_OPS]


def hcu_catalog_ops():
    from hcu_adapter import hcu_config

    return list(hcu_config.BACKEND_CATALOG.keys())
