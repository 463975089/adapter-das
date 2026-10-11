"""Config-file backend selection for the HCU adapter (deployment tier).

The adapter exposes several operators from more than one source (our
triton sgl_kernel, the DTK vendor wheels aiter / lightop, and plain
torch). Which backend serves each operator is a DEPLOYMENT decision and
belongs in a config file -- not in environment variables (user directive
2026-10-12). Env vars remain as a DEBUG-tier override only: an explicitly
set env var still wins over the file, but a correct deployment needs
zero env vars.

Schema (hcu_adapter.yaml, first hit wins in discovery order):

    version: 1
    ops:
      silu_and_mul: triton      # triton | lightop | torch
      rms_norm: aiter           # triton | aiter | torch
      allreduce: custom         # custom | aiter | nccl
      paged_mqa_logits: emu     # emu (aiter/lightop planned)

Discovery: $HCU_CONFIG > ./hcu_adapter.{yaml,yml,json} >
<adapter root>/hcu_adapter.{yaml,yml,json}. No file -> pure legacy
behavior (env > auto > default); the file is optional but once present
it must be valid: unknown ops, unknown backends, or selecting a backend
whose status is "planned" fail LOUDLY at load (typo guard -- a silent
fallback to defaults would hide exactly the mistake the user made).

BACKEND_CATALOG is the single source of truth for what can be selected;
the interactive wizard (tools/hcu_config_wizard.py) and the future web
UI both render from it. status:
    production  default, fully gated
    validated   passed all gates, opt-in (or per-wheel auto rule)
    candidate   wired, awaiting its gate matrix
    planned     catalogued for the future UI, NOT wired -- selecting
                raises at config load
"""

from __future__ import annotations

import json
import os
import sys


class HcuConfigError(RuntimeError):
    """Raised for any invalid config file; message is user-facing."""


# -- catalog ----------------------------------------------------------------
# var        registry variable the op maps onto (None = no runtime switch yet)
# vars       compound ops: backend -> {registry var: value}
# backends   selectable backend -> {status, note}

BACKEND_CATALOG = {
    "silu_and_mul": {
        "var": "HCU_SILU_MUL_IMPL",
        "summary": "gated activation out = silu(x[:,:d]) * x[:,d:]",
        "backends": {
            "triton": {"status": "production",
                       "note": "default (auto); fp32 accumulate"},
            "lightop": {"status": "validated",
                        "note": "lightop.fuse_silu_and_mul "
                                "(input, output, mask_m, expect_m)"},
            "torch": {"status": "production",
                      "note": "reference path, eager debug"},
        },
    },
    "rms_norm": {
        "var": "HCU_RMS_NORM_IMPL",
        "summary": "RMSNorm; fused_add_rmsnorm/gemma variants stay triton",
        "backends": {
            "triton": {"status": "production", "note": "default"},
            "aiter": {"status": "candidate",
                      "note": "aiter.rmsnorm_forward(x,w,eps,training) "
                              "-> (out, ..); plain rmsnorm only"},
            "torch": {"status": "production",
                      "note": "reference path"},
        },
    },
    "allreduce": {
        "vars": {
            "custom": {"HCU_TRITON_AR": "1"},
            "aiter": {"HCU_TRITON_AR": "0", "HCU_AITER_AR": "1"},
            "nccl": {"HCU_TRITON_AR": "0", "HCU_AITER_AR": "0"},
        },
        "summary": "TP all-reduce (graph-capture path uses custom AR)",
        "backends": {
            "custom": {"status": "production",
                       "note": "adapter host-shm custom AR; <512KB fast"},
            "aiter": {"status": "validated",
                      "note": "aiter.dist GraphSafe (15/15 graph replay)"},
            "nccl": {"status": "production",
                     "note": "fallback; multi-GPU graph OFF"},
        },
    },
    "paged_mqa_logits": {
        "vars": {"emu": {"HCU_MQA_EMU": "1"}},
        "summary": "DSA paged MQA logits",
        "backends": {
            "emu": {"status": "production",
                    "note": "torch emulation (default on gfx93*)"},
            "aiter": {"status": "planned",
                      "note": "aiter.paged_mqa_logits; gate pending"},
            "lightop": {"status": "planned",
                        "note": "lightop.attention.paged_mqa_logits; "
                                "gate pending"},
        },
    },
    "rotary_embedding": {
        "var": None,
        "summary": "RoPE NeoX triton in place; GPT-J interleave = torch",
        "backends": {
            "triton": {"status": "production", "note": "no switch needed"},
            "aiter": {"status": "planned",
                      "note": "rope_fwd/cached/thd family"},
            "lightop": {"status": "planned",
                        "note": "rotary_embedding_fuse family"},
        },
    },
    "moe_gate": {
        "var": None,
        "summary": "topk_softmax / topk_sigmoid / moe_fused_gate "
                   "(contract = torch baseline)",
        "backends": {
            "triton": {"status": "production", "note": "no switch needed"},
            "aiter": {"status": "planned",
                      "note": "moe_fused_gate_sqrtsoftplus_v41_hcu / "
                              "biased_grouped_topk"},
            "lightop": {"status": "planned",
                        "note": "fast_topk_transform_* family"},
        },
    },
    "fused_moe": {
        "var": "HCU_FUSED_MOE_IMPL",
        "summary": "w8a8 fused MoE experts (standard dispatch, "
                   "triton runner entry; deepep runner unaffected)",
        "backends": {
            "triton": {"status": "production",
                       "note": "stock kernel; default fallback"},
            "aiter": {"status": "validated",
                      "note": "DTK aiter_moe; blockwise->ASM tuned csv, "
                              "channelwise->moe_c; rel 0.02-0.04 gate "
                              "PASS (nmz18 gfx938); unlisted shapes "
                              "auto-fallback to triton"},
        },
    },
    "per_token_quant": {
        "var": None,
        "summary": "per-token fp8/int8 quant (dynamic)",
        "backends": {
            "torch": {"status": "production", "note": "current path"},
            "aiter": {"status": "planned",
                      "note": "per_token_quant_hip/_triton"},
            "lightop": {"status": "planned",
                        "note": "per-token quant family"},
        },
    },
}

CONFIG_NAME = "hcu_adapter"
_STATE = {"loaded": False, "path": None, "ops": {}, "overrides": {}}


def _candidate_paths():
    out = []
    env = os.environ.get("HCU_CONFIG", "")
    if env:
        out.append(env)
    for base in (os.getcwd(), os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))):
        for ext in ("yaml", "yml", "json"):
            out.append(os.path.join(base, f"{CONFIG_NAME}.{ext}"))
    return out


def _parse_file(path):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        import yaml  # sglang hard-dep; present wherever the adapter runs

        return yaml.safe_load(f)


def _validate(doc, path):
    if not isinstance(doc, dict):
        raise HcuConfigError(f"{path}: top level must be a mapping")
    ver = doc.get("version")
    if ver != 1:
        raise HcuConfigError(
            f"{path}: unsupported version {ver!r} (expected 1)"
        )
    ops = doc.get("ops", {})
    if not isinstance(ops, dict):
        raise HcuConfigError(f"{path}: 'ops' must be a mapping")
    overrides = {}
    chosen = {}
    for op, backend in ops.items():
        if op not in BACKEND_CATALOG:
            raise HcuConfigError(
                f"{path}: unknown op {op!r}; known ops: "
                + ", ".join(sorted(BACKEND_CATALOG))
            )
        spec = BACKEND_CATALOG[op]
        if not isinstance(backend, str):
            raise HcuConfigError(
                f"{path}: ops.{op}: backend must be a string, got "
                f"{backend!r}"
            )
        if backend not in spec["backends"]:
            raise HcuConfigError(
                f"{path}: ops.{op}: unknown backend {backend!r}; "
                f"available: "
                + ", ".join(sorted(spec["backends"]))
            )
        status = spec["backends"][backend]["status"]
        if status == "planned":
            raise HcuConfigError(
                f"{path}: ops.{op}: backend {backend!r} is catalogued "
                f"but not wired yet (planned); selectable backends: "
                + ", ".join(
                    sorted(
                        b for b, s in spec["backends"].items()
                        if s["status"] != "planned"
                    )
                )
            )
        chosen[op] = backend
        if spec.get("vars") is not None:  # compound op
            for var, val in spec["vars"][backend].items():
                overrides[var] = str(val)
        elif spec.get("var"):
            overrides[spec["var"]] = backend
    return chosen, overrides


def load(force=False):
    """Discover, parse and validate the config file (once).

    No file found -> quiet no-op (legacy env/auto/default behavior).
    Invalid file -> HcuConfigError (loud, at first use).
    """
    if _STATE["loaded"] and not force:
        return _STATE
    path = None
    for cand in _candidate_paths():
        if os.path.isfile(cand):
            path = cand
            break
    _STATE.update(loaded=True, path=path, ops={}, overrides={})
    if path is None:
        return _STATE
    doc = _parse_file(path)
    chosen, overrides = _validate(doc, path)
    _STATE.update(ops=chosen, overrides=overrides)
    return _STATE


def lookup(registry_var):
    """-> configured value for a registry variable, or None."""
    st = load()
    return st["overrides"].get(registry_var)


def path():
    return load()["path"]


def summary_line():
    """One startup line; empty string when no file is in effect."""
    st = load()
    if st["path"] is None:
        return ""
    ops = " ".join(f"{k}={v}" for k, v in sorted(st["ops"].items())) or "-"
    return f"HCU adapter config: {st['path']} ({ops})"
