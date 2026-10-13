# Runtime implementation registry — who is actually serving each op.

# The catalog (hcu_config.BACKEND_CATALOG) declares what CAN be selected;
# this module records what IS installed right now, per catalog op:
# which call-site got rebound, into what, and by which install mechanism.
# It is the data side of the runtime dump (dump()/log_summary()) and the
# arbiter for the L0 static rules that forbid duplicate / unregistered
# patches -- see docs/governance.md.

# Roles (mirrors governance §2):
#   active  — serves production traffic; exactly one per catalog entry
#   ref     — framework-free reference implementation for numerics triage;
#             NEVER routed into production by a bridge
# Bridges may only bind `active` implementations; binding a `ref` impl is
# an L0 violation (tests/tools has _test_ scripts that exercise refs
# directly instead).

from __future__ import annotations

import json
import logging
import os
import sys

LOG = logging.getLogger("hcu.adapter")

# mechanism -- how the binding landed (governance §3 ladder, top = preferred)
MECH_SEAM = "seam"          # official upstream hook (--quantization, backend registry)
MECH_REBIND = "rebind"      # module-attribute swap of a pure-python entry point
MECH_TEXT = "text-patch"    # source text replace + recompile (assert relax) — allowlist

# roles
ROLE_ACTIVE = "active"
ROLE_REF = "ref"

# op -> record; op keys ARE catalog keys (l0_gate cross-checks this).
_RECORDS = {}
# target symbol -> op, for duplicate-bind detection at install time
_BOUND = {}

# text-patch allowlist: exact (op, needle) pairs permitted. Anything else
# attempting MECH_TEXT must raise at install time; l0_gate also counts
# occurrences in sitecustomize.py against this table.
TEXT_PATCH_ALLOWLIST = {
    ("fused_moe", "assert _is_cuda or _is_xpu, ("),
    ("kda", "t < t_slot"),  # prefill-CP rope fused guard, same pattern
    # torch2.11 cannot infer dtype from numpy scalars (np.prod int64) when
    # building the KV-pool stride tensor (dlpack route); int() restores the
    # pre-dlpack behavior "python int infers an integer dtype".
    ("kv_pool_strides", "np.prod(x.shape[1:]) * x.dtype.itemsize"),
    # gfx936 DTK hipcc (clang) rejects nvcc-isms in upstream JIT rope:
    # bare __syncwarp() (needs mask arg via portability shim) and structured
    # bindings into HIP_vector_type (anonymous union). Patch both to
    # portable equivalents at JIT-source-read time.
    ("rotary_embedding", "  __syncwarp();  // to avoid warp divergence"),
    ("rotary_embedding", "      const auto [x0, x1] = cast<fp32x2_t>(input_vec_x[j]);"),
    # torch2.11 exports numpy arrays through DLPack; numpy 1.25 cannot signal
    # the read-only flag, so the frombuffer payload must be a writable copy
    # before ByteTensor construction (source-level: every import path binds
    # the patched body).
    ("dist_broadcast", "torch.ByteTensor(\n                np.frombuffer(serialized_data, dtype=np.uint8)\n            )"),
    # JIT per-token-group-quant fp8: the CUDA DTypeTrait<fp8>::kFloatMax
    # member is compiled out under USE_ROCM; the HIP-side packed convert is
    # not the CUDA intrinsic. Repoint the clamp constant at the literal
    # 448.0f and the packed convert at the DTK __hip_cvt family (OCP e4m3
    # == CUDA e4m3_fn; deep_gemm EP runner needs this JIT kernel).
    ("per_token_quant", "static constexpr float kMaxValue = DTypeTrait<fp8_e4m3_t>::kFloatMax;"),
    (
        "per_token_quant",
        "  SGL_DEVICE static packed2_t quant(const float2 v) {\n"
        "    return packed2_t{float2{fminf(v.x, kMaxValue), fminf(v.y, kMaxValue)}};\n"
        "  }",
    ),
    (
        "per_token_quant",
        "#else\n"
        "      value = Trait::reduce(value, __shfl_xor(value, mask, kStart));\n"
        "#endif",
    ),
    (
        "per_token_quant",
        "#else\n"
        "      value = Trait::reduce(value, __shfl_xor(value, mask, kFinish));\n"
        "#endif",
    ),
}


def register(op, impl, target, mechanism, role=ROLE_ACTIVE, source=None,
             detail=None):
    """Record that `op` is now served by `impl` via `mechanism`.

    target  — dotted symbol that got rebound/patched (what a debugger would
              set a breakpoint in, e.g. "sglang...fused_moe.fused_experts")
    source  — module that performed the install; defaults to caller's file
    Raises on: unknown op (not in catalog), role violation, duplicate
    binding of a target, non-allowlisted text patch.
    """
    from . import hcu_config

    if op not in hcu_config.BACKEND_CATALOG:
        raise ValueError(
            f"registry: op {op!r} not in BACKEND_CATALOG; a binding without"
            " a catalog entry does not exist (see docs/governance.md)"
        )
    if mechanism == MECH_TEXT:
        needle = detail if isinstance(detail, str) else None
        if (op, needle) not in TEXT_PATCH_ALLOWLIST:
            raise ValueError(
                f"registry: text-patch for {op!r} ({needle!r}) is not in"
                " TEXT_PATCH_ALLOWLIST -- new patches need a governance"
                " review (docs/governance.md §3)"
            )
    src = source or _caller()
    prev = _BOUND.get(target)
    if prev is not None and prev != op:
        raise ValueError(
            f"registry: target {target!r} already bound to op {prev!r};"
            " duplicate patches are forbidden -- extend the existing entry"
        )
    _BOUND[target] = op
    _RECORDS[op] = {
        "op": op,
        "impl": getattr(impl, "__qualname__", repr(impl)) if impl is not None else repr(impl),
        "target": target,
        "mechanism": mechanism,
        "role": role,
        "source": src,
    }
    if role == ROLE_REF:
        # refs never serve traffic; nag loudly if registered from a bridge
        LOG.debug("registry: ref impl for %s recorded from %s", op, src)


def _caller():
    stack = sys._getframe(2)
    return f"{stack.f_globals.get('__name__', '?')}:{stack.f_lineno}"


def get(op):
    return _RECORDS.get(op)


def records():
    return dict(_RECORDS)


TEXT_PATCH_TARGET_PREFIX = "text:"  # target convention for text patches


def dump(path=None):
    """Machine-readable dump. dict if path is None, else JSON written to path.

    Structure: {"ops": {op: record}, "unbound": [catalog ops with no
    active record], "env": every resolved HCU_* value+source (from
    hcu_env.log_summary's data)}. CI and the config-level bisection
    scripts consume this; humans read log_summary().
    """
    from . import hcu_config, hcu_env

    unbound = [op for op in hcu_config.BACKEND_CATALOG if op not in _RECORDS]
    doc = {
        "ops": _RECORDS,
        "unbound": unbound,
        "env": {
            name: dict(zip(("value", "source"), hcu_env.resolve(name)))
            for name in hcu_env.REGISTRY
        },
    }
    if path is None:
        return doc
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return doc


def log_summary(logger=None):
    """One line per installed binding — the 'who is running' startup log."""
    lg = logger or LOG
    if not _RECORDS:
        lg.warning("HCU adapter runtime: no ops installed (pure upstream)")
        return
    for op in sorted(_RECORDS):
        r = _RECORDS[op]
        lg.warning(
            "HCU adapter runtime: %-18s <- %-40s [%s/%s] via %s",
            op, r["impl"], r["role"], r["mechanism"], r["target"],
        )
    from . import hcu_config

    unbound = [op for op in hcu_config.BACKEND_CATALOG if op not in _RECORDS]
    if unbound:
        lg.warning(
            "HCU adapter runtime: catalog ops on upstream path (no patch):"
            " %s", ", ".join(sorted(unbound)),
        )
