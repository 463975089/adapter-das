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
    # v0.5.22 per-token-group quant JIT: DTK hipcc declares plain __all but
    # not the mask-form __all_sync, killing the whole module family at name
    # lookup (glm53f decode graph capture, 2026-10-20). Full-mask vote from
    # kernel entry == __all(pred); see sgl_kernel/_adapters/jit_ptgq.py.
    ("per_token_quant", "if (__all_sync(warp::kFullMask, skip)) return;"),
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
    # --- GLM-5.2-DSA (gfx938, box16) launch chain: three more hipcc
    # --- portability text patches, same evidence class as the rope ones.
    # DSA indexer k-cache store: HIP fp8x2 alias is a bare uint16_t (no
    # brace ctor) -> delegate pack_fp8 to the proven deepseek_v4 software
    # packer (E4M3FN 448 clip == this file's kFP8E4M3Max assumption).
    (
        "per_token_quant",
        "#include <sgl_kernel/vec.cuh>\n#include <sgl_kernel/warp.cuh>",
    ),
    (
        "per_token_quant",
        "[[maybe_unused]]\n"
        "SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {\n"
        "  return fp8x2_e4m3_t{fp32x2_t{fp8_e4m3_clip(x), fp8_e4m3_clip(y)}};\n"
        "}",
    ),
    (
        "per_token_quant",
        "  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);",
    ),
    (
        "per_token_quant",
        "  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);",
    ),
    # warp.cuh reduce<T>: HIP arm hardcodes a __hip_bfloat162 shuffle
    # receiver -> reduce<T> un-compilable for every non-bf16 T (fp32 warp
    # max in the store kernel). Receiver type widened to T.
    ("per_token_quant", "__hip_bfloat162 _v;"),
    # warp.cuh inclusive scan / broadcast helpers use the *_sync shuffle
    # variants; DTK device headers only declare the non-sync family. First
    # hit: deepseek_v4/topk_impl.cuh radix tie-select -> warp::inclusive_sum
    # -> '__shfl_up_sync' use-of-undeclared-identifier, which failed every
    # dsa_mha_one_shot_* subcase of upstream
    # test/registered/attention/unittests/dsa/test_dsa.py
    # (sgl_kernel_jit_dpsk_v4_topk_v2_false build). HIP shuffles are
    # wave-collective and ignore the mask, so the plain form is the
    # drop-in equivalent.
    ("per_token_quant", "__shfl_up_sync(active_mask, val, offset, kWidth);"),
    ("per_token_quant", "__shfl_down_sync(active_mask, val, offset, kWidth);"),
    ("per_token_quant", "__shfl_sync(active_mask, value, src_lane, kWidth);"),
    # concat_mla.cuh: hand-written PTX (laneid register, L1::no_allocate
    # st/ld with "l" constraint, prefetch.global.L2) -> HIP-portable
    # equivalents (utils.cuh lane identity / plain volatile derefs).
    (
        "platform_detect",
        "SGL_DEVICE int get_lane_id() {\n"
        "  int lane_id;\n"
        '  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));\n'
        "  return lane_id;\n"
        "}",
    ),
    (
        "platform_detect",
        "SGL_DEVICE void st_na_global_v1(const int* ptr, int v) {\n"
        '  asm volatile("st.global.L1::no_allocate.s32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");\n'
        "}",
    ),
    (
        "platform_detect",
        "SGL_DEVICE void st_na_global_v2(const int2* ptr, const int2& v) {\n"
        '  asm volatile("st.global.L1::no_allocate.v2.s32 [%0], {%1, %2};" ::"l"(ptr), "r"(v.x), "r"(v.y) : "memory");\n'
        "}",
    ),
    (
        "platform_detect",
        "SGL_DEVICE int ld_na_global_v1(const int* ptr) {\n"
        "  int r;\n"
        '  asm volatile("ld.global.nc.L1::no_allocate.s32 %0, [%1];" : "=r"(r) : "l"(ptr));\n'
        "  return r;\n"
        "}",
    ),
    (
        "platform_detect",
        "SGL_DEVICE int2 ld_na_global_v2(const int2* ptr) {\n"
        "  int2 r;\n"
        '  asm volatile("ld.global.nc.L1::no_allocate.v2.s32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(ptr));\n'
        "  return r;\n"
        "}",
    ),
    # deepseek_v4/paged_mqa_metadata.cuh includes NVIDIA cub for BlockScan;
    # DTK ships hipCUB only (namespace cub mirrors the API, umbrella header
    # per the tree's own lora JIT convention) — DSV4-Flash decode graph
    # capture, nmz16 gfx938 2026-10-20. See sgl_kernel/_adapters/jit_dpskmeta.py.
    ("platform_detect", "#include <cub/block/block_scan.cuh>"),
    # same file, Phase-1/2 hand scans: mask-form shuffles undeclared on DTK;
    # width=32 HIP forms keep the exact CUDA 32-lane warp segmentation.
    ("platform_detect", "uint32_t y = __shfl_up_sync(0xffffffff, v, o);"),
    ("platform_detect", "running = __shfl_sync(0xffffffff, v, 31);"),
    # c_plan.cuh compress-plan: structured bind into HIP uint2 (anonymous
    # union) — 4th site of the ABI fact; unpack_w returns a real uint2 so
    # member reads are type-identical (DSV4-Flash nmz16 capture chain).
    (
        "platform_detect",
        "    const auto [ragged_id, batch_id] = unpack_w(plan_w);",
    ),
    # main_norm_rope / topk_bf16_small / flashmla_sched_meta / fused_norm_rope:
    # bare __syncwarp, %laneid PTX, mask-form shuffles, float2 binds — all
    # established needle classes (jit_rope / jit_concat_mla / warp.cuh
    # precedents); DSV4-Flash capture chain 2026-10-20.
    ("platform_detect", "  __syncwarp();"),
    ("platform_detect", "  const auto [freq_real, freq_imag] = freq;"),
    (
        "platform_detect",
        "        const float other = __shfl_xor_sync(0xFFFFFFFFu, data[i], mask, kWarpThreads);",
    ),
    (
        "platform_detect",
        '  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));',
    ),
    (
        "platform_detect",
        "    const auto t = __shfl_down_sync(device::warp::kFullMask, inc, offset);",
    ),
    (
        "platform_detect",
        "  warp_base = __shfl_sync(warp::kFullMask, warp_base, kWarpThreads - 1);",
    ),
    (
        "platform_detect",
        "    partial += __shfl_xor_sync(uint32_t(-1), partial, offset);",
    ),
    (
        "platform_detect",
        "    const auto [freq_real, freq_imag] = freq;",
    ),
    # concat_mla.cuh host side: TensorMatcher pins the dldevice enum to
    # kDLCUDA; ROCm tensors arrive as kDLROCM (upstream's own kDLGPU
    # indirection in utils.cuh exists for this reason). Six identical sites.
    ("platform_detect", "        .with_device<kDLCUDA>(device)"),
    # gemm/dot_product.cuh: bf16 dot-product accumulation decomposes
    # fp32x2_t (HIP float2, anonymous union) — third site of the same ABI
    # fact after rope.cuh / fused_store_index_cache.cuh. .x/.y member reads.
    (
        "platform_detect",
        "    const auto [a0, a1] = cast<fp32x2_t>(a[i]);",
    ),
    (
        "platform_detect",
        "    const auto [b0, b1] = cast<fp32x2_t>(b[i]);",
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
