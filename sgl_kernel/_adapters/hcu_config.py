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
whose status is "planned" or "broken" fail LOUDLY at load (typo guard --
a silent fallback to defaults would hide exactly the mistake the user
made).

BACKEND_CATALOG is the single source of truth for what can be selected;
the interactive wizard (tools/hcu_config_wizard.py) and the future web
UI both render from it. status:
    production  default, fully gated
    validated   passed all gates, opt-in (or per-wheel auto rule)
    candidate   wired, awaiting its gate matrix
    broken      once shipped, now convicted of corruption -- selecting
                raises at config load (env override still possible for
                A/B triage, e.g. HCU_TRITON_AR=1)
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
        "summary": "TP all-reduce",
        "backends": {
            "custom": {"status": "broken",
                       "note": "cross-block device-scope rendezvous does not "
                               "order host-shm payload stores; stale reads "
                               "flip outputs (skip-think 48/80 vs 1/40 on "
                               "the =0 route, nmz26 tp4 2026-10-15). Default "
                               "OFF pending ar_ext.cu fix"},
            "aiter": {"status": "production",
                      "note": "aiter.dist GraphSafe; 15/15 graph replay; "
                              "1/40 skip-think in the R-01 verdict arms — "
                              "the default landing spot for TRITON_AR=0"},
            "nccl": {"status": "production",
                     "note": "fallback (HCU_AITER_AR=0); graph-correct"},
        },
    },
    "paged_mqa_logits": {
        "vars": {"emu": {"HCU_MQA_EMU": "1"},
                 "deepgemm": {"HCU_MQA_LOGITS_IMPL": "vendor"}},
        "summary": "DSA paged MQA logits",
        "backends": {
            "emu": {"status": "production",
                    "note": "torch emulation (default on gfx93*)"},
            "aiter": {"status": "planned",
                      "note": "probed absent 2026-10-18: the on-box aiter "
                              "wheel ships no paged_mqa_logits API -- "
                              "revisit when the wheel grows one"},
            "lightop": {"status": "planned",
                        "note": "lightop.attention.paged_mqa_logits; "
                                "gate pending"},
            "deepgemm": {"status": "candidate",
                         "note": "deepgemm wheel direct call via "
                                 "mqa_vendor.py bridge. Metadata gates "
                                 "all passed on-box (probe 2026-10-18): "
                                 "q must be fp8e4m3fn 4D [B,R,H,128] "
                                 "with H in {8,16,32,64}; kv uint8 4D "
                                 "(P,S,1,132); weights fp32 (N,H); "
                                 "lens/block_table int32. LAYOUT GATE "
                                 "NOT WIRED: the kernel reads the "
                                 "ORIGINAL preshuffle kpool page layout "
                                 "(8192B 16-tile-interleaved fp8 + 256B "
                                 "fp32 scales per 64-slot page), so "
                                 "feeding it the flat (T,132) tensor "
                                 "the kpool read-emu unpacks yields "
                                 "garbage (rel=1.0) -- proper wiring "
                                 "must feed the RAW pool at the dsa "
                                 "layer. KNOWN-RED watchtower: "
                                 "tests/kernels/test_mqa_vendor_gfx938.py"},
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
    "platform_detect": {
        "var": "HCU_PLATFORM_SEMANTICS",
        "summary": "sglang 平台判定语义（§0.5 is_cuda 路线的落地点）",
        "backends": {
            "cuda": {"status": "production",
                     "note": "default (§0.5)；sglang is_cuda/is_hip/platforms "
                             "统一回 CUDA 语义，新模型天然走 CUDA 分支"},
            "hip": {"status": "validated",
                    "note": "legacy 适配路线；降级调试 / A-B 对照"},
        },
    },
    "model_env": {
        "var": "HCU_MODEL_ENV",
        "summary": "模型感知 env 自动化（R-08 B 类）",
        "backends": {
            "auto": {"status": "production",
                     "note": "ModelRunner init 按 model_config setdefault："
                             "DSA 模型关两项融合（无 HIP 臂）、fp16 GDN 底模"
                             " conv dtype float16；显式 export 永远赢"},
            "off": {"status": "production",
                    "note": "退回手敲 env（triage）"},
        },
    },
    "kv_pool_strides": {
        "var": None,
        "summary": "KV 池 stride 张量构造（行为还原：torch2.11 不能从 "
                   "numpy 标量推 dtype）",
        "backends": {
            "text-patch": {"status": "production",
                           "note": "int(np.prod(...))，白名单 needle"},
        },
    },
    "dist_broadcast": {
        "var": None,
        "summary": "TP init 广播 payload（行为还原：torch2.11 dlpack 只读 "
                   "BufferError）",
        "backends": {
            "writable-copy": {"status": "production",
                              "note": "common.np.frombuffer 代理返回可写副本"},
        },
    },
    "triton_pdl": {
        "var": None,
        "summary": "triton 编译期 PDL 指令剥离（行为还原：gfx936 无 "
                   "griddepcontrol）",
        "backends": {
            "strip": {"status": "production",
                      "note": "make_amdgcn 输出剔除 griddepcontrol.* 两行，"
                              "语义=PDL 关"},
        },
    },
    "triton_libdevice": {
        "var": None,
        "summary": "triton extra.cuda libdevice -> hip/OCML 重定向（is_cuda "
                   "语义选路修正）",
        "backends": {
            "hip-redirect": {"status": "production",
                             "note": "__nv_* -> __ocml_*；hip 缺的名响亮 stub"},
        },
    },
    "flash_attention": {
        "var": "HCU_FA_IMPL",
        "summary": "fa3 backend 注意力面（kvcache 分页/统一窗 + varlen prefill）",
        "backends": {
            "vendor": {"status": "candidate",
                       "note": "DTK flash_attn 2.8.4 wheel（gfx93x kernel）；"
                               "uniform-window 离线对拍逐位等，但真实 decode "
                               "词汤（2026-10-12 bw7，gfx936 wheel paged 内核"
                               "问题）；=vendor 显式 opt-in 做 A-B"},
            "emu": {"status": "production",
                    "note": "chunked online-softmax torch 模拟（默认；全契约 "
                            "graph-safe）"},
            "triton": {"status": "candidate",
                       "note": "分页 triton 核（自研，fp32 累加）：decode L=1 "
                               "+ 宽窗 L>1 bottom-right causal 全接（"
                               "tests/tools/_test_paged_attn_triton.py 10/10 "
                               "双卡型 ALL-PASS；decode 4.7×、prefill ~7× "
                               "emu）；varlen/paged-decode 之外的面仍 emu；"
                               "页号 int64 展开"},
        },
    },
    "sampling": {
        "var": None,
        "summary": "top-k/top-p/min-p renorm+sample（§0.5 is_cuda sampler 面）",
        "backends": {
            "torch": {"status": "production",
                      "note": "masked filter->renorm->multinomial，fp32；"
                              "每 decode 一步一次，torch 档合理起步"},
            "triton": {"status": "planned",
                       "note": "与 flashinfer renorm 核对拍后可选"},
        },
    },
    "linear_attn_gdn": {
        "var": "HCU_LINEAR_ATTN_GDN",
        "summary": "GDN/gated-delta-rule 线性注意力（sglang fla 面）",
        "backends": {
            "triton": {"status": "production",
                       "note": "上游 fla triton 核（auto 默认；gfx93x "
                               "wave64 友好，BT=64）"},
            "aiter": {"status": "validated",
                      "note": "DTK aiter CK/triton 核路由（fwd_h 状态递推 "
                              "blockdim64 + packed decode 同契约直换绑，"
                              "K=V=128）：门禁 6/6（h 逐位一致、状态池写回 "
                              "≤1e-4 fp32 舍入）+ Qwen3.5-27B TP4 e2e 四锚点"
                              "全对（2026-10-13 bw7）；decode graph 36 桶捕"
                              "获 577s→20.9s；=aiter 显式启用"} ,
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
                        if s["status"] not in ("planned", "broken")
                    )
                )
            )
        if status == "broken":
            note = spec["backends"][backend].get("note", "")
            raise HcuConfigError(
                f"{path}: ops.{op}: backend {backend!r} is known-broken "
                f"and refused: {note} Selectable backends: "
                + ", ".join(
                    sorted(
                        b for b, s in spec["backends"].items()
                        if s["status"] not in ("planned", "broken")
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
