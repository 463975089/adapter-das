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
      allreduce: aiter          # aiter | nccl (custom AR retired 2026-10-19)
      paged_mqa_logits: triton   # triton (DEFAULT, R-13 #4) | emu | vendor

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
                A/B triage)
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
            "aiter": {"HCU_AITER_AR": "1"},
            "nccl": {"HCU_AITER_AR": "0"},
        },
        "summary": "TP all-reduce (+fused AR+RMSNorm seam, HCU_AR_NORM_IMPL)",
        "backends": {
            "aiter": {"status": "production",
                      "note": "aiter.dist GraphSafe; 15/15 graph replay; "
                              "1/40 skip-think in the R-01 verdict arms — "
                              "the default landing spot. 2026-10-19 final "
                              "stress on image sf_b075-20261018-0048 "
                              "(aiter 2610171925): 4-rank AND 8-rank "
                              "15/15 PASS (eager gold + NCCL sanity + "
                              "graph capture with 4 fresh-input replays "
                              "per size/dtype), latency on par with "
                              "NCCL (88 vs 69us small msgs, 0.93-0.95x "
                              "at 1-16MB). The homegrown custom AR was "
                              "RETIRED the same day (user decision): the "
                              "R-13#1 fix is moot and the broken kernel "
                              "is useless as a numerics reference. "
                              "R-09① fused AR+RMSNorm seam (2026-10-19, "
                              "ar_norm_aiter 桥 v2, HCU_AR_NORM_IMPL auto): "
                              "arm=apply_flashinfer_allreduce_fusion 三副本"
                              "重绑（上游候选门要 SGLANG_USE_AITER 或 "
                              "sm90/100，HIP 恒关——不开全局门的外科开法）"
                              "+dispatch=flashinfer 入口→GroupCoordinator."
                              "fused_allreduce_rmsnorm→custom_fused_ar_rms；"
                              "8rank 探针 fp64 真值 1ulp/0ulp（优于 rccl "
                              "3-8ulp）、vs customAR+norm 1.38-2.11×、图回"
                              "放精确；tp8 serving 实证介入（图捕获期起服、"
                              "216 次 0 失败、输出正常）；dp-attention 恒退 "
                              "stock（arm 层即拒）"},
            "nccl": {"status": "production",
                     "note": "fallback (HCU_AITER_AR=0); graph-correct"},
        },
    },
    "kda_prefill": {
        "vars": {"fork": {"HCU_KDA_PREFILL_IMPL": "fork"},
                 "emu": {"HCU_KDA_PREFILL_IMPL": "emu"},
                 "stock": {"HCU_KDA_PREFILL_IMPL": "stock"}},
        "summary": "KDA (gated delta) prefill chunked kernel",
        "backends": {
            "fork": {"status": "production",
                     "note": "vendored sglang-das hcu chunk_kda triton — "
                             "DEFAULT on hip since 2026-10-18. Parity cos "
                             "0.99999 vs emu at serving shapes, 473-1009x "
                             "faster (0.88 vs 414 ms @T=2048). The 10-16 "
                             "'fork flips Berlin' serving gate was "
                             "re-attributed to the custom-AR store-ordering "
                             "bug (160-arm A/B: custom 60/100 vs non-custom "
                             "1/60); clean-route re-verification 2026-10-18 "
                             "(post host GPU reset): bothfork anchors 4/4 + "
                             "marathon 0/20 + long-ctx 4972 tok correct."},
            "emu": {"status": "production",
                    "note": "sequential torch emulation — correct opt-out "
                            "(HCU_KDA_PREFILL_IMPL=emu), 473-1009x slower"},
            "stock": {"status": "broken",
                      "note": "v0.5.21 stock triton mis-compiles on gfx938 "
                              "(int64 scalar-load div-4 family)"},
        },
    },
    "conv1d": {
        "vars": {"fork": {"HCU_CONV1D_IMPL": "fork"},
                 "emu": {"HCU_CONV1D_IMPL": "emu"},
                 "stock": {"HCU_CONV1D_IMPL": "stock"}},
        "summary": "causal conv1d fn + decode update",
        "backends": {
            "fork": {"status": "production",
                     "note": "vendored hcu conv1d triton — DEFAULT on hip "
                             "since 2026-10-18. fn cos 0.999996/16.4x, "
                             "update cos 0.999997/4.3x vs emu; conv-state "
                             "bit-exact. Polluted-era 'conv1d flips "
                             "9-10/20' observation voided (KFD bad-state "
                             "data); clean re-verification 2026-10-18: "
                             "conv-only arm marathon 0/20 + Tokyo + "
                             "long-ctx correct."},
            "emu": {"status": "production",
                    "note": "torch emulation — correct opt-out "
                            "(HCU_CONV1D_IMPL=emu)"},
            "stock": {"status": "broken",
                      "note": "red gates kernels/test_conv1d_* on gfx938"},
        },
    },
    "paged_mqa_logits": {
        "vars": {"emu": {"HCU_MQA_EMU": "1"},
                 "triton": {"HCU_MQA_LOGITS_IMPL": "triton"},
                 "deepgemm": {"HCU_MQA_LOGITS_IMPL": "vendor"}},
        "summary": "DSA paged MQA logits",
        "backends": {
            "emu": {"status": "production",
                    "note": "torch emulation — correct opt-out "
                            "(HCU_MQA_LOGITS_IMPL=vec; pre-R-13#4 default)"},
            "triton": {"status": "production",
                       "note": "homegrown triton kernel (mqa_logits_triton.py, "
                               "R-02 2026-10-18) — DEFAULT since R-13 #4 "
                               "promotion 2026-10-18 (HCU_MQA_LOGITS_IMPL, "
                               "was emu-vec). Reads the production "
                               "pool layout (writer census: GLM-5.3 "
                               "IndexerKPool/kpool=4 writes via the kpool "
                               "family + SetKAndS under the forced "
                               "preshuffle gate -> 16-tile interleave k + "
                               "fp32 tail scales; prefill gather reads the "
                               "same order and is consistent). Sibling fix: "
                               "the torch emu's row-level 132-byte split "
                               "misread that page-level layout — decode "
                               "DSA top-k was deterministically wrong in "
                               "production (anchors top-k-insensitive); "
                               "fixed in dsa_mqa_logits_emu.py (interleave-aware "
                               "page-level unpack; row-major arm behind "
                               "HCU_DSA_PRESHUFFLE=0 for the kpool=1/"
                               "fused_store world). Parity: exact vs direct "
                               "math on real-writer pools, both arms (rel "
                               "0.0, top-k full); decode-shape timing "
                               "13.2x over emu-vec (0.19 vs 2.52 ms, N=8 "
               "window 64K). Gate tests/kernels/test_mqa_triton_gfx938.py "
                               "(fixed-emu cross-check included). Serving "
                               "A/B passed 2026-10-18 (zero-env tp4 "
                               "GLM-5.3, nmz26 GPU4-7): four gates green, "
                               "marathon 0/20 + Tokyo (unified post-think "
                               "metric), long-ctx 4972 tok correct; decode "
                               "+9% short / +4% long-ctx vs emu-vec arm "
                               "(31.35 vs 34.18 s / 256 tok; 64.6 vs 66.7 "
                               "s). Promoted to production + default flip "
                               "2026-10-18 (R-13 #4); emu-vec stays as the "
                               "opt-out (HCU_MQA_LOGITS_IMPL=vec)."},
            "aiter": {"status": "planned",
                      "note": "probed absent 2026-10-18: the on-box aiter "
                              "wheel ships no paged_mqa_logits API -- "
                              "revisit when the wheel grows one"},
            "lightop": {"status": "planned",
                        "note": "lightop.attention.paged_mqa_logits; "
                                "gate pending"},
            "deepgemm": {"status": "broken",
                         "note": "deepgemm wheel direct call via "
                                 "mqa_vendor.py bridge. Layout contract "
                                 "RECOVERED 2026-10-18 (window-collapse "
                                 "probes): (P,64,1,132) uint8 pages, k "
                                 "row-major + 64 fp32 scales at page tail "
                                 "(upstream SM90 layout = the kpool=1/"
                                 "fused_store world; the GLM-5.3 kpool=4 "
                                 "pool is 16-tile interleave instead — "
                                 "extra relayout needed for kpool models "
                                 "even with a fixed wheel). Metadata gates "
                                 "passed: q fp8e4m3fn 4D [B,R,H,128], H in "
                                 "{8,16,32,64}, weights fp32 (N,H), "
                                 "lens/bt int32. CONVICTED WHEEL BUG: the "
                                 "gfx938 kernel clamps NEGATIVE fp8 q AND "
                                 "k bytes to 0 (logits == sum w*max(q,0)*"
                                 "max(k,0)*s, exact to 4 decimals) — real "
                                 "signed data yields garbage (rel~3 vs "
                                 "emu). All-positive data matches the emu "
                                 "exactly. Blocked on a DTK kernel fix "
                                 "(deepgemm 2.1.0+dtk2604); env override "
                                 "HCU_MQA_LOGITS_IMPL=vendor stays for "
                                 "triage. Dual-fact watchtower: "
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
                               "emu）；页号 int64 展开；"
                               "varlen 预填充面=自研 fa_varlen_triton.py"
                               "（ragged dense K/V、fp32+ieee dot、swa + "
                               "fp8-KV descale、静态 grid 图安全；哨兵 "
                               "test_fa_varlen_triton.py 三段式 + "
                               "capture-replay + 零长序列 pit）；"
                               "Dv!=D/fp8-q/softcap/qv/sinks 面仍 emu"
                               "（BW1101 对拍，性能待 BW1102 复核）。"
                               "Serving A/B executed 2026-10-19 (glm53f "
                               "tp4 zero-env vs HCU_FA_IMPL=triton, "
                               "nmz26): quality+perf parity <=8k ctx "
                               "(anchors/decode 8.27 tok/s identical/"
                               "prefill within +-10%/LONGCTX facts pass); "
                               "first boot killed by a wrapper bug="
                               "q_v face (only_qv+qv) dropped by the ops "
                               "rebind -> emu assert (fixed 2759711). "
                               "RED AT THE EDGE (superseded): long-ctx "
                               "(11k tok, 8192+2797 chunks) killed the "
                               "triton-arm server 2/2 boots -- root cause "
                               "was the emu's un-tiled (maxL x C) score "
                               "slabs, FIXED for ALL arms by the q-row "
                               "sub-tile (c59135b, peak 22.02 -> 6.80 GiB "
                               "on the 8192-face; fp64 golden 0.0046; "
                               "one-pit test_extend_face_peak_bounded). "
                               "Post-fix serving verdict (2026-10-19, "
                               "torch arm on c59135b): wall 11k x3 green "
                               "(11.3s, = auto arm), full battery green, "
                               "anchors byte-identical to the auto arm "
                               "(the pre-fix textual drift was the same "
                               "tile sensitivity), LONGCTX-6k 28.4 -> "
                               "3.1s (~9x, tile working-set bonus), "
                               "decode 8.27 tok/s unchanged. Stays "
                               "catalog validated/candidate: glm53f's "
                               "q_v paged face keeps the emu, so flipping "
                               "the default is inert on this model -- "
                               "promote when a model exercising the "
                               "triton core passes serving A/B."},
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
    "w4a16_moe": {
        "var": "HCU_W4A16_MOE_IMPL",
        "summary": "W4A16 fused MoE experts (moe_wna16 AWQ/GPTQ-int4 line)",
        "backends": {
            "triton": {"status": "production",
                       "note": "stock MoeWNA16 triton arm; auto answers it "
                               "until the deepgemm tier passes serving A/B"},
            "deepgemm": {"status": "candidate",
                         "note": "R-05 bridge 2026-10-19: DTK deepgemm "
                                 "masked w4a16 grouped C++ GEMMs as the "
                                 "experts compute behind a hand dispatch/"
                                 "silu/combine pipeline (eager-only v1 -- "
                                 "capture/inplace stay on triton). Engage "
                                 "guards: no-zeros symmetric (q-8)*s (same "
                                 "fixed convention as the wheel), group "
                                 "32/64/128 (one-time scale expansion to "
                                 "the g32 grid), K%128/I%128/2I%16 "
                                 "aligned, qweight already in the wheel's "
                                 "b layout so engagement is a marlin "
                                 "repack + cached scale expansion only. "
                                 "Recon rel 0.0022; three-tier gate "
                                 "tests/kernels/test_w4a16_moe_deepgemm.py; "
                                 "bench vs triton + e2e pending."},
        },
    },
    "per_token_quant": {
        "var": "HCU_QUANT_IMPL",
        "summary": "per-token fp8/int8 quant (dynamic)",
        "backends": {
            "torch": {"status": "production",
                      "note": "upstream natives（scaled_fp8_quant torch "
                              "fallback+triton raw group；rowwise 包装在 "
                              "HIP NameError=断，quant_aiter 桥修复）"},
            "aiter": {"status": "candidate",
                      "note": "R-09③ 2026-10-19 落位（quant_aiter 桥，"
                              "auto 默认）：rowwise=行为还原（HIP NameError "
                              "修复）+group plain 臂 15.1×（尺度与金标准"
                              "逐位一致/q RTN 命中 99.8%）+scaled 两 "
                              "dynamic natives=上游 SGLANG_USE_AITER 同款"
                              "调用（static 恒原生：aiter 轮 kernel 无 "
                              "clamp，|x/s|>448 溢出 NaN 21% 实证）+整行 "
                              "fused "
                              "silu+quant 臂（R-09②，7.5×）；门禁 "
                              "tests/kernels/test_quant_aiter_gfx938.py；"
                              "异形臂恒走上游、aiter 失败 log-once 回退；"
                              "production 晋级待 fp8 模型 serving A/B"},
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
