# 算子档位矩阵（自动生成，勿手改）

> 生成：`python tools/gen_tier_matrix.py`（最近 2026-10-28）；事实源 = `BACKEND_CATALOG`（hcu_adapter/hcu_config.py）。
> L0 `impl-naming` 做新鲜度校验改表须随 catalog 同 commit（governance §3b）。
> 文件命名导航规则见 governance §3b；缺档 = 该行空白——那是实现排期的直接输入。

## silu_and_mul

gated activation out = silu(x[:,:d]) * x[:,d:]

| 档位 | 状态 / 备注 |
|---|---|
| triton | production default (auto); fp32 accumulate |
| lightop | validated lightop.fuse_silu_and_mul (input, ou |
| torch | production reference path, eager debug |

路由变量：`HCU_SILU_MUL_IMPL`

## rms_norm

RMSNorm; fused_add_rmsnorm/gemma variants stay triton

| 档位 | 状态 / 备注 |
|---|---|
| triton | production default |
| aiter | candidate aiter.rmsnorm_forward(x,w,eps,traini |
| torch | production reference path |

路由变量：`HCU_RMS_NORM_IMPL`

## allreduce

TP all-reduce (+fused AR+RMSNorm seam, HCU_AR_NORM_IMPL)

| 档位 | 状态 / 备注 |
|---|---|
| aiter | production aiter.dist GraphSafe; 15/15 graph re |
| nccl | production fallback (HCU_AITER_AR=0); graph-cor |

路由变量：`1`/`0`

## kda_prefill

KDA (gated delta) prefill chunked kernel

| 档位 | 状态 / 备注 |
|---|---|
| emu | production sequential torch emulation — correct |
| fork | production vendored sglang-das hcu chunk_kda tr |
| stock | broken v0.5.21 stock triton mis-compiles on |

路由变量：`fork`/`emu`/`stock`

## conv1d

causal conv1d fn + decode update

| 档位 | 状态 / 备注 |
|---|---|
| emu | production torch emulation — correct opt-out (H |
| fork | production vendored hcu conv1d triton — DEFAULT |
| stock | broken red gates kernels/test_conv1d_* on g |

路由变量：`fork`/`emu`/`stock`

## paged_mqa_logits

DSA paged MQA logits

| 档位 | 状态 / 备注 |
|---|---|
| triton | production homegrown triton kernel (mqa_logits_ |
| emu | production torch emulation — correct opt-out (H |
| aiter | planned probed absent 2026-10-18: the on-box |
| lightop | planned lightop.attention.paged_mqa_logits;  |
| deepgemm | broken deepgemm wheel direct call via mqa_v |

路由变量：`1`/`triton`/`vendor`

## rotary_embedding

RoPE NeoX triton in place; GPT-J interleave = torch

| 档位 | 状态 / 备注 |
|---|---|
| triton | production no switch needed |
| aiter | planned rope_fwd/cached/thd family |
| lightop | planned rotary_embedding_fuse family |

路由变量：（auto）

## moe_gate

topk_softmax / topk_sigmoid / moe_fused_gate (contract = torch baseline)

| 档位 | 状态 / 备注 |
|---|---|
| triton | production no switch needed |
| aiter | planned moe_fused_gate_sqrtsoftplus_v41_hcu  |
| lightop | planned fast_topk_transform_* family |

路由变量：（auto）

## platform_detect

sglang 平台判定语义（§0.5 is_cuda 路线的落地点）

| 档位 | 状态 / 备注 |
|---|---|
| cuda | production default (§0.5)；sglang is_cuda/is_hip |
| hip | validated legacy 适配路线；降级调试 / A-B 对照 |

路由变量：`HCU_PLATFORM_SEMANTICS`

## model_env

模型感知 env 自动化（R-08 B 类）

| 档位 | 状态 / 备注 |
|---|---|
| auto | production ModelRunner init 按 model_config setd |
| off | production 退回手敲 env（triage） |

路由变量：`HCU_MODEL_ENV`

## kv_pool_strides

KV 池 stride 张量构造（行为还原：torch2.11 不能从 numpy 标量推 dtype）

| 档位 | 状态 / 备注 |
|---|---|
| text-patch | production int(np.prod(...))，白名单 needle |

路由变量：（auto）

## dist_broadcast

TP init 广播 payload（行为还原：torch2.11 dlpack 只读 BufferError）

| 档位 | 状态 / 备注 |
|---|---|
| writable-copy | production common.np.frombuffer 代理返回可写副本 |

路由变量：（auto）

## triton_pdl

triton 编译期 PDL 指令剥离（行为还原：gfx936 无 griddepcontrol）

| 档位 | 状态 / 备注 |
|---|---|
| strip | production make_amdgcn 输出剔除 griddepcontrol.* 两行 |

路由变量：（auto）

## triton_libdevice

triton extra.cuda libdevice -> hip/OCML 重定向（is_cuda 语义选路修正）

| 档位 | 状态 / 备注 |
|---|---|
| hip-redirect | production __nv_* -> __ocml_*；hip 缺的名响亮 stub |

路由变量：（auto）

## flash_attention

fa3 backend 注意力面（kvcache 分页/统一窗 + varlen prefill）

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate 分页 triton 核（自研，fp32 累加）：decode L=1 + |
| emu | production chunked online-softmax torch 模拟（默认；全 |
| vendor | candidate DTK flash_attn 2.8.4 wheel（gfx93x ke |

路由变量：`HCU_FA_IMPL`

## sampling

top-k/top-p/min-p renorm+sample（§0.5 is_cuda sampler 面）

| 档位 | 状态 / 备注 |
|---|---|
| triton | planned 与 flashinfer renorm 核对拍后可选 |
| torch | production masked filter->renorm->multinomial，f |

路由变量：（auto）

## linear_attn_gdn

GDN/gated-delta-rule 线性注意力（sglang fla 面）

| 档位 | 状态 / 备注 |
|---|---|
| triton | production 上游 fla triton 核（auto 默认；gfx93x wave6 |
| aiter | validated DTK aiter CK/triton 核路由（fwd_h 状态递推 b |

路由变量：`HCU_LINEAR_ATTN_GDN`

## fused_moe

w8a8 fused MoE experts (standard dispatch, triton runner entry; deepep runner unaffected)

| 档位 | 状态 / 备注 |
|---|---|
| triton | production stock kernel; default fallback |
| aiter | validated DTK aiter_moe; blockwise->ASM tuned  |

路由变量：`HCU_FUSED_MOE_IMPL`

## w4a16_moe

W4A16 fused MoE experts (moe_wna16 AWQ/GPTQ-int4 line)

| 档位 | 状态 / 备注 |
|---|---|
| triton | production stock MoeWNA16 triton arm; auto answ |
| deepgemm | candidate R-05 bridge 2026-10-19: DTK deepgemm |

路由变量：`HCU_W4A16_MOE_IMPL`

## per_token_quant

per-token fp8/int8 quant (dynamic)

| 档位 | 状态 / 备注 |
|---|---|
| aiter | candidate R-09③ 2026-10-19 落位（quant_aiter 桥，au |
| lightop | planned per-token quant family |
| torch | production upstream natives（scaled_fp8_quant to |

路由变量：`HCU_QUANT_IMPL`

