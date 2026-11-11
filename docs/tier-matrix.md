# 算子档位矩阵（自动生成，勿手改）

> 生成：`python tools/gen_tier_matrix.py`（最近 2026-11-12）；事实源 = `BACKEND_CATALOG`（hcu_adapter/hcu_config.py）。
> L0 `impl-naming` 做新鲜度校验改表须随 catalog 同 commit（governance §3b）。
> 文件命名导航规则见 governance §3b；缺档 = 该行空白——那是实现排期的直接输入。

## cuda_graph

HIP driver guard for breakable prefill graph capture

| 档位 | 状态 / 备注 |
|---|---|
| prefill-bcg-driver-truth | candidate Breakable prefill uses the upstream  |

路由变量：（auto）

## vllm_flashmla_decode

vLLM 0.33 V4.1 native packed-FP8 sparse decode without BF16 KV gather

| 档位 | 状态 / 备注 |
|---|---|
| native | candidate Default gfx938 D512 OCP 584-byte cac |
| stock | candidate Explicit stock fallback: original de |

路由变量：`HCU_VLLM_FLASHMLA_DECODE`

## vllm_flashmla_prefill

vLLM 0.33 V4.1 gathered-BF16 sparse prefill via native FlashMLA

| 档位 | 状态 / 备注 |
|---|---|
| native | candidate Default gfx936/gfx938 D512 BF16 dens |
| stock | candidate Explicit stock fallback: unchanged o |

路由变量：`HCU_VLLM_FLASHMLA_PREFILL`
## vllm_fp4_indexer

V41 LightOp FP4 indexer with zero-copy 64-record cache pages

| 档位 | 状态 / 备注 |
|---|---|
| lightop | candidate Default: applies only when the model |
| stock | candidate Explicit stock fallback: existing FP |

路由变量：`HCU_VLLM_FP4_INDEXER_IMPL`

## vllm_fp4_packer

Plugin-owned vLLM 0.33 E2M1 arithmetic lowering

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Default: gfx938 Plugin packer; CPU c |
| stock | candidate Explicit stock fallback: audited ups |

路由变量：`HCU_VLLM_FP4_PACKER_IMPL`

## vllm_dsv41_channel_fp8

vLLM 0.33 V4.1 channel-wise FP8 config via the public quantization API

| 档位 | 状态 / 备注 |
|---|---|
| native | candidate Register hcu_dsv41_channel_fp8 witho |
| stock | candidate Do not register the distinct config; |

路由变量：`HCU_VLLM_DSV41_CHANNEL_FP8_IMPL`

## vllm_ct_w4a8

vLLM 0.33 V4.1 compressed-tensors W4A8 config via the public quantization API

| 档位 | 状态 / 备注 |
|---|---|
| native | candidate Register hcu_dsv41_ct_w4a8; HCU MoE  |
| stock | candidate Do not register the distinct config; |

路由变量：`HCU_VLLM_CT_W4A8_IMPL`

## vllm_pcp

vLLM 0.33 V4.1 PCP attention owned by the adapter kernel module

| 档位 | 状态 / 备注 |
|---|---|
| native | candidate Adapter-owned V41 PCP backends/index |
| stock | candidate Default: adapter builds no PCP class |

路由变量：`HCU_VLLM_PCP_IMPL`

## vllm_sparse_indexer

vLLM 0.33 C4 sparse indexer through public OOT and official helpers

| 档位 | 状态 / 备注 |
|---|---|
| native | candidate Opt-in gfx936 C4 H64/D128/top512; of |
| stock | candidate Default upstream dispatch; gfx936 ar |

路由变量：`HCU_VLLM_SPARSE_INDEXER_IMPL`

## vllm_mhc

vLLM 0.33 mHC via public OOT; AITER project with TileLang stage2 by default

| 档位 | 状态 / 备注 |
|---|---|
| aiter | candidate Default gfx936/gfx938 AITER project  |
| torch | candidate Opt-in gfx936/gfx938 official pre/po |
| stock | candidate No longer the adapter default; expli |

路由变量：`HCU_VLLM_MHC_IMPL`

## vllm_scaled_mm

vLLM native FP8/INT8 Linear execution through LightOp with official Triton fallback

| 档位 | 状态 / 备注 |
|---|---|
| triton | validated Preserve official providers without  |
| lightop | candidate Require LightOp wheel; native quanti |
| auto | candidate LightOp when installed: INT8 gfx936/ |

路由变量：（auto）

## vllm_fp8_linear

vLLM official dense FP8 fallback via public linear registration

| 档位 | 状态 / 备注 |
|---|---|
| official-triton | validated gfx936 and gfx938: official TritonFP |

路由变量：（auto）

## vllm_allreduce

vLLM official AR constructor compatibility; separate from historical SGLang AR

| 档位 | 状态 / 备注 |
|---|---|
| official-aiter | candidate HCU ABI/copy-in/limits only; CPU con |
| official-fallback | candidate AITER dispatch OFF; preserves offici |

路由变量：`1`/`0`

## vllm_allreduce_admission

vLLM AR-only gate compatibility; global AITER and real architecture unchanged

| 档位 | 状态 / 备注 |
|---|---|
| official-aiter-gate | candidate Existing HCU kill/resolver policy; a |

路由变量：（auto）

## silu_and_mul

gated activation out = silu(x[:,:d]) * x[:,d:]

| 档位 | 状态 / 备注 |
|---|---|
| triton | production default (auto); fp32 accumulate |
| lightop | validated lightop.fuse_silu_and_mul (input, ou |
| torch | production reference path, eager debug |

路由变量：`HCU_SILU_MUL_IMPL`

## rms_norm

RMSNorm (plain); the fused-add family is its own op below

| 档位 | 状态 / 备注 |
|---|---|
| triton | production default |
| aiter | candidate aiter.rmsnorm_forward(x,w,eps,traini |
| lightop | candidate Public BF16 plain RMSNorm; other dty |
| torch | production reference path |

路由变量：`HCU_RMS_NORM_IMPL`

## fused_add_rms_norm

residual += input; input = rmsnorm(residual)*w, in-place

| 档位 | 状态 / 备注 |
|---|---|
| triton | production default (auto) |
| lightop | validated lightop.fused_add_rms_norm(x, residu |
| torch | production reference path |

路由变量：`HCU_FUSED_ADD_RMS_IMPL`

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

## flashmla_kv_indices

FlashMLA static-pool page-table import compatibility

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Reuse SGLang's existing page64 kerne |

路由变量：（auto）

## w4a8_contiguous_gemm

DTK W4A8 HIPC code-object ABI compatibility

| 档位 | 状态 / 备注 |
|---|---|
| dtk | candidate Optional gfx936 COV5 extension for p |

路由变量：（auto）

## w4a16_moe

W4A16 fused MoE experts (moe_wna16 AWQ/GPTQ-int4 line)

| 档位 | 状态 / 备注 |
|---|---|
| triton | production stock MoeWNA16 triton arm; auto answ |
| deepgemm | candidate R-05 bridge 2026-10-19: DTK deepgemm |

路由变量：`HCU_W4A16_MOE_IMPL`

## ct_channel_w4a8_components

Channel INT4 / token INT8 components and explicit BF16 DeepEP LL route candidate; model acceptance pending

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate w4a8_int8_triton: masked absmax/127. |

路由变量：（auto）

## int8_linear_gemm

SGLang CT and explicit W8A8 INT8 linear scaled GEMM

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Existing sgl_kernel.gemm_int8 ABI; b |

路由变量：（auto）

## hadamard

SGLang DSA indexer rotate_activation Hadamard transform

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Radix-2 butterfly (reshape/permute/s |
| emu | candidate Torch Sylvester matrix multiply refe |

路由变量：`HCU_HADAMARD_IMPL`

## per_token_quant

per-token fp8/int8 quant (dynamic)

| 档位 | 状态 / 备注 |
|---|---|
| aiter | candidate R-09③ 2026-10-19 落位（quant_aiter 桥，au |
| lightop | planned per-token quant family |
| torch | production upstream natives（scaled_fp8_quant to |

路由变量：`HCU_QUANT_IMPL`

## deep_gemm.cublaslt_gemm_nt

deep_gemm.cublaslt_gemm_nt

| 档位 | 状态 / 备注 |
|---|---|
| native | candidate BF16 contiguous aligned-K uses quali |

路由变量：（auto）

## tile_kernels.quant.swiglu_forward

tile_kernels.quant.swiglu_forward

| 档位 | 状态 / 备注 |
|---|---|
| lightop | candidate Explicit LightOp b2f8c9d; preserve r |
| torch | candidate Standalone BF16/FP32 V4.1 ViT activa |

路由变量：（auto）

## tile_kernels.utils.dtype_to_str

tile_kernels.utils.dtype_to_str

| 档位 | 状态 / 备注 |
|---|---|
| native | candidate Pinned host map: FP32/BF16/E4M3/int8 |

路由变量：（auto）

## deep_select.topk

deep_select.topk

| 档位 | 状态 / 备注 |
|---|---|
| lightop | candidate Explicit LightOp b2f8c9d; preserve r |
| torch | candidate Pinned GPU top-k, native stable sort |

路由变量：（auto）

## tilelang_ops.gather_tl

tilelang_ops.gather_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Pinned int8 factory contract; no mod |
| torch | candidate Pinned int8 factory contract; no mod |

路由变量：（auto）

## tilelang_ops.norm_tl

tilelang_ops.norm_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate BF16/FP32, FP8 per32/per128; no mode |
| lightop | candidate Explicit LightOp b2f8c9d; preserve r |
| torch | candidate BF16/FP32, FP8 per32/per128; no mode |

路由变量：（auto）

## tilelang_ops.expand_swa_index_tl

tilelang_ops.expand_swa_index_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Exact int32 contract incl. wraparoun |
| torch | candidate Exact int32 contract incl. wraparoun |

路由变量：（auto）

## tilelang_ops.fused_norm_rope_hadamard_cast_store_kvcache_tl

tilelang_ops.fused_norm_rope_hadamard_cast_store_kvcache_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Norm via norm_tl provider, torch RoP |
| lightop | candidate Native norm/RoPE and block32/128 FP8 |
| torch | candidate Norm via norm_tl provider, torch RoP |

路由变量：（auto）

## tile_kernels.transform.apply_rotary

tile_kernels.transform.apply_rotary

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate In-place FP32 fma rotation, NeoX/int |
| lightop | candidate Explicit LightOp b2f8c9d; preserve r |

路由变量：（auto）

## tile_kernels.quant.per_token_cast_and_cast_back

tile_kernels.quant.per_token_cast_and_cast_back

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate In-place E4M3/E2M1 round trip with F |

路由变量：（auto）

## tilelang_ops.load_paged_kv_cache_tl

tilelang_ops.load_paged_kv_cache_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Torch gather/scatter of compact segm |
| torch | candidate Torch gather/scatter of compact segm |

路由变量：（auto）

## tilelang_ops.store_kv_cache_tl

tilelang_ops.store_kv_cache_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Torch quantize+scatter mirroring the |
| torch | candidate Torch quantize+scatter mirroring the |

路由变量：（auto）

## flash_mla.fused_norm_rope_attn_rope_cast.decode

flash_mla.fused_norm_rope_attn_rope_cast.decode

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Torch sparse MLA decode on V4.1 FP8  |
| native | candidate Explicit code-object-v4 1287a01 plus |

路由变量：（auto）

## flash_mla.fused_norm_rope_attn_rope_cast.prefill

flash_mla.fused_norm_rope_attn_rope_cast.prefill

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Torch sparse MLA prefill mirroring v |
| native | candidate Explicit code-object-v4 1287a01 plus |

路由变量：（auto）

## tile_kernels.quant.swiglu_forward_and_per_token_cast

tile_kernels.quant.swiglu_forward_and_per_token_cast

| 档位 | 状态 / 备注 |
|---|---|
| lightop | candidate Explicit LightOp b2f8c9d; preserve r |
| torch | candidate FP32 SwiGLU (+clamp, topk weight) th |

路由变量：（auto）

## tilelang_ops.compress_and_update_state_prefill_tl

tilelang_ops.compress_and_update_state_prefill_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Torch gather/softmax/scatter on devi |
| torch | candidate Torch gather/softmax/scatter on devi |

路由变量：（auto）

## tilelang_ops.compress_and_update_state_decode_tl

tilelang_ops.compress_and_update_state_decode_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Torch gather/softmax/scatter on devi |
| torch | candidate Torch gather/softmax/scatter on devi |

路由变量：（auto）

## tilelang_ops.load_kv_index_tl

tilelang_ops.load_kv_index_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Torch index op mirroring the upstrea |
| torch | candidate Torch index op mirroring the upstrea |

路由变量：（auto）

## tilelang_ops.kv_index_cat_tl

tilelang_ops.kv_index_cat_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Torch index op mirroring the upstrea |
| torch | candidate Torch index op mirroring the upstrea |

路由变量：（auto）

## tilelang_ops.convert_kv_index_tl

tilelang_ops.convert_kv_index_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Torch index op mirroring the upstrea |
| torch | candidate Torch index op mirroring the upstrea |

路由变量：（auto）

## tilelang_ops.candidate_block_max_tl

tilelang_ops.candidate_block_max_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Torch index op mirroring the upstrea |
| torch | candidate Torch index op mirroring the upstrea |

路由变量：（auto）

## tilelang_ops.get_candidate_compact_lens_tl

tilelang_ops.get_candidate_compact_lens_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Torch index op mirroring the upstrea |
| torch | candidate Torch index op mirroring the upstrea |

路由变量：（auto）

## tilelang_ops.restore_candidate_indices_inplace_tl

tilelang_ops.restore_candidate_indices_inplace_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Torch index op mirroring the upstrea |
| torch | candidate Torch index op mirroring the upstrea |

路由变量：（auto）

## tile_kernels.quant.batched_transpose_weight_sf

tile_kernels.quant.batched_transpose_weight_sf

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate GPU UE8M0 int32, block32/block128, g |

路由变量：（auto）

## tile_kernels.quant.per_token_cast

tile_kernels.quant.per_token_cast

| 档位 | 状态 / 备注 |
|---|---|
| lightop | candidate Explicit LightOp b2f8c9d; preserve r |
| torch | candidate Deterministic BF16/FP32 to E4M3 or p |

路由变量：（auto）

## deep_gemm.bf16_gemm_nt

deep_gemm.bf16_gemm_nt

| 档位 | 状态 / 备注 |
|---|---|
| lightop | candidate Contiguous BF16 inputs, K divisible  |

路由变量：（auto）

## deep_gemm.transform_sf_into_required_layout

deep_gemm.transform_sf_into_required_layout

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate SM100 packed UE8M0 ABI; no PSUM gaps |

路由变量：（auto）

## deep_gemm.set_num_sms

deep_gemm.set_num_sms

| 档位 | 状态 / 备注 |
|---|---|
| host | candidate Host-side vendor knob mirrored from  |

路由变量：（auto）

## deep_gemm.get_num_sms

deep_gemm.get_num_sms

| 档位 | 状态 / 备注 |
|---|---|
| host | candidate Host-side vendor knob mirrored from  |

路由变量：（auto）

## deep_gemm.set_mk_alignment_for_contiguous_layout

deep_gemm.set_mk_alignment_for_contiguous_layout

| 档位 | 状态 / 备注 |
|---|---|
| host | candidate Host-side vendor knob mirrored from  |

路由变量：（auto）

## deep_gemm.get_mk_alignment_for_contiguous_layout

deep_gemm.get_mk_alignment_for_contiguous_layout

| 档位 | 状态 / 备注 |
|---|---|
| host | candidate Host-side vendor knob mirrored from  |

路由变量：（auto）

## deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout

deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout

| 档位 | 状态 / 备注 |
|---|---|
| host | candidate Host-side vendor knob mirrored from  |

路由变量：（auto）

## deep_gemm.get_tma_aligned_size

deep_gemm.get_tma_aligned_size

| 档位 | 状态 / 备注 |
|---|---|
| host | candidate Host-side vendor knob mirrored from  |

路由变量：（auto）

## tile_kernels.set_num_sms

tile_kernels.set_num_sms

| 档位 | 状态 / 备注 |
|---|---|
| host | candidate Host-side vendor knob mirrored from  |

路由变量：（auto）

## tile_kernels.get_num_sms

tile_kernels.get_num_sms

| 档位 | 状态 / 备注 |
|---|---|
| host | candidate Host-side vendor knob mirrored from  |

路由变量：（auto）

## tile_kernels.set_token_alignment

tile_kernels.set_token_alignment

| 档位 | 状态 / 备注 |
|---|---|
| host | candidate Host-side vendor knob mirrored from  |

路由变量：（auto）

## tile_kernels.get_token_alignment

tile_kernels.get_token_alignment

| 档位 | 状态 / 备注 |
|---|---|
| host | candidate Host-side vendor knob mirrored from  |

路由变量：（auto）

## deep_gemm.fp8_fp4_gemm_nt

deep_gemm.fp8_fp4_gemm_nt

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate FP8 E4M3 operands only; exact BF16/F |

路由变量：（auto）

## deep_gemm.fp8_einsum

deep_gemm.fp8_einsum

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate 'bhr,hdr->bhd' only; exact dequant + |

路由变量：（auto）

## deep_gemm.mega_mhc

deep_gemm.mega_mhc

| 档位 | 状态 / 备注 |
|---|---|
| lightop | candidate Explicit LightOp b2f8c9d; preserve r |
| torch | candidate Composed torch mHC post+mixes+pre+RM |

路由变量：（auto）

## deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous

deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Route A fast tier: per-BLOCK_M group |
| torch | candidate Route A correctness tier: exact BF16 |

路由变量：（auto）

## deep_gemm.fp8_fp4_mqa_logits

deep_gemm.fp8_fp4_mqa_logits

| 档位 | 状态 / 备注 |
|---|---|
| lightop | candidate D128 H8/16/32 MXFP4 uses packed Ligh |
| torch | candidate Dense prefill indexer logits (compre |

路由变量：（auto）

## deep_gemm.fp8_fp4_sparse_mqa_logits

deep_gemm.fp8_fp4_sparse_mqa_logits

| 档位 | 状态 / 备注 |
|---|---|
| lightop | candidate D128 H8/16/32 MXFP4 uses packed Ligh |
| torch | candidate Sparse (candidate-block) indexer log |

路由变量：（auto）

## deep_gemm.get_sparse_mqa_logits_metadata

deep_gemm.get_sparse_mqa_logits_metadata

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Sparse (candidate-block) indexer log |

路由变量：（auto）

## deep_gemm.fp8_fp4_paged_mqa_logits

deep_gemm.fp8_fp4_paged_mqa_logits

| 档位 | 状态 / 备注 |
|---|---|
| lightop | candidate D128 H8/16/32 MXFP4 uses packed Ligh |
| torch | candidate Sparse (candidate-block) indexer log |

路由变量：（auto）

## deep_gemm.get_paged_mqa_logits_metadata

deep_gemm.get_paged_mqa_logits_metadata

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Sparse (candidate-block) indexer log |

路由变量：（auto）

## deep_gemm.fp8_fp4_paged_sparse_mqa_logits

deep_gemm.fp8_fp4_paged_sparse_mqa_logits

| 档位 | 状态 / 备注 |
|---|---|
| lightop | candidate D128 H8/16/32 MXFP4 uses packed Ligh |
| torch | candidate Sparse (candidate-block) indexer log |

路由变量：（auto）

## deep_gemm.get_paged_sparse_mqa_logits_metadata

deep_gemm.get_paged_sparse_mqa_logits_metadata

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Sparse (candidate-block) indexer log |

路由变量：（auto）

## deep_gemm.bf16_mega_gate

deep_gemm.bf16_mega_gate

| 档位 | 状态 / 备注 |
|---|---|
| lightop | candidate Explicit LightOp b2f8c9d; preserve r |
| torch | candidate FP32 gate GEMM + graph-safe sqrtsoft |

路由变量：（auto）

## tile_kernels.moe.moe_topk_gate

tile_kernels.moe.moe_topk_gate

| 档位 | 状态 / 备注 |
|---|---|
| lightop | candidate Explicit LightOp b2f8c9d; preserve r |
| torch | candidate Graph-safe rewrite of tile_kernels t |

路由变量：（auto）

## deep_gemm.tf32_hc_prenorm_gemm

deep_gemm.tf32_hc_prenorm_gemm

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate BF16 x FP32 prenorm GEMM, contiguous |

路由变量：（auto）

## tile_kernels.mhc.expand_kernel.expand_to_mhc_fwd

tile_kernels.mhc.expand_kernel.expand_to_mhc_fwd

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Pinned mHC forward contract; eager c |

路由变量：（auto）

## tile_kernels.mhc.pre_apply_mix_kernel.mhc_pre_apply_mix_fwd

tile_kernels.mhc.pre_apply_mix_kernel.mhc_pre_apply_mix_fwd

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Pinned mHC forward contract; eager c |

路由变量：（auto）

## tile_kernels.mhc.head_compute_mix_kernel.mhc_head_compute_mix_fwd

tile_kernels.mhc.head_compute_mix_kernel.mhc_head_compute_mix_fwd

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Pinned mHC forward contract; eager c |

路由变量：（auto）

## tile_kernels.mhc.pre_split_mixes_kernel.mhc_pre_split_mixes_fwd

tile_kernels.mhc.pre_split_mixes_kernel.mhc_pre_split_mixes_fwd

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Pinned mHC forward contract; eager c |

路由变量：（auto）

## tile_kernels.mhc.sinkhorn_kernel.mhc_sinkhorn_fwd

tile_kernels.mhc.sinkhorn_kernel.mhc_sinkhorn_fwd

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Pinned mHC forward contract; eager c |

路由变量：（auto）

## tile_kernels.mhc.post_kernel.mhc_post_fwd

tile_kernels.mhc.post_kernel.mhc_post_fwd

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Pinned mHC forward contract; eager c |

路由变量：（auto）

## tile_kernels.mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd

tile_kernels.mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Pinned mHC forward contract; eager c |

路由变量：（auto）

## tile_kernels.mhc.norm_fn_kernel.mhc_reduce_partials_and_rmsnorm_fwd

tile_kernels.mhc.norm_fn_kernel.mhc_reduce_partials_and_rmsnorm_fwd

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Pinned mHC forward contract; eager c |

路由变量：（auto）

## tile_kernels.mhc.pre_big_fuse_kernel.mhc_pre_big_fuse_fwd

tile_kernels.mhc.pre_big_fuse_kernel.mhc_pre_big_fuse_fwd

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Pinned mHC forward contract; eager c |

路由变量：（auto）

## flash_mla.fused_norm_rope_attn_rope_cast.permute_q_b_proj

flash_mla.fused_norm_rope_attn_rope_cast.permute_q_b_proj

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Byte-preserving initialization permu |

路由变量：（auto）

## flash_mla.fused_norm_rope_attn_rope_cast.permute_wv_proj

flash_mla.fused_norm_rope_attn_rope_cast.permute_wv_proj

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Byte-preserving initialization permu |

路由变量：（auto）

## deep_ep.BucketBuffer

deep_ep.BucketBuffer

| 档位 | 状态 / 备注 |
|---|---|
| rccl | candidate RCCL session storage with pinned 2Mi |

路由变量：（auto）

## deep_ep.BucketSession

deep_ep.BucketSession

| 档位 | 状态 / 备注 |
|---|---|
| rccl | candidate RCCL session storage with pinned 2Mi |

路由变量：（auto）

## deep_ep.get_num_allocation_alignment

deep_ep.get_num_allocation_alignment

| 档位 | 状态 / 备注 |
|---|---|
| rccl | candidate RCCL session storage with pinned 2Mi |

路由变量：（auto）

## deep_ep.get_num_rdma_alignment

deep_ep.get_num_rdma_alignment

| 档位 | 状态 / 备注 |
|---|---|
| rccl | candidate RCCL session storage with pinned 2Mi |

路由变量：（auto）

## flash_attn.cute.interface._flash_attn_fwd

flash_attn.cute.interface._flash_attn_fwd

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Pinned ViT noncausal varlen Q/K/V ca |

路由变量：（auto）

## deep_ep.EngramBuffer

deep_ep.EngramBuffer

| 档位 | 状态 / 备注 |
|---|---|
| rccl_gpu | candidate GPU tables, eager, byte-exact collec |
| rccl_gpu_graph | candidate Explicit graph candidate for GPU tab |

路由变量：（auto）

## deep_ep.EPBuffer

deep_ep.EPBuffer

| 档位 | 状态 / 备注 |
|---|---|
| dtk_v1_expanded | candidate Single-host eager expanded dispatch; |
| dtk_v1_expanded_graph | candidate Explicit single-host graph candidate |

路由变量：（auto）

## deep_ep.get_physical_domain_size

deep_ep.get_physical_domain_size

| 档位 | 状态 / 备注 |
|---|---|
| lsa1 | candidate Reports LSA domain 1 on DTK2610 (no  |

路由变量：（auto）

## deep_ep.get_logical_domain_size

deep_ep.get_logical_domain_size

| 档位 | 状态 / 备注 |
|---|---|
| lsa1 | candidate Reports LSA domain 1 on DTK2610 (no  |

路由变量：（auto）

## tilelang_ops.fused_sample_tl

tilelang_ops.fused_sample_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Mirrors upstream fused_sample_ref/ra |
| torch | candidate Mirrors upstream fused_sample_ref/ra |

路由变量：（auto）

## tilelang_ops.random_sampling_tl

tilelang_ops.random_sampling_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Mirrors upstream fused_sample_ref/ra |
| torch | candidate Mirrors upstream fused_sample_ref/ra |

路由变量：（auto）

## tilelang_ops.prepare_v4_decode_metadata_tl

tilelang_ops.prepare_v4_decode_metadata_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Fused device-only Triton metadata; 5 |
| torch | candidate Device-only torch reference for deco |

路由变量：（auto）

## tilelang_ops.copy_block_tables_tl

tilelang_ops.copy_block_tables_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Fused device-only Triton metadata; 5 |
| torch | candidate Device-only torch reference for deco |

路由变量：（auto）

## tilelang_ops.gather_input_ids_zos_tl

tilelang_ops.gather_input_ids_zos_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Fused device-only Triton metadata; 5 |
| torch | candidate Device-only torch reference for deco |

路由变量：（auto）

## tilelang_ops.gather_engram_tokens_zos_tl

tilelang_ops.gather_engram_tokens_zos_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Fused device-only Triton metadata; 5 |
| torch | candidate Device-only torch reference for deco |

路由变量：（auto）

## tilelang_ops.compute_seqlens_k_zos_tl

tilelang_ops.compute_seqlens_k_zos_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Fused device-only Triton metadata; 5 |
| torch | candidate Device-only torch reference for deco |

路由变量：（auto）

## tilelang_ops.prepare_verification_metadata_tl

tilelang_ops.prepare_verification_metadata_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only torch mirror of the upst |
| torch | candidate Device-only torch mirror of the upst |

路由变量：（auto）

## tilelang_ops.expand_sampling_params_tl

tilelang_ops.expand_sampling_params_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only torch mirror of the upst |
| torch | candidate Device-only torch mirror of the upst |

路由变量：（auto）

## tilelang_ops.map_lookup_tl

tilelang_ops.map_lookup_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only torch mirror of the upst |
| torch | candidate Device-only torch mirror of the upst |

路由变量：（auto）

## tilelang_ops.remap_draft_topk_tl

tilelang_ops.remap_draft_topk_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only torch mirror of the upst |
| torch | candidate Device-only torch mirror of the upst |

路由变量：（auto）

## tilelang_ops.get_accept_result_tl

tilelang_ops.get_accept_result_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only torch mirror of the upst |
| torch | candidate Device-only torch mirror of the upst |

路由变量：（auto）

## tilelang_ops.finalize_verification_tl

tilelang_ops.finalize_verification_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only torch mirror of the upst |
| torch | candidate Device-only torch mirror of the upst |

路由变量：（auto）

## tilelang_ops.sum_smaller_probs_tl

tilelang_ops.sum_smaller_probs_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only torch mirror of the upst |
| torch | candidate Device-only torch mirror of the upst |

路由变量：（auto）

## tile_kernels.engram.fused_weight

tile_kernels.engram.fused_weight

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Pinned initialization contract; no m |

路由变量：（auto）

## tile_kernels.torch.engram.make_offsets

tile_kernels.torch.engram.make_offsets

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Pinned initialization contract; no m |

路由变量：（auto）

## tile_kernels.engram.engram_gate_fwd

tile_kernels.engram.engram_gate_fwd

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate SGLang 7b91060 fused inference gate  |

路由变量：（auto）

## tile_kernels.engram.engram_hash

tile_kernels.engram.engram_hash

| 档位 | 状态 / 备注 |
|---|---|
| torch | candidate Hash requires nonnegative INT64-safe |

路由变量：（auto）

## tilelang_ops.confidence_head_fused_tl

tilelang_ops.confidence_head_fused_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only MTP/state/constraint con |
| torch | candidate Device-only MTP/state/constraint con |

路由变量：（auto）

## tilelang_ops.dflash_input_store_tl

tilelang_ops.dflash_input_store_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only MTP/state/constraint con |
| torch | candidate Device-only MTP/state/constraint con |

路由变量：（auto）

## tilelang_ops.copy_dflash_decode_input_tl

tilelang_ops.copy_dflash_decode_input_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only MTP/state/constraint con |
| torch | candidate Device-only MTP/state/constraint con |

路由变量：（auto）

## tilelang_ops.store_mtp_info_tl

tilelang_ops.store_mtp_info_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only MTP/state/constraint con |
| torch | candidate Device-only MTP/state/constraint con |

路由变量：（auto）

## tilelang_ops.compute_seqlens_q_by_confidence_rates_tl

tilelang_ops.compute_seqlens_q_by_confidence_rates_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only MTP/state/constraint con |
| torch | candidate Device-only MTP/state/constraint con |

路由变量：（auto）

## tilelang_ops.get_fused_dflash_get_input_tl

tilelang_ops.get_fused_dflash_get_input_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only MTP/state/constraint con |
| torch | candidate Device-only MTP/state/constraint con |

路由变量：（auto）

## tilelang_ops.select_state_slice_tl

tilelang_ops.select_state_slice_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only MTP/state/constraint con |
| torch | candidate Device-only MTP/state/constraint con |

路由变量：（auto）

## tilelang_ops.load_state_to_cache_tl

tilelang_ops.load_state_to_cache_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only MTP/state/constraint con |
| torch | candidate Device-only MTP/state/constraint con |

路由变量：（auto）

## tilelang_ops.apply_token_bitmask_by_step_inplace_tl

tilelang_ops.apply_token_bitmask_by_step_inplace_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only MTP/state/constraint con |
| torch | candidate Device-only MTP/state/constraint con |

路由变量：（auto）

## tilelang_ops.detect_outlier_and_abort_tl

tilelang_ops.detect_outlier_and_abort_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only MTP/state/constraint con |
| torch | candidate Device-only MTP/state/constraint con |

路由变量：（auto）

## tilelang_ops.batch_downsample_tl

tilelang_ops.batch_downsample_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only vision/CP metadata contr |
| torch | candidate Device-only vision/CP metadata contr |

路由变量：（auto）

## tilelang_ops.prepare_cp_mirror_tl

tilelang_ops.prepare_cp_mirror_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only vision/CP metadata contr |
| torch | candidate Device-only vision/CP metadata contr |

路由变量：（auto）

## tilelang_ops.replace_input_embeds_tl

tilelang_ops.replace_input_embeds_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only vision/CP metadata contr |
| torch | candidate Device-only vision/CP metadata contr |

路由变量：（auto）

## tilelang_ops.rgb_pad_to_patches_tl

tilelang_ops.rgb_pad_to_patches_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only vision/CP metadata contr |
| torch | candidate Device-only vision/CP metadata contr |

路由变量：（auto）

## tilelang_ops.yoco_cp_boundary_fused_tl

tilelang_ops.yoco_cp_boundary_fused_tl

| 档位 | 状态 / 备注 |
|---|---|
| triton | candidate Device-only vision/CP metadata contr |
| torch | candidate Device-only vision/CP metadata contr |

路由变量：（auto）

## runtime.device_visibility

runtime.device_visibility

| 档位 | 状态 / 备注 |
|---|---|
| ordinal_mask | candidate Before GPU initialization, align HIP |

路由变量：（auto）

## worker.driver_exposure

worker.driver_exposure

| 档位 | 状态 / 备注 |
|---|---|
| rccl_engram | candidate RCCL Engram has no NVIDIA peer-memor |

路由变量：（auto）

## platform.device_inventory

platform.device_inventory

| 档位 | 状态 / 备注 |
|---|---|
| drm | candidate DRM cardinality through GPU discover |

路由变量：（auto）

## runtime.upstream_defaults

runtime.upstream_defaults

| 档位 | 状态 / 备注 |
|---|---|
| compatible | candidate Default GPU Engram storage and non-b |

路由变量：（auto）

## runtime.configuration

runtime.configuration

| 档位 | 状态 / 备注 |
|---|---|
| file | candidate Strict JSON/YAML installer options;  |

路由变量：（auto）

## runtime.dump

runtime.dump

| 档位 | 状态 / 备注 |
|---|---|
| json | candidate Atomic per-PID report of actual bind |

路由变量：（auto）

## worker.cpu_affinity

worker.cpu_affinity

| 档位 | 状态 / 备注 |
|---|---|
| inherited_mask | candidate Worker and SO builder share inherite |

路由变量：（auto）

## worker.bootstrap.context

worker.bootstrap.context

| 档位 | 状态 / 备注 |
|---|---|
| inherited | candidate Internal provider context propagated |

路由变量：（auto）

## worker.bootstrap.import_hook

worker.bootstrap.import_hook

| 档位 | 状态 / 备注 |
|---|---|
| lazy | candidate No torch import at Python startup; i |

路由变量：（auto）

## worker.runtime

worker.runtime

| 档位 | 状态 / 备注 |
|---|---|
| hip_eager | candidate Eager by default; explicit allow_cud |

路由变量：（auto）

## platform.bf16_tflops_limit

platform.bf16_tflops_limit

| 档位 | 状态 / 备注 |
|---|---|
| guard | candidate MFU denominator only; BW1101 peak no |

路由变量：（auto）

## vendors.import_surface

vendors.import_surface

| 档位 | 状态 / 备注 |
|---|---|
| stub | candidate Missing fused MLA, HC GEMM and visio |

路由变量：（auto）

## tile_kernels.import_surface

tile_kernels.import_surface

| 档位 | 状态 / 备注 |
|---|---|
| stub | candidate Pinned production imports; only sepa |

路由变量：（auto）

## deepinfer._C

deepinfer._C

| 档位 | 状态 / 备注 |
|---|---|
| hip_staged | candidate Pinned source build; external GCC13  |

路由变量：（auto）

