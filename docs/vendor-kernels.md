# DTK vendor 算子库 API 梳理（实机内省，优化参考手册）

> 2026-10-22 在 nmz26 `dsv41f-dspark` 容器（dtk2604 栈）对 8 个 vendor wheel
> 做导入级 API 内省整理。目的：回答"还有哪些算子可用、能让性能更快"，并把
> 每个库的 API 面、我们栈的集成现状、与当前热点的映射关系沉淀成一张可查的账。
> 取证脚本 `tmp_fetch/introspect_libs.py` / `introspect2.py`（python
> importlib+pkgutil 遍历，不执行任何核），原始 JSON 存
> `tmp_fetch/introspect_libs.json` / `introspect2.json`。

## 0. wheel 清单与总览

| 库 | wheel 版本（git 后缀=分叉点） | 定位 | sglang 树接入 | adapter 接入 | 一句话机会 |
|---|---|---|---|---|---|
| aiter | 0.1.7+dtk2604.**ga76f3d** | AMD/HCU 通用算子+通信大库（51 预编 .so 零编译） | 174 处（深度集成） | AR/AR+Norm/quant/moe 桥在用 | MHC×AR 融合、qknorm-AR-rope 三融、QR 小 AR、moe_c marlin fp4 |
| lightop | 0.6.0+dtk2604.gaa4a34 | HCU 专属全家桶，**带 dsv41 命名算子** | **0 处** | silu_and_mul + fused_add_rms_norm（含 gemma）candidate 档 | norm+rope+quant 前融合、silu_clamp×quant、moe gate v41、fp4 mqa logits |
| boltops | 0.1.0+dtk2604.g1f36f1 | HCU 新核库（14 子包，覆盖 mhc/ihc/dsa/mqa/gemm/moe） | **0 处（全未接入）** | 0 处 | MHC vendor 链、step4_dsa 全链、conv1d、a8w8/afp4wfp4 gemm |
| flash_mla | 1.2.0+dtk2604.g29ff42 | MLA decode/稀疏 attention（DeepSeek 系） | 10 处 | dsa decode 档在用 | fp8 变体、q_nope_pe 变体 |
| flash_attn | 2.8.4+dtk2604.g007e38 | FA2 全家 + sparse_gqa/fa_sparse 扩展 | sgl-kernel 内 fa3 emu 桥 | HCU_FA_IMPL vendor 档 | sparse_gqa fp8、fa_sparse fp8 变体 |
| deepgemm（连写） | 2.1.0+dtk2604.g3c303b | grouped GEMM/MoE GEMM（DTK fork，桥别名 deep_gemm） | deep_gemm 90 处 | HCU_DEEPGEMM_BRIDGE=1（EP 必带） | wfp4a16 grouped、tf32_hc_pernorm（MHC 链） |
| flash_kda | 0.0.1+dtk2604.g9a698c | KDA gated-delta attention（fwd+cp 布局） | 1 处 | — | KDA 模型线备用 |
| tokenspeed_mla | 0.1.8 | MLA decode/prefill（fp16/fp8）+kv pack 量化 | 1 处 | — | MLA fp8 备用路径 |

版本纪律：aiter 的 JIT cache 以**操作名键**存 .so，跨 wheel 播种会遮蔽重编译
（bootstrap 4c 成对铁律；现役两镜像 aiter 已分叉 ga76f3d / g722998）。引用
任何 vendor 核时文档必须带 wheel 版本后缀。

## 1. aiter（0.1.7+dtk2604.ga76f3d，289 顶层函数 / 12 类 / 51 预编 .so）

按族分类（只列与推理栈相关的，全量见内省 JSON）：

| 族 | 代表 API | 备注 |
|---|---|---|
| norm | `rms_norm` `fused_add_rms_norm_cu` `rms_norm_cu` `head_rms_norm` `rmsnorm2d_fwd*`（add/smoothquant/dynamicquant 变体）`layer_norm` `layernorm2d_fwd*` | 2d 族带 quant 融合 |
| quant | `pertoken_quant` `per_token_quant_hip/triton` `per_token_group_quant_fp8` `per_tensor_quant*` `per_1x32_f4_quant*` `moe_swiglu_dynamic_quant(_wrapper)` `static/dynamic_per_token_scaled_quant` `dynamic_per_group_scaled_quant_fp4` | fp4 (1x32) 量化在库 |
| activation | `silu_and_mul` `scaled_silu_and_mul` `gelu_and_mul(_eager)` `gelu_tanh_and_mul` `relu2` `swiglu_variant` `fuse_silu_mul_quant` `fuse_silu_mul_fp8_quant(_ep)` `fuse_silu_mul_per_token_quant` `fuse_silu_and_mul_ep` | ep 变体=EP 布局直写 |
| **custom AR/RS/AG** | `init_custom_ar` `configure_small_ar` `all_reduce` `reduce_scatter(_dim/_8bit)` `all_gather_reg/unreg` `register_graph_buffers` `get_graph_buffer_ipc_meta` | GraphSafe 图内路径（现役 HCU_AITER_AR auto 档） |
| **AR 融合族** | `fused_allreduce_rmsnorm(_pad/_quant/_quant_per_group/_mxfp4_quant)` `fused_qknorm_allreduce` **`fused_qknorm_allreduce_rope`** **`fused_allreduce_mhc_post_only/_one_stage/_split`** + `launch_*` 对应 | norm/rope/MHC 与 AR 的多级融合——我们拍瓶颈的直接武器库 |
| QR 快路 | `init_custom_qr` `qr_all_reduce` `qr_all_reduce_rmsnorm` `qr_max_size` | 小张量 AR 低延迟路 |
| MoE-C 族 | `moe_c_moe_gemm_marlin_w8a8(_fp8/_tensorwise)` `_w4a8` `_w4a16` **`_wfp4a16`** **`_wfp4a8_channelwise/_groupwise(/_qgroup)`** `_w8a16` `_w16a16(_asm)`；`moe_c_moe_w8a8_gemm_block_wise(_kernel2/_fp8)`；`moe_c_topk_softmax` `moe_c_silu_and_mul` **`moe_c_situ_glu`** `moe_c_gelu*` `moe_c_moe_sum(_opt_v2)` `moe_c_(sgl_)moe_align_block_size` | marlin 后端 quant 覆盖最全；**fp4 专家 GEMM 两种 scale 制式都有** |
| MoE asm/ck | `asm_fmoe_stage1/2` `asm_fmoe_a8` `asm_moe_get_solutions` `ck_moe` `ck_shuffle_moe` `ck_moe_stage_1/2` `ck_moe_per_token_quant` `moe_sorting_fwd` | asm/ck 双后端 |
| EP | `ep_scatter` `ep_gather` `ep_build_m_indices` `ep_fused_quant_scatter` `ep_fused_fp8_quant_scatter` `ep_fused_smooth_quant_scatter` | scatter+量化融合 |
| routing | `topk_softmax` `moe_sum` `asm_moe_sum` `(sgl_)moe_align_block_size` | |
| rope | `rope_fwd_impl` `rope_cached_fwd_impl` `rope_cached_positions(_offsets)_fwd_impl` `rope_2c_*` `rope_thd_fwd_impl` `RoPE/RoPECached/RoPETHD/RoPE2D` 类 | cached/positions 变体省 cos/sin 重算 |
| 其它 | `awq_gemm_asm(_tuning)` `awq_dq_asm` `partial_transpose` `wvSpltK` `LLMM1` `OpusK3AttnResKernel` `blockwise_scaled_cast_transpose` 系 `bias*` | LLMM1=低精 matmul 专用 |
| 子包 | `aiter.mla` `aiter.paged_attn` `aiter.dist` `aiter.fused_moe{,_asm,_ck,_c}` `aiter.tuned_gemm` `aiter.rotary_embedding` | mla/paged_attn 子包 sglang aiter_backend/aiter_mla_gluon 在用 |

**集成现状**：sglang 树 174 处（`srt/utils/aiter.py` 桥、`aiter_backend`、
`aiter_mla_gluon`、`dsa_backend/dsa_indexer(_kpool)`、**`dsv4/indexer.py`+
`dsv4/compressor.py`**、`qwen_sparse_attn_backend`、multimodal_gen 侧若干）。
adapter 侧：AR GraphSafe（HCU_AITER_AR auto）、fused AR+RMSNorm
（HCU_AR_NORM_IMPL=aiter，boot 横幅实证在装）、aiter fp8-quant tier
（HCU_QUANT_IMPL=aiter）、aiter moe bridge（fused_experts wrap）。

**机会（按我们热点排序）**：
- `fused_allreduce_mhc_post_*` + `fused_qknorm_allreduce_rope`——把逐层
  AR 与后处理/rope 折进同核，直接减每步核数（bs16 图 5694 核/步）与同步暴露；
- `moe_c_moe_gemm_marlin_wfp4a8_channelwise/groupwise`——V4.1-Flash 专家
  fp4（现走 SGLANG_DSV4_FP4_DEQUANT=1 转 fp8 块量化）若有原生 wfp4a8 路可省
  dequant 内存churn（trace：bf16→f32 copy 1104ms/1700 次第一大核族）；
- `moe_c_situ_glu` / `fuse_silu_mul_*_ep`——对应我们 silu_mul_clamp
  （636ms）+per_token_group_quant（245ms）两段分离；
- `qr_all_reduce(_rmsnorm)` 小 AR 低延迟路（dp-attention 小聚合备选）。

## 2. lightop（0.6.0+dtk2604.gaa4a34，128 函数 / 35 类）——HCU 专属全家桶

| 族 | 代表 API | 备注 |
|---|---|---|
| **norm+rope+quant 前融合** | `fused_rms_norm_rope_contiguous` `fuse_rmsnorm_rope_quant_qkv` **`rms_rotary_embedding_fuse_with_kv_store`** `gammarms_rotary_..._with_kv_store` **`split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant`** `mimo_v2_split_rope_vscale_kv_store` `fuse_fill_rms_x2_concat` | rmsnorm+rope+KV写+量化一条龙；`mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang` 带 sglang 兼容名 |
| norm 单体 | `rms_norm(_forward/backward)` `fused_add_rms_norm` `fused_rms_norm_contiguous` `rms_norm_dynamic_per_token_quant(_sglang)` `rms_norm_per_token_fp8_quant` `gemma_*` `layer_norm/fused_add_layer_norm` | |
| gemm/量化 | `gemm` `gemm_w4a16` `gemm_w8a16_fp8` `gemm_w8a8_smooth` `gemm_w8a8_asm(+wgrad/xgrad)` `fp8_to_16` `w4tow16` `blockwise_scaled_cast_transpose` | asm 后端 wgrad/xgrad 训练向 |
| **groupgemm/MoE GEMM** | `moe_gemm_w8a8` `moe_gemm_marlin_w8a8(_fp8/_blockwise_fp8)` `moe_gemm_w4a8(+marlin)` `moe_groupgemm_w4a8(+marlin)` `m_grouped_w4a8/w8a8_gemm_nt_masked` `m_grouped_w8a8_gemm_nt_contig_asm` `m_grouped_w8a8_gemm_nt_nopad_contig_asm` `m_grouped_bf16_gemm_nt_masked` `m_grouped_w16a16_gemm_nt_contig_asm` `groupgemm(_permute/unpermute)` `fuse_groupgemm_permute_quant` | blockwise_fp8 正对我们 [128,128] 块量化 |
| **MoE gate/topk** | `moe_fused_gate` `moe_fused_gate_sqrtsoftplus` **`moe_fused_gate_sqrtsoftplus_v41_hcu`** `moe_fuse_gate_per_gourp_topk` `topk` `fast_topk_transform_fused` `fast_topk_transform_sparse_mask_fused` `fast_topk_transform_ragged_fused` `topk_transform_512/1024` `topk_transform_ragged_v2` `fast_kpool_topk_transform_fused` | **v41_hcu 变体点名 V4.1**；topk_transform 族对应 DSA indexer topk/select |
| **silu 族** | `fuse_silu_and_mul(_ep)` `fuse_silu_mul_quant(_ep)` `fuse_silu_mul_fp8_quant(_ep)` `fuse_silu_mul_per_token_quant` **`fuse_silu_mul_clamp_quant(_ep)`** **`fuse_situ_mul_quant_contiguous/_ep`** | clamp/situ 变体≈我们 silu_mul_clamp 的 fused-quant 版 |
| **KV cache 族** | `reshape_and_cache_cuda` `gather_and_maybe_dequant_cache` `dequantize_k_cache_paged_hcu` **`combine_topk_swa_indices_hcu`** `hcu_alloc_extend_kernel` | swa indices 组合是我们 SWA 模型正需 |
| **dsv41 族** | **`dsv41_gather_index_k_fp4`** **`dsv41_mqa_prefill_metadata`** **`dsv41_mqa_query_lens`** **`fp8_fp4_mqa_logits`** `page_mqa_logits_sparse_mask` `mqa_logits` | V4.1-Flash indexer/MQA 的厂商原生路（现走我们 mqa_logits_triton 自研档） |
| 杂项 | `apply_rotary_pos_emb_index_fused` `rotary_embedding_fuse` `ds_cat` `fill_zeros` `batch_linear_assignment` `grid_sample_forward/backward` `bilinear_upsample(_backward)` `destindex_copy/dequantize_int8_cuda` `sum/mean/topk` + 大量 nn.Module 类（RMSNorm/RotaryEmbedding/FuseBiasSwiGLU/FusedScaleMaskSoftmax…） | vision/训练侧为主 |

**集成现状**：sglang 树 0 处；adapter 接入两处——
`HCU_SILU_MUL_IMPL=lightop`（candidate 档，`lightop.fuse_silu_and_mul(input,
output, mask_m, expect_m)`，未过图重放矩阵前不转正）与
`HCU_FUSED_ADD_RMS_IMPL=lightop`（**validated 档 2026-11-03**：`lightop.
fused_add_rms_norm` / `gemma_fused_add_rmsnorm` 原地核，fp32 累加同 triton
数值契约，parity + cuda-graph 重放双过、py3.10/py3.12 双轮皆有此函数；perf
0.76x 慢于 triton @4096×7168 bf16，默认保持 auto=triton；一次异常即降级；
门禁 `tests/tools/_lightop_fused_add_ab.py`）。

**机会（热点直击，V4.1-Flash bs16 trace 排序）**：
1. `fused_rms_norm_rope_contiguous` / `split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant`——对第一大核族 `fused_k_norm_rope_flashmla`（1165ms/490 次）与 `apply_rotary_emb_triton`（211ms）的深度融合位；
2. `fuse_silu_mul_clamp_quant_ep`——合并 silu_mul_clamp（636ms）+per_token_group_quant（245ms）两段；
3. `moe_fused_gate_sqrtsoftplus_v41_hcu`——router（`_router_triton_kernel` 80ms + reduce 族 561ms 里的 topk 部分）厂商原生 V4.1 位；
4. `dsv41_gather_index_k_fp4`+`fp8_fp4_mqa_logits`——fp4 indexer 不经 dequant 的原生路（对照我们 mqa_logits_triton 档与 deepgemm vendor 档的负值钳 0 缺陷）；
5. `moe_gemm_marlin_w8a8_blockwise_fp8` / `m_grouped_w8a8_gemm_nt_masked`——fp8 块量化 MoE GEMM 的 marlin 备胎。

## 3. boltops（0.1.0+dtk2604.g1f36f1，14 子包）——全未接入的增量库

sglang/adapter 均 0 引用。子包与代表 API（tilelang 子包为懒加载，导入级
枚举为空，需用时逐个验）：

| 子包 | 代表 API | 对我们栈的意义 |
|---|---|---|
| `mhc` | **`mhc_pre_big_fuse`** `mhc_post_fwd` `hc_split_sinkhorn` `mhc_fused_tilelang` `pre_big_fuse_tilelang` | 我们的 tilelang mhc_pre 曾 XID:81 页故障被关（SGLANG_OPT_USE_TILELANG_MHC_PRE=false）——vendor 版可直接对照/替换 |
| `ihc` | `ihc_pre` `ihc_post` `ihc_head` | IHC 变体（HCU 模型线） |
| `step4_dsa` | `triton_build_rope_cache` `triton_indexer_norm_rope` `triton_indexer_logits` `triton_region_topk_ids` **`triton_topk_select_and_pack`** `triton_prefill/decode_sparse_meta` `triton_sparse_attention_prefill/decode` `triton_csa_{region_layout,compress_regions,prefill}` | **DSA 全链 triton 实现**——对照我们 dsa-topk-backend torch 与 mech 账里 topk_v2 构建失败的口子 |
| `mqa_logits` | `tilelang_qsa_mqa_prefill` **`tilelang_qsa_mqa_decode`** `triton_mqa_logits` | indexer MQA logits 第三实现（自研 triton / deepgemm vendor 之外） |
| `qsa` | `qsa_mqa_paged` `qsa_select_paged_tokens` `qsa_sparse_paged_attention` `expand_qsa_block_indices` | QSA 稀疏 attention 全家 |
| `mla` | `flash_mla_sparse_fwd` `decode_attention_fwd(_grouped/_normal)` `compute_num_kv_splits`（triton 后端同名） | MLA decode 多实现 |
| `fa` | **`unified_attention`**（varlen+block_table+softcap+descale）`extend_attention_fwd(_unified)` `dense_prefill_attention_fwd`（fp8 可用判定 `can_use_dense_prefill_fp8`）`triangle_attention_fwd` `build_unified_kv_indices` `reshape_and_cache_int4` | 统一 attention 面，fp8 prefill 路自带判定 |
| `gemm` | **`gemm_a8w8_blockscale`** **`gemm_afp4wfp4`** `batched_gemm_a16wfp4` `batched_gemm_bf16` `w8a8_triton_block_scaled_mm` `w8a8_block_int8_matmul` `awq_gemm_triton` `matmul_persistent`(tilelang) `gemm_a16w16` | fp4×fp4 GEMM 在库；blockscale=int8/fp8 块量化 |
| `fused_moe` | **`fused_experts`** `moe_activation`（situ/clamp 参数齐）`normalize_moe_activation` + triton 子包 23 函数（`invoke_fused_moe_kernel` `moe_sorting_fwd` `per_token_quant_triton` `moe_align_block_size_aiter` …） | vendor 版 fused_experts 整替代选 |
| `generic` | **`causal_conv1d_fn`** **`causal_conv1d_update`** `fused_mamba_state_scatter_with_mask` `attn_res` | conv1d（mech 账 conv1d 6F 的对照）、mamba 状态散射 |
| `sparse_gqa` | `qwen_sparse_kv_extraction_compact_hcu_fa_triton` `sparse_gqa_packed_decode_triton` | Qwen 稀疏 KV 路 |
| `fla` | 子包 common/gdn/kda（懒加载） | GDN/KDA 线性注意力（对照 HCU_LINEAR_ATTN_GDN=aiter 档） |
| `tools`/`utility` | 工具层 | |

**机会**：MHC vendor 链（boltops.mhc + aiter.fused_allreduce_mhc_post_* +
deepgemm.tf32_hc_pernorm_gemm 三件可拼全链）；step4_dsa 对照替换 topk/select
 Torch 段；conv1d；fp4 gemm。风险：0 接入=全量未过我们图重放/精度门，逐算子
 灰度（先 standalone 对拍再进图）。

## 4. flash_mla（1.2.0+dtk2604.g29ff42，9 函数）

| API | 说明 |
|---|---|
| `get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)` | 调度元数据（返回 FlashMLASchedMeta） |
| `flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v, tile_scheduler_metadata, num_splits_k, ...)` | 标准 paged decode |
| `_fp8` / `_quantization` / `_q_nope_pe` / `_quantization_q_nope_pe` / `_fp8_with_cat` 变体 | **fp8 KV 与 q_nope/q_pe 分离布局**各档 |
| `get_mla_decoding_metadata_dense_fp8(...)` | dense fp8 decode 元数据 |
| `flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512, attn_sink, topk_length)` | 稀疏路（我们 `--dsa-decode-backend flashmla_sparse` 现役；sglang `hip_flash_mla.py`/`dsa_backend` 桥） |

注意：trace 里的 `sglang::fused_k_norm_rope_flashmla` 是 sglang 自研融合核，
与 flash_mla 库不同物。机会：fp8/q_nope_pe 变体若过精度门可替换现役 bf16 路。

## 5. flash_attn（2.8.4+dtk2604.g007e38，35 函数）

- 标准 FA2 面：`flash_attn_func` `_kvpacked_func` `_qkvpacked_func`
  `_varlen_func` `_with_kvcache` + vllm_* 别名 + `triton_*` 后备 +
  `flash_attn_with_mask_func`（含 HG 扩展位）。
- **sparse 扩展**（对 Qwen 类稀疏）：`sparse_attn_func` `sparse_gqa_attn_func`
  **`sparse_gqa_attn_fp8_func`** `mqa_paged_score_func`
  `sparse_gqa_paged_attn_func` **`sparse_gqa_paged_attn_fp8_func`**。
- **fa_sparse**：`fa_sparse_attention_prefill/decode`（+`_fp8`）。
- 其它：`hg_flash_attn_varlen_func` `flash_attn_func_blasst`
  `sparse_attn_with_sla` `spas_fa2_attn_meansim(_topk)_cuda`。

集成现状：adapter `HCU_FA_IMPL` vendor 档（历史 gfx936 wheel 词汤账——
paged 内核红，缺口清单待 DTK 修，docs/env-vars.md 有案）；生产现走
auto=宽窗 emu / triton 自研档。机会：sparse_gqa fp8 族待 wheel 质量复核。

## 6. deepgemm（2.1.0+dtk2604.g3c303b，连写包，59 函数）

| 族 | API 矩阵 |
|---|---|
| grouped contiguous | `m_grouped_{fp8,w4a8,w8a8,i8,bf16,wfp4a16}_gemm_nt_contiguous`（fp8 另有 `_tensor_wise`；i8 有 `_nopad`；w4a8 有 `_hipc`） |
| grouped masked | `m_grouped_{fp8,w4a8,w8a8,i8,bf16}_gemm_nt_masked`（fp8/w8a8 另有 `_ll` 低延迟与 `_tensor_wise`；w8a8 有 `_gfx936` 与 `_pack_permute`；w8a8 有 `_compact_masked`（expert_offsets 入参）） |
| dense | `fp8_gemm(_hipc)` `tf32_hc_pernorm_gemm`（sqr_sum 出参=**MHC 链配件**）`w8a16_linear` `matmul_persistent` |
| MoE prefill | `moe_w8a8_marlin_prefill_down_fp8` + weight pack（`marlin_fp8_{contiguous,masked}_weight` 等） |
| **paged MQA logits** | `paged_mqa_logits` `page_mqa_logits_sparse_mask` `_grouped` `_grouped_int8` —— **R-02 已定罪 q/k 负值 fp8 钳 0（rel~3 垃圾），待 DTK 修（deepgemm 2.1.0+dtk2604）；哨兵 tests/kernels/test_mqa_vendor_gfx938.py** |
| w4/wfp4 masked | `grouped_gemm_w4a16_nt_masked_entry` `grouped_gemm_wfp4a16_nt_masked_entry` `moe_w4a16_groupwise_triton` |

集成：`HCU_DEEPGEMM_BRIDGE=1`（EP 启动必带）+ `SGLANG_DEEPEP_BF16_DISPATCH=1`；
sglang 侧 deep_gemm 90 处。机会：`m_grouped_wfp4a16_*`（fp4 专家免 dequant）、
`tf32_hc_pernorm_gemm`+boltops.mhc 拼 MHC 全链。

## 7. flash_kda（0.0.1）与 tokenspeed_mla（0.1.8）

- flash_kda：`fwd(q,k,v,g,beta,scale,out,A_log,dt_bias,lower_bound,
  initial_state/final_state,cu_seqlens,enable_cp,...)` + cp 布局族
  （`make_cp_layout` `select_fixed_cp` `get_workspace_size`）——KDA gated-delta
  注意力完整前向（sglang 1 处模型线引用）。
- tokenspeed_mla：`tokenspeed_mla_decode`（paged，fp16/fp8 子模块）
  `tokenspeed_mla_prefill` **`mla_kv_pack_quantize_fp8`**
  `warmup_compile_prefill`（PDL 可开）——MLA 第三方 decode 实现，
  fp8 KV pack 量化件可单独摘用。

## 8. 机会优先级（映射 V4.1-Flash bs16 decode 热点账）

trace 预算（10 步 TP-0，总量 9.23s）：elem/mem 3209ms（35%，含 bf16→f32
copy 1104ms/1700 次）> other 1965 > sample/logits 1308（markov 等待为主）>
attention 1166（`fused_k_norm_rope_flashmla` 1165/490 次）> reduce 561 >
nccl 387 > quant 258（`per_token_group_quant_8bit_v2` 245/1770 次）> rope 211
> norm 130 > gemm 19 > moe 16。拍（图内大同步等待，~64% 步时）仍是第一瓶颈
（见 env-vars.md 拍定性段），算子收益在拍之外独立成立、拍死后全额兑现：

1. **norm+rope(+quant/KV写) 前融合**（lightop `fused_rms_norm_rope_contiguous`
   / `split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant`；aiter
   `fused_qknorm_allreduce_rope`）——打 attention 族第一大核+rope 分离段；
2. **silu_mul_clamp×quant 融合**（lightop `fuse_silu_mul_clamp_quant_ep`、
   aiter `moe_c_situ_glu`/`fuse_silu_mul_fp8_quant_ep`）——636+245ms 两段并一；
3. **copy/dequant churn**（aiter/浅层 `blockwise_scaled_cast_transpose` 类、
   wfp4 GEMM 免 dequant：aiter marlin wfp4a8 / deepgemm wfp4a16 /
   boltops gemm_afp4wfp4 / lightop fp8_fp4_mqa_logits）——打 1104ms copy 族；
4. **MoE gate/topk**（lightop `moe_fused_gate_sqrtsoftplus_v41_hcu` +
   `fast_topk_transform_*`；aiter `moe_c_topk_softmax`）；
5. **MHC vendor 链**（boltops.mhc + aiter fused_allreduce_mhc_post_* +
   deepgemm tf32_hc_pernorm_gemm）——tilelang 自研页故障的替代路线；
6. **DSA topk/select**（boltops.step4_dsa `triton_topk_select_and_pack` 等）
   ——对照 dsa-topk-backend=torch 与 topk_v2 构建失败账；
7. **conv1d**（boltops.generic `causal_conv1d_fn/_update`）；
8. MLA fp8（flash_mla fp8 变体 / tokenspeed_mla）——精度门后。

接入纪律：任何 vendor 核先 standalone 对拍（金标准 rel 阈值）→ 图捕获重放
压测（cookbook）→ 四门验收 → 再谈默认档（hcu_config catalog，不裸 import；
governance.md）。新库（boltops/lightop 全量）须逐算子建哨兵测试。

## 9. 复现内省

```bash
# 容器内（不执行任何核，纯导入枚举）
python3 tmp_fetch/introspect_libs.py   # 8 库顶层+关键子包 -> JSON
python3 tmp_fetch/introspect2.py       # boltops 14 子包全枚举 -> JSON
find /usr/local/lib/python3.10/dist-packages/aiter -name '*.so' | wc -l  # 51
```

原始 JSON 与脚本随仓 `tmp_fetch/`（scratch，不入库；本文档为沉淀面）。
