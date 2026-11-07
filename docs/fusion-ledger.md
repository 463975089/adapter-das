# 上游融合面账本（R-09 盘点，2026-10-18）

is_cuda 语义路由的红利 = 上游 CUDA 分支的算子融合被上游自己选中。本账本
盘点上游 v0.5.21 的融合入口全集，对照我方实现形态，四态：**vendor 融合 /
triton 融合 / 已拆解（登记理由）/ 缺失（桩）**。原则见 governance"融合沿用、
拆解为例外、禁止静默拆解、正确性优先"。

扫描方法：容器 /sglang-v0521 srt 树 flashinfer.*/sgl_kernel.* 调用点聚类
（flashinfer.fused_moe 39 处 / comm 28 / decode 13 / sgl_kernel.flash_mla 11
等为调用频次，下表按入口族归并）。

## 一、flashinfer 系（HCU 上 flashinfer wheel 整体不可用，is_flashinfer_available=False）

| 入口族 | 上游路径 | 我方形态 | 态 | 登记/理由 |
|---|---|---|---|---|
| fused decode/prefill attention | flashinfer.decode/prefill | fa3 emu / 分页 triton | 已拆解 | BACKEND_CATALOG flash_attention（emu production / triton candidate 4.7×） |
| MLA absorb | flashinfer.mla | flashmla_sparse + fa3 emu | 已拆解 | 战报 10-14 起 in 役；flash_mla 单测在 L1 |
| sampling（renorm+sample） | flashinfer.sampling | torch masked filter→renorm→multinomial | 已拆解 | catalog sampling（torch production，triton planned 对拍后可选） |
| AR+norm 融合（trtllm 家族） | flashinfer.comm | aiter AR / NCCL + triton norm 两步 | **缺失** | 头部候补①，见下 |
| fused norm | flashinfer.norm | triton rms_norm / fused_add_rmsnorm | 已拆解 | catalog rms_norm（triton production） |
| gemm（含 fp8） | flashinfer.gemm | torch mm + DTK deepgemm 桥 | 已拆解 | HCU_DEEPGEMM_BRIDGE（R-04）；fp8 contiguous 解锁=R-04 fp8 段 |
| DSA topk/page-table/ragged transform | flashinfer.top_k* | torch topk（dsa-topk-backend torch） | 已拆解 | SGLANG_DSA_FUSE_TOPK=0（model_env 自动化，R-08 B） |
| kda_decode | flashinfer.kda_decode | KDA emu（chunk 链 torch） | 已拆解 | HCU_KDA_PREFILL_IMPL=emu；fork 转正中（R-01） |
| mhc | flashinfer.mhc | 纯 torch 链（HCU_MHC_UNFUSE=1） | 已拆解 | 10-16 词汤终局战报（融合门控只查 env 不查 is_hip 的上游 bug） |
| mamba/conv | flashinfer.mamba | conv emu / fork | 已拆解 | HCU_CONV1D_IMPL；fork 长上下文 VMFault 复验挂起（R-01） |
| rope | flashinfer.rope | triton NeoX | 已拆解 | catalog rotary_embedding（triton production） |
| fp4_quantization | flashinfer.fp4_quantization | 无 | 缺失→显式不做 | gfx938 无 fp4 矩阵单元，无场景 |
| moe_ep（EP dispatch 面） | flashinfer.moe_ep | deepep + DTK 桥 | 已拆解 | HCU_DEEPEP_BRIDGE / bf16 dispatch（EP-LL 30197 留证） |
| fused_moe | flashinfer.fused_moe | triton fused_moe + aiter moe | 已拆解 | catalog fused_moe（triton production / aiter validated 调优表） |

## 二、sgl_kernel 系（上游 C++ 轮；HCU 由本仓影子库 sgl_kernel 承接）

| 入口族 | 上游路径 | 我方形态 | 态 | 登记/理由 |
|---|---|---|---|---|
| flash_mla | sgl_kernel.flash_mla | flashmla_sparse vendor + emu | vendor 融合 | DTK flashmla 轮 in 役；单测 L1 |
| flash_attn | sgl_kernel.flash_attn | triton / fa3 emu | 已拆解 | flash_attention op |
| per-token quant | sgl_kernel.quantization | torch | 已拆解 | catalog per_token_quant（aiter/lightop planned=vendor 候补③） |
| causal_conv1d fn/update | sgl_kernel.mamba | emu / fork | 已拆解 | HCU_CONV1D_IMPL |
| rotary_embedding | sgl_kernel.rotary_embedding | triton | triton 融合 | rotary_embedding op |
| fused_add_rms_norm | sgl_kernel（norm 族） | triton | triton 融合 | rms_norm op（fused_add/gemma 变体恒 triton） |
| silu_and_mul(+quant 融合族) | sgl_kernel（activation 族） | triton / lightop | vendor 融合 | silu_and_mul op；lightop.fuse_silu_and_mul validated。**+fp8 quant 单核融合=候补②**（现两步：silu 后单独 per-token quant） |
| moe gate（topk_softmax/sigmoid） | sgl_kernel.topk_* | triton | triton 融合 | moe_gate op |
| moe_align | sgl_kernel（moe 布局） | GPU triton 直通 + CPU torch 移植 | triton 融合 | cf43e21（CPU 哨兵桶修复） |
| weight_packed_linear | sgl_kernel.weight_packed_linear | 未接 | 缺失 | **候补④**：随 deepgemm weight pack 面（R-04 摸底 wheel API：group_gemm_weight_pack）；HCU 走 DTK 桥自身 pack，上游此入口暂无对应场景 |
| sgl_kernel_npu.* | NPU 分支 | — | 排除 | is_npu 路由，HCU 无关 |

## 三、头部缺失项排序（验收标准 2）

1. **AR+residual+rmsnorm 融合**（trtllm_allreduce_fusion 族）：TP 每层
   边界两步→一步的时延收益，decode 小 batch 敏感。去向：评估 aiter 是否
   有 fused AR（GraphSafe 之外的 API 面）→ 无则 lightop → 都无则列自研
   候补（依赖 R-01 ar_ext.cu 修复后的 custom AR 底座）。
   **定裁更新（2026-10-19 探针役，战报 08bce33）**：aiter
   fused_allreduce_rmsnorm(_quant/_per_group/_pad) 家族 op schema 全确认
   （reg_ptr/reg_bytes IPC 注册缓冲 + _fa 句柄）——需 4 rank 联调探针排期；
   接入评估须对照 R-06 custom AR 定罪史（store-ordering 教训）。
   **落位（2026-10-19，R-09①）**：8-rank 探针（validation-host-26 BW1102，
   tmp_fetch/probe_r09_arnorm.py，hidden=4096 t=1..4096）三段全绿——
   ①数值：对 fp64 精确 AR 真值 norm **1ulp / residual 0ulp**（rccl 参照
   3-8ulp；IPC 精确和 vs 环形累加，融合核反而更准）；②perf：vs
   customAR+norm **1.38-2.11×**（decode 区 1.5-1.7×，t=256 峰值 2.11×）；
   ③图回放：新鲜数据两轮 1ulp/0ulp 精确（R-06 旧类顾虑清除，GraphSafe
   类融合面同底座）。桥 `ar_norm_aiter.py`（`HCU_AR_NORM_IMPL` auto
   默认）：上游 layernorm 休眠链（`GroupCoordinator.
   fused_allreduce_rmsnorm`→aiter `custom_fused_ar_rms`，aiter 类原生供
   给）经 **flashinfer 入口一符号外科激活**（不开 SGLANG_USE_AITER 全局
   门）；仅全 TP 组目标融合——上游 aiter 分支无视 use_attn_tp_group 错组
   （dp-attention 下数值错），本桥按 get_parallel() 组尺寸门控逐调用退
   stock；_quant_per_group 变体上游自门 gfx95 → gfx938 恒不走（正确）。
   注意事项：aiter 类挂 gloo 组（cpu_group），IPC 握手不走 NCCL 组；
   多进程并发跑 sitecustomize 时 DTK 头文件 patch 竞争是既有噪音
   （needle-missing 告警，try 包裹无害）。**v2 教训（serving 冒烟抓出）**
   ：上游候选门 `apply_aiter_all_reduce_fusion`（要 SGLANG_USE_AITER
   全局门）或 `apply_flashinfer_allreduce_fusion`（要 sm90/100）在 HIP
   恒 False——只绑 flashinfer 入口（dispatch）架构性不可达，GLM5 的
   stages 也不传 backend fusions provider；v2 改为**双符号**：arm=
   apply_flashinfer_allreduce_fusion 三副本重绑（武装候选 try+exit 的
   生产端推迟），dispatch=原 flashinfer 入口包装（全 TP 组门+逐调用退
   stock）。**tp8 serving 实证（GLM-5.3 纯 TP 无 dp，mtp-minimal 图开）**
   ：图捕获期起服、attn+FFN 双边界、216 次融合 0 失败、输出正常；dp-
   attention（用户点：dp==tp 时 attn 边界本无 AR）在 arm 层即拒。**EP
   排除（R-04 EP-LL 复验役实证 2026-10-18）**：TP2+EP2 下 arm 武装使生产
   端推迟归约、fused 消费者因组非 attn-tp decline、回退走 reduce-scatter
   完成路径——aiter `ca_comm.reduce_scatter` 拒重叠输入输出张量
   （ValueError）崩调度器；arm 收紧为**仅纯 TP 拓扑**（moe_ep>1 或
   moe_tp≠tp 即拒），EP 形态恢复上游 eager 完成路径（=10-11 绿配置）。
   该 reduce_scatter 重叠契约缺口本身是上游 HIP 面 finding（潜在 R-13
   收编项）。日志
   纪律：sitecustomize 期 INFO 被 sglang 前的默认 WARNING 吞——安装/介
   入行用 warning+once 守卫（v2 首跑 216 条刷屏教训）。catalog
   allreduce/aiter note 补融合面；门禁 tests/kernels/
   test_ar_norm_aiter_gfx938.py（<2 可见卡 skip——L1 kernels 段单卡，
   8rank 全量窗口手跑）。
2. **silu_and_mul + fp8 per-token quant 融合**：MoE/FFN 边界读写省一遍。
   去向：核 lightop.fuse_silu_and_mul 的 mask_m 形态是否可挂 scale 输出
   → 可则升档写门禁测试。
   **定裁更新（2026-10-19）**：aiter `fuse_silu_mul_per_token_quant`
   数值健康（cos 0.9999，激活 bf16 舍入差级）且 **7.5×**（56 vs 419us，
   4096×2048；行级尺度 [M,1]，MoE 布局带 topk/expert_ids 参数）——候选
   定案。
   **落位（2026-10-19，R-09②）**：上游 group 入口原生带
   `fuse_silu_and_mul` 臂——整行组（group_size==K_out）路由 aiter
   fused（quant_aiter 桥），窄组保持上游两步；门禁
   tests/kernels/test_quant_aiter_gfx938.py 常驻 L1。
3. **per-token quant vendor 档**：aiter per_token_quant_hip / lightop 族
   对拍（catalog 已 planned，直接排期）。
   **落位（2026-10-19，R-09③）**：aiter 桥全档接入（quant_aiter，
   `HCU_QUANT_IMPL` auto 默认）——rowwise=**行为还原**（上游
   `sglang_per_token_quant_fp8` 在 HIP NameError：`sgl_per_token_quant_
   fp8` 不存在，group 入口整行快捷分支同踩）；group plain 臂 15.1×
   （尺度逐位一致+q RTN 99.8%）；scaled_fp8_quant 两 dynamic natives=
   上游 SGLANG_USE_AITER 同款调用；**static 恒原生**（首跑实证：aiter
   轮 kernel 无 clamp，|x/s|>448 溢出 NaN 占 21%≈|x|>4.48 理论占比，
   原生 clamp 逐位一致；上游 SGLANG_USE_AITER 同 NaN 但本桥承诺
   stock parity 或回退，static 不在 serving 路径即裁）。catalog
   planned→candidate；lightop 维持 planned。
   **定裁更新（2026-10-19）**：aiter `per_token_group_quant_fp8`（注意
   out 式契约 out!/input/scales!/group_size，非返回式）数值健康（尺度
   逐位一致 + q RTN 命中 99.8% + 去量化 cos 0.999994）且 **15.1×**
   （68 vs 1021us，4096×8192）；per_token_quant_hip 行级档同健康——候选
   定案，桥接实施+catalog planned→candidate 待排期。
4. **weight_packed_linear**：与 R-04 fp8 contiguous 同批评估（同一
   weight-pack 机制面）。
   **定裁更新（2026-10-19）**：R-04 同族探针已判该轮 wheel 三重不可用
   （gfx936 无 fp8 contiguous 臂 / wasp_pair32 契约+符号双坏 / marlin
   tensor-wise 坏）——本项随下轮 DTK wheel 复评。

注意力与 GEMM-量化两大类无缺失（替代路径全部在册在役）。

## 四、治理

- 本账本为 R-09 验收物；新增融合拆解必须同步 BACKEND_CATALOG 登记（防
  静默拆解），vendor 融合升档必须带门禁测试。
- 与 BACKEND_CATALOG 对账结论（2026-10-18）：全部"已拆解"项均可在
  catalog/env-vars.md 找到对应 op 或 env 条目，无未登记拆解。

## 五、v0.5.22 对齐复核（2026-10-22，性能分支对齐役）

方法：/sglang-v0522 树 flashinfer.*/sgl_kernel.* 调用点重聚类（对照 §一/§二
的 0.5.21 基线）+ V4.1-Flash dp8ep8+dspark bs16 带栈 20 步 trace 的
torch-native 热核归因（`prof_dsv41f_dec20/1792592316*.trace.json.gz`，图内
核无运行时关联，按"次数/步 ÷ 61 层"指纹 + 模型源码逐行对位）。热点总账见
docs/vendor-kernels.md §8。

### 5.1 MHC（超连接）pre 边界 = 最大 torch-native 面，实锤定罪

**trace 指纹**（20 步 TP-0）：`bfloat16_to_float_copy` 1935ms/3720 次
（186/步 ≈ 3×61 层）、`at::native::reduce_kernel` 1725ms/4920 次
（246/步 ≈ 4×61 层）、`CUDAFunctorOnSelf_add` 174ms/1880 次、`FillFunctor`
92ms、`direct_copy` 族 1716ms——四族合计 ~3.8s/20 步 ≈ ~190ms/步，全为
逐层整数倍节律 = 每层模型代码里的 torch 链。

**源码对位**（`srt/models/deepseek_v4_mhc.py::_mix_stats_impl`，两个子层
×61 层）：

```
use_fused_stats = _is_gfx95_supported or (torch.version.cuda is not None
                  and (is_blackwell or hopper_fused_stats))     # gfx938 全落空
if use_fused_stats:  hc_mix_stats_sinkhorn(...)                  # 融合核
elif torch.version.cuda is not None:  hc_mix_stats(...)          # triton 保底
else:  x_flat.float(); rsqrt(square().mean(-1)); F.linear(...)   # ← 我们在这
       + hc_split_sinkhorn(...)                                   # triton（健康）
```

HIP 上 `torch.version.cuda is None` 恒成立 → 我们每层走 `.float()`（fp32 全
宽拷贝=bf16→f32 族）+ `square().mean`（reduce 族）+ fp32 skinny linear +
rsqrt/mul（self_add/Fill 族）。post 侧已走 tilelang `mhc_post_split_h`
（trace 860 次 29.5ms，健康）——**烧钱的只是 pre 边界**。

**关键发现：上游自备 HIP 融合核，只被门拦住**。
`kernels/ops/layernorm/mhc.py` L2513 `if _is_hip: hc_mix_reduce_sinkhorn_vec(...)`
（`mhc_boundary_hip.py`：CDNA 无 tf32 → ieee fp32 mix-dot、块尺寸/滑窗常量
显式调优、批不变树序设计——就是给 HIP 写的，MI350 gfx95 同款路径）。我们
到不了它 purely 因为 `use_fused_stats` 的平台门。

**对齐行动（按优先级）**：
- **A1（首选，改一行门）**：needle/semantic 把 `use_fused_stats` 对 gfx938
  放行（保持 batch-invariant 模式排除与 bf16 前提）→ 自动进
  `hc_mix_stats_sinkhorn` 的 HIP 分支。验收：hc_split_sinkhorn 数值对拍
  （pre/post/comb 三元组逐位或 fp32 容差）+ 四门 + 图捕获重放。收益上界
  ~190ms/步（拍外真算部分），另每层 ~10 核 → 2-3 核。
- A2（备选）：boltops.mhc `mhc_pre_big_fuse`/`hc_split_sinkhorn`（vendor，
  见 vendor-kernels.md §3）；我方自研 tilelang mhc_pre 的 XID:81 页故障史
  （SGLANG_OPT_USE_TILELANG_MHC_PRE=false）不构成对上游 HIP 分支的否证
  ——不同实现。
- A3（远期）：`SGLANG_OPT_DEEPGEMM_HC_PRENORM`（上游默认 True）被
  sm90/sm100 门拦；DTK deepgemm `tf32_hc_pernorm_gemm` 在库，A1 之后评估。
- 边界融合增量：aiter `fused_allreduce_mhc_post_*`（AR+MHC-post 同核）在
  vendor-kernels §1——A1 落位后可与 R-09① 的 AR+norm 桥同框架收编。

### 5.2 现状校正（§一/§二 表格 stale 行，2026-10-19 定裁已落但表未刷）

| 表格行 | 账面态 | 实际态（2026-10-22） |
|---|---|---|
| AR+norm 融合 | 缺失（头部候补①） | **已落位**（R-09① aiter 桥 ar_norm_aiter.py，HCU_AR_NORM_IMPL auto 默认，纯 TP 拓扑门控） |
| silu_and_mul+quant 融合 | 候补② | **已落位**（R-09② 整行组路由 aiter fused；V4.1 的 clamp 形态未盖——见 5.4-②） |
| per-token quant | torch 已拆解 | **已落位**（R-09③ aiter 桥 quant_aiter auto 默认；boot 横幅实证） |
| MQA logits（DSA indexer） | （0.5.21 未列） | triton 自研档 production（R-13 #4，13.2× emu）；vendor deepgemm 负值钳 0 缺陷在案 |
| fa3 面 | emu/triton 已拆解 | triton varlen 面 q 行子分块修复后 11k 绿（c59135b），候选转强 |

### 5.3 v0.5.22 新融合家族差量（0521→0522）

新出现（均为 flashinfer-gated，HCU 不可用，维持已拆解形态）：`trtllm_mla_
backend`/`trtllm_mha_backend`/`deepseek_v4_trtllm_backend`、`flashinfer_
trtllm_moe`（moe_runner + token_dispatcher.flashinfer）、`flashinfer_comm_
fusion`（trtllm allreduce fusion 面，我方对应物=R-09① aiter 桥）、
`flashinfer.autotuner`、`top_k_page_table_transform`(3)/`top_k_ragged_
transform`、`flashinfer.cute_dsl`、`flashinfer.fp4_quantization`（gfx938 无
fp4 单元维持显式不做）。

sgl_kernel 新面：`kvcacheio`(13 处，影子库 sgl_kernel/kvcacheio.py 已承接)、
`gdn_attention`（HCU 走 HCU_LINEAR_ATTN_GDN=triton fla/aiter 档）、
`qkv_proj_with_rope_fused_weight`（暂无 HCU 在役场景）、`fused_sigmoid_mul_
cpu` 等 CPU 族（L0 路径，影子库按需补）。在树 kernel 新文件：
`mhc_boundary_hip.py`（见 5.1）、`hy4_ihc.py`、`rms_normalize_hip.py`、
`hc_combine_norm.py`、`mhc_mega.py`、`mhc_post_combine_norm_prefill.py`、
`rmsnorm_fanout.py`、`fused_eh_norm.py`、`grouped_gemma_rmsnorm.py`、
`mxfp8_epilogue.py`——**HIP-aware norm 家族比 0521 显著扩容，逐个按模型
需求盘点**（IHC=GDN 系、fanout/eh_norm 待查）。

### 5.4 对齐优先级重排（结合拍瓶颈与 V4.1 热点账）

1. **MHC pre 门（5.1-A1）**——一行门 + 上游现成 HIP 核，性价比最高；
2. **V4.1 silu_clamp×quant 融合**——R-09② 落的是标准 silu 整行组；
   V4.1-Flash 走 clamp 形态（trace `silu_mul_clamp` 636ms +
   `per_token_group_quant` 245ms 分离），对齐物=lightop
   `fuse_silu_mul_clamp_quant(_ep)` / aiter `moe_c_situ_glu`；
3. norm+rope(+quant/KV 写)前融合——lightop 族（vendor-kernels §2），
   对 `fused_k_norm_rope_flashmla` 1165ms 大核位；
4. DSA topk/select——上游 flashinfer.top_k* 维持已拆解（torch），对齐物=
   boltops.step4_dsa `triton_topk_select_and_pack` 族 / lightop
   `fast_topk_transform_*`；与 mech 账 topk_v2 构建失败合流处理；
5. copy churn 常态治理——index_select/direct_copy 875 次/步多为布局胶水，
   随 1-3 落位自然消退，剩余项在下次 trace 复盘归因。
