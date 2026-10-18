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
   （尺度逐位一致+q RTN 99.8%）；scaled_fp8_quant 三 natives=上游
   SGLANG_USE_AITER 同款调用。catalog planned→candidate；lightop 维持
   planned。
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
