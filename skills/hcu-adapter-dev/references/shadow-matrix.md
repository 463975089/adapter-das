# 影子库矩阵（shadow-library matrix）

## 边界定义

以**社区代码的 import 面**为边界：sglang / vllm 源码里 `import` 到的算子
库名，本仓库都要让 `import` 成功。不锚定任何库的内部实现（内部实现随上游
版本漂移，import 面稳定得多）。

DeepInfer 契约实现按算子族和档位放入 `hcu_adapter/kernels/`，文件位置及入口职责见[算子文件导航](../../../docs/frameworks/deepinfer-kernels.md)。
拆分实现时同步 bridge/catalog 与 plugin 的导入和能力检查；删除迁移后的旧算子转发路径。

当前覆盖的 import 面（发现新的就加进矩阵）：

| 社区 import 的包 | 我们的形态 | 位置 | 状态 |
|---|---|---|---|
| `sgl_kernel` | 完整同名包（triton/torch） | `sgl_kernel/` | 生产 |
| `vllm._custom_ops` | SGLang 启动链使用的 RMS 属性级兼容绑定；不是 vLLM 宿主的默认重绑定 | `hcu_adapter/bridges/vllm.py::install_rms_norm_bridge` | SGLang 兼容路径 |
| `aiter` | 真包优先 + 缺口填空 + 高危守卫 | `hcu_adapter/aiter/` | 生产 |
| `flashinfer` | 填空桥 + 响亮 stub | `hcu_adapter/flashinfer/` | 生产 |
| `flash_attn` / `flash_attn_interface` | 响亮 stub；装 HCU flash_attn wheel 后真包自动胜出（零改动切 vendor 档） | `hcu_adapter/flash_attn/` | 生产 |
| `lightop` | vendor 直调档（不出现在 import 面，按 ops 路由调用；见下盘点） | 系统包直调 | silu/MQA 已入 catalog |

**RMS 宿主 ABI 分开核对**：上述 helper 的生产调用点在
`hcu_adapter/bridges/sglang.py`；vLLM runtime plugin 不调用它。helper 的
`fused_add_rms_norm(out, input, residual_out, residual, weight, epsilon)` 是
六参数、独立输出目标的兼容签名；vLLM 0.32 官方
`fused_add_rms_norm(input, residual, weight, epsilon)` 是四参数原位接口，
修改 input/residual 并返回 `None`。不能因同名把前者装进 vLLM 启动链。
该属性绑定也不是 `vllm._custom_ops` / `_C` 的完整影子实现；缺失的原生符号
须逐项验证。vendor 的响亮 stub 可让 import / `hasattr` / `callable` 成功，
这些检查不能证明所需 provider 的 ABI 或计算可用。

FlashInfer shadow 的 norm 转发在调用时导入既有 `sgl_kernel.layernorm`
实现；不能在模块顶层先导入它，否则 shadow-first 顺序会经
`sgl_kernel` 的 stub 注册重入尚未定义 `register()` 的模块。
注册仍在启动期完成，不延迟换绑。新解释器回归覆盖 shadow-first、
kernel-first、公开入口及原 `out=`/原位 ABI；这是 import/转发证据，
不能替代 GPU 数值、Graph 或模型门禁。

**flash_attn wheel 兼容性现状**（2026-10-10 实测）：共享盘
`/public/share/dtk_fa_whl/*/flash_attn-2.6.1+das.opt1.dtk2604-cp310`
只匹配 torch2.5 栈；dtk26041/torch2.11 上 `c10::hip` 符号不匹配，需
vendor 出新轮。轮子就位后 `tests/vendor/test_flash_attn.py` 的探针会自动
切到真包跑全场景矩阵（sglang-das 侵入仓同款回退链
`kernels → sgl_kernel.flash_attn → flash_attn 包`）。

## 三档实现策略

每个算子出口按优先级路由，出口签名逐字对齐社区版（含 `out=`、
destination-passing、占位参数）：

1. **triton（生产主力）**：自研移植，fp32 累加、graph-safe、无主机同步。
2. **vendor（转发档）**：DTK wheel 里真包已有且经过我们的压测+守卫的
   （如 aiter 部分 kernel、lightop）。接入条件 = 过场景压测 + 注册快关。
   arch 探测失败自动降级到 triton/torch 并打日志。
3. **torch（参考档）**：语义金标 + 功能兜底（慢但永远正确），也是
   契约测试的参考实现来源。

**档位状态机与演进（catalog 管制）**：每算子
出口的可用实现在 BACKEND_CATALOG 挂状态——
`planned → candidate（离线三段式绿+对拍+bench）→ validated（serving A/B
过）→ production（默认档，连续稳定后）`，定罪不可用标 `broken`（选配即
拒，bug 签名与复现探针在册；哨兵意外转绿=轮子修好，重跑门禁即晋级）。
演进**单向**：**emu（torch 调试图稳/金标准）→ triton → vendor**；
emu/triton 换下不删、转 ref 角色只留离线测试对照。校准方向不变：emu 校准
triton，emu/triton 暴露 vendor bug。晋级证据门与登记要求见
`docs/governance.md`。

路由已落地（env + yaml 双层）：`HCU_SILU_MUL_IMPL` / `HCU_RMS_NORM_IMPL` /
`HCU_FUSED_MOE_IMPL` 等已在册（hcu_env REGISTRY），正式机制是
`hcu_adapter.yaml` 的 ops 键（见 docs/config.md 与 BACKEND_CATALOG），
`auto` = 探测+压测结论决定；新算子接入时同步登记 catalog。

## lightop 接入盘点（DTK 0.6.0，nmz26 dtk26041 实测 export 面与 aiter 交集）

gfx938 有 fp8（gfx936 只有 int8）。**与 aiter 同用法的交集算子**（同一语义
两边都有实现 → 同一份契约测试跑双 backend，正是 73-297 要的"默认打开、
排精度时可切"）：

| 语义 | lightop 出口 | aiter 出口 | 接入优先级 |
|---|---|---|---|
| silu_and_mul(+fp8 quant 融合) | `activation.existing.fuse_silu_and_mul` / `fuse_silu_mul_fp8_quant` 系列（含 _ep） | `silu_and_mul` / fused_clamp_act_mul 融合 | P1（契约最简单，跑通双 backend 模式） |
| MQA logits（DSA） | `attention.mqa.paged_mqa_logits` / `mqa_logits` / `_fp4` 变体 | `aiter.ops.triton(.attention).pa_mqa_logits` | P1（我们有 mqa_logits_emu torch 金标可直接对拍） |
| MoE group gemm | `moe.groupgemm_permute / fuse_groupgemm_permute_quant / gemm_ops.quantized.gemm_w8a8_*`（marlin w4a8/w4a16 配置齐） | aiter 项目前 moe 出口少 | P2 |
| RMSnorm+RoPE 融合 | `fuse_rmsnorm_rope.rms_rotary_embedding_fuse(_with_kv_store)` | 无对应（aiter 分开） | P2（性能档） |
| KV cache 反量化 | `kvcache.dequantize_k_cache_paged` / `gather_and_maybe_dequant_cache` | `cp_gather_indexer_k_quant_cache`（我们已桥） | P3 |

接入顺序：P1 两个先做（silu 融合→mqa），每个出口注册 `HCU_<OP>_IMPL`
env 档位（auto=vendor 过压测则用/triton/torch），四个维度测试各跑一遍。
 أصل轻盘点记录：`lightop._lmslim_native.quantize` 顶层无导出（空）、
marlin repack 只在 lightop 有（aiter 无交）。

## 新增一个算子垫片的流程

1. **定契约**：找到社区调用点，抄下完整签名/布局/返回语义（读上游源码，
   不要猜；rope 教训：同一个函数名在 cache 布局上还有隐含约定）。
2. **写参考**：`torch` 档参考实现（fp32），这就是测试金标。
3. **写生产档**：triton 或 vendor 转发；遵守黄金法则（fp32 累加、无
   主机同步、import 不失败）。
4. **写契约测试**：按 `test-suite.md` 四维度；vendor 高危面同时写场景
   压测 + 注册快关（`kill-switches.md`）。
5. **更新矩阵**：本文件表格 + README 覆盖范围表。
