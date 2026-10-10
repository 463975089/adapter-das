# 影子库矩阵（shadow-library matrix）

## 边界定义

以**社区代码的 import 面**为边界：sglang / vllm 源码里 `import` 到的算子
库名，本仓库都要让 `import` 成功。不锚定任何库的内部实现（内部实现随上游
版本漂移，import 面稳定得多）。

当前覆盖的 import 面（发现新的就加进矩阵）：

| 社区 import 的包 | 我们的形态 | 位置 | 状态 |
|---|---|---|---|
| `sgl_kernel` | 完整同名包（triton/torch） | `sgl_kernel/` | 生产 |
| `vllm._custom_ops` | 属性级桥接（rms_norm 等 HIP 形态） | `sitecustomize.py` | 生产 |
| `aiter` | 真包优先 + 缺口填空 + 高危守卫 | `sgl_kernel/_adapters/aiter/` | 生产 |
| `flashinfer` | 填空桥 + 响亮 stub | `sgl_kernel/_adapters/flashinfer/` | 生产 |
| `flash_attn` / `flash_attn_interface` | 响亮 stub；装 HCU flash_attn wheel 后真包自动胜出（零改动切 vendor 档） | `sgl_kernel/_adapters/flash_attn/` | 生产 |

**flash_attn wheel 兼容性现状**（2026-10-10 实测）：共享盘
`/public/share/dtk_fa_whl/*/flash_attn-2.6.1+das.opt1.dtk2604-cp310`
只匹配 torch2.5 栈；dtk26041/torch2.11 上 `c10::hip` 符号不匹配，需
vendor 出新轮。轮子就位后 `tests/test_flash_attn_gpu.py` 的探针会自动
切到真包跑全场景矩阵（sglang-das 侵入仓同款回退链
`kernels → sgl_kernel.flash_attn → flash_attn 包`）。
| `lightop` | **待接**（DTK 0.6.0 内置，见下） | — | 规划 |

## 三档实现策略

每个算子出口按优先级路由，出口签名逐字对齐社区版（含 `out=`、
destination-passing、占位参数）：

1. **triton（生产默认）**：自研移植，fp32 累加、graph-safe、无主机同步。
2. **vendor（转发档）**：DTK wheel 里真包已有且经过我们的压测+守卫的
   （如 aiter 部分 kernel、lightop）。接入条件 = 过场景压测 + 注册快关。
   arch 探测失败自动降级到 triton/torch 并打日志。
3. **torch（参考档）**：语义金标 + 功能兜底（慢但永远正确），也是
   契约测试的参考实现来源。

路由统一走 env（规划中，接入 lightop 时落地）：
`HCU_<OP>_IMPL=auto|triton|vendor|torch`，`auto` = 探测+压测结论决定。

## lightop 接入盘点（DTK 0.6.0，nmz18 实测）

gfx938 有 fp8（gfx936 只有 int8），这些出口可作为 vendor 档候选：

- attention：`paged_mqa_logits`（gfx936 int8 / gfx938 fp8 分支，head_dim=128）
  —— 替换 `_adapters/mqa_logits_emu.py` 的 torch 模拟，MQA 性能主杠杆；
- quant：`per_token_quant_int8`（W4A8 激活量化直接可替换）、fp8/blockwise；
- moe：gemm / marlin / routing / permutation；
- gemm_ops：quantized / grouped；
- 融合：`fuse_silu_mul_quant`、`fuse_rmsnorm_rope`、`fuse_rotary_positional_embedding`。

接入顺序建议：per_token_quant_int8（契约简单）→ paged_mqa_logits（有 emu
金标可直接对拍）→ moe/routing。

## 新增一个算子垫片的流程

1. **定契约**：找到社区调用点，抄下完整签名/布局/返回语义（读上游源码，
   不要猜；rope 教训：同一个函数名在 cache 布局上还有隐含约定）。
2. **写参考**：`torch` 档参考实现（fp32），这就是测试金标。
3. **写生产档**：triton 或 vendor 转发；遵守黄金法则（fp32 累加、无
   主机同步、import 不失败）。
4. **写契约测试**：按 `test-suite.md` 四维度；vendor 高危面同时写场景
   压测 + 注册快关（`kill-switches.md`）。
5. **更新矩阵**：本文件表格 + README 覆盖范围表。
