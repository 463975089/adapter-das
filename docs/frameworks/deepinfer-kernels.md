# DeepInfer 算子文件导航与兼容入口

算子实现位于 `hcu_adapter/kernels/`，按算子族与实现档位命名。
bridge 负责上游接口接线，catalog 记录实际绑定入口；不将模型策略移入算子文件。

| 算子族 | 实现位置 | 入口或兼容路径 |
|---|---|---|
| 量化 | `quant_emu.py`、`quant_lightop.py` | `quant.py`；旧 `lightop_ops.per_token_cast` 转发 |
| Gather | `gather_emu.py`、`gather_triton.py` | `gather.py` 校验并选后端；`elementwise.gather` 兼容导出 |
| RMSNorm | `norm_emu.py`、`norm_triton.py`、`norm_lightop.py` | `norm.py` 校验并选后端；`elementwise.norm` 兼容导出 |
| SwiGLU LightOp | `swiglu_lightop.py` | 旧 `lightop_ops.swiglu_*` 转发；原量化融合的 Torch 契约仍在 `quant_emu.py` |
| MoE gate | `moe_gate_emu.py`、`moe_gate_lightop.py`、`draft_router_triton.py` | `moe_gate.py`、`draft_router.py` 是兼容导入 |
| Top-K | `deepselect_emu.py`、`deepselect_lightop.py` | `deepselect.py`；LightOp 上游绑定暂保留 `lightop_ops.topk` |
| RoPE | `rope_triton.py`、`rope_lightop.py` | `rope.py`；旧 `lightop_ops.apply_rotary` 转发 |
| mHC | `mhc_emu.py`、`mhc_lightop.py` | `mhc.py` 模块别名；LightOp 上游绑定暂保留 `lightop_ops.mega_mhc` |
| Dense GEMM | `gemm_emu.py`、`gemm_lightop.py` | `gemm.py` 保留 dense NT 后端选择及旧符号导出 |
| Grouped GEMM | `grouped_gemm_emu.py`、`grouped_gemm_triton.py` | `grouped_gemm.py`；旧 `gemm.m_grouped_*` 导出保留 |
| LightOp MQA logits | `mqa_logits_lightop.py` | `lightop_mqa.py` 保留四个上游入口；dense packing 与 sparse/paged 实现在同一处 |
| LightOp 调用统计 | `lightop_usage.py` | `lightop_ops.usage()` 读取同一份计数，不另建计数器 |

`lightop_ops.py`、`lightop_mqa.py`、`elementwise.py`、`elementwise_triton.py`
不再存放计算实现。旧的 `lightop_gemm.py`、`triton_elementwise.py` 等转发路径继续兼容，
仓内新调用应引用上表的入口或具体档位，不向兼容文件添加计算代码。

## 为什么保留少量旧绑定

当前独立 plugin 会检查以下函数的 `__module__`，并检查 MQA 的 `output_dtype` 参数：

- `lightop_ops.topk`、`lightop_ops.mega_mhc`；
- `lightop_mqa.fp8_fp4_mqa_logits`、`fp8_fp4_sparse_mqa_logits`、
  `fp8_fp4_paged_mqa_logits`、`fp8_fp4_paged_sparse_mqa_logits`。

这些函数保留原模块身份和显式签名，函数体只转发到规范文件；catalog 与实际绑定保持一致。
其他迁移项的 bridge 和 catalog 直接引用规范路径。
必须先迁移 plugin 的能力检查及外部导入，再移除这些兼容入口，不能仅因仓内没有引用就删除。
`mhc.py` 使用模块别名，保持旧路径属性修改与实际实现模块的共享状态。

## 变更与验证范围

本次只整理上述算子文件及引用。52 个迁移函数的 AST 在忽略导入路径和函数重命名后一致，
旧 LightOp 入口参数签名保持一致。Dense NT GEMM 单独保留原校验、空维处理与后端选择：
非空、连续、K 为 32 倍数的 BF16 输入调用 LightOp；其他已支持情况调用 Torch 实现。

不引入新的融合 kernel，不更改默认后端、量化边界、缓存布局或候选档位的验收状态。
本地执行项目 L0、Python 语法检查及 `git diff --check`；没有新增单测或运行 GPU 作业。
这不等价于新目录组合的整模型验收，部署切换仍需主线组合验证。

本页仅覆盖此次整理的算子族，不表示 `mtp.py`、`index_ops.py` 等其他历史混合模块已完成归类。
