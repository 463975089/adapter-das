# DeepInfer 算子文件导航

算子实现位于 `hcu_adapter/kernels/`，按算子族与实现档位命名。
bridge 负责上游接口接线，catalog 记录实际绑定入口；模型策略保留在 plugin。

| 算子族 | 实现位置 | 入口职责 |
|---|---|---|
| 量化 | `quant_emu.py`、`quant_lightop.py` | 直接引用所需实现 |
| Gather | `gather_emu.py`、`gather_triton.py` | `gather.py` 校验并选择后端 |
| RMSNorm | `norm_emu.py`、`norm_triton.py`、`norm_lightop.py` | `norm.py` 校验并选择后端 |
| SwiGLU | `quant_emu.py`、`swiglu_lightop.py` | 保留原融合量化契约 |
| MoE gate | `moe_gate_emu.py`、`moe_gate_lightop.py`、`draft_router_triton.py` | 路由评分与 gate 实现 |
| Top-K | `deepselect_emu.py`、`deepselect_lightop.py` | bridge 直接绑定实际实现 |
| RoPE | `rope_triton.py`、`rope_lightop.py` | 直接引用所需实现 |
| mHC | `mhc_emu.py`、`mhc_lightop.py` | bridge 直接绑定实际实现 |
| Dense GEMM | `gemm_emu.py`、`gemm_lightop.py` | `gemm.py` 保留 dense NT 后端选择 |
| Grouped GEMM | `grouped_gemm_emu.py`、`grouped_gemm_triton.py` | 直接引用所需实现 |
| LightOp MQA logits | `mqa_logits_lightop.py` | dense packing 与 sparse/paged 实现在同一处 |
| LightOp 调用统计 | `lightop_usage.py` | 所有 LightOp 实现共享同一份计数 |

旧算子别名与薄转发文件已删除。导入、bridge 绑定和 catalog 统一使用上述路径。
其他已命名的档位模块（例如 `context_parallel_rccl.py`、`compress_emu.py`、
`packed_indexer_triton.py`）也直接引用，不再通过无档位的旧模块名转发。

## Plugin 同步迁移

使用这些内部模块的 plugin 必须与本次 adapter 变更配套更新：

- 导入改为实际实现模块，包括 grouped GEMM、RoPE、量化、CP、压缩和索引打包。
- Top-K、mHC、MQA 的 provider 模块检查分别改为 `deepselect_lightop`、
  `mhc_lightop`、`mqa_logits_lightop`；MQA 保留原有 `output_dtype` 参数检查。
- LightOp 计数从 `lightop_usage.usage()` 读取。

不能单独将这版 adapter 覆盖到仍引用旧路径的 plugin 环境。
上游 `deep_gemm`、`tile_kernels` 等宿主接口名称不变；变更的是 adapter 内部实现路径。

## 变更与验证范围

本次整理文件及引用，不引入新的融合 kernel，不更改默认后端、量化边界、
缓存布局或候选档位的验收状态。Dense NT GEMM 保留原校验和空维处理：
非空、连续、K 为 32 倍数的 BF16 输入调用 LightOp；其他已支持情况调用 Torch 实现。

本地执行项目 L0、Python 语法检查及 `git diff --check`；没有新增单测或运行 GPU 作业。
目录和引用变更后的 adapter/plugin 组合仍需整模型验收。
本页仅覆盖此次整理的算子族，不表示 `mtp.py`、`index_ops.py` 等其他历史模块已完成归类。
