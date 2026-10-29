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


## 本轮更名与分类

下表列出在已整理版本基础上补齐的项目。旧名仅作为迁移说明，不是兼容入口。

| 旧文件或分组 | 当前位置 | 分类与理由 |
|---|---|---|
| environment、ops config/registry/kill_switch、桥内 config/registry 壳 | 公共 `hcu_env`、`hcu_config`、`registry`、`kill_switch`；DeepInfer 清单直接引用 `ops.deepinfer_catalog` | 机制：删除跨层别名，两个 catalog 保持原有不同职责 |
| 桥内 deep_select | `bridges/_deepinfer/deepselect.py` | 绑定：内部拼写统一；上游 `deep_select` 包与算子键保留 |
| bucket | `bucket_rccl.py` | 单档位实现：RCCL CP 集合通信 |
| deepep_compat | `deepep_dtk.py` | DTK DeepEP 适配实现：保留事件、布局与 Graph 协议 |
| native_flashmla | `mla_attention_dtk.py` | DTK 原生 MLA 实现：与 `mla_attention_emu.py` 共用词根 |
| verification | `verification_emu.py` | 单档位实现：Torch 验证与采样后处理 |
| mtp | `mtp_emu.py`、`mtp_triton.py` | 按原函数拆分 Torch 运算与提案写入 Triton 内核；未引入新融合 |
| state_and_constraints | `state_and_constraints_emu.py`、`state_and_constraints_triton.py` | Torch 状态搬运/诊断与 Triton mask 分开 |
| vision | `vision_emu.py`、`vision_triton.py` | Torch embedding/downsample 与带 Triton 归一化的图像打包分开 |
| index_ops 内计算 | `index_ops_emu.py`、`index_ops_triton.py` | 单档位实现；`index_ops.py` 仅保留实际后端选择 |
| fused_transform 内单档位 helper | `kv_cache_emu.py`、`rope_emu.py`、`rope_lightop.py` | KV 数值实现及 RoPE 计算归档；组合执行和 plugin 接点留在原门面 |
| gemm、gather、norm、fused_transform | 原文件保留 | 门面：实际校验、分发或按原阶段组织执行，不是旧路径薄转发 |
| runtime_config、lightop_usage | 原文件保留 | 机制豁免：配置状态与共享计数；登记在 governance §3b 和 L0 |
| 先前 lightop_mqa 整理结果 | `mqa_logits_lightop.py` | 已符合算子＋档位规则，不再重复更名为 mqa_lightop |

已发布的 `torch` 后端选择值保持不变；它与 `_emu` 文件档位是两个概念。
更改配置默认值、算子键或删除旧配置值不属于本次结构规范化。

## 桥层归属核验

| 文件 | 留在 bridges / 移入 kernels | 理由 |
|---|---|---|
| `tile_kernels.py` | 留在 bridges | 创建上游包和符号、选择并登记 provider；当前没有独立计算函数 |
| `tilelang_backend/factories.py` | 留在 bridges | 契约 factory 导出、未实现项的报错闭包；没有数值实现 |
| `tilelang_ops.py` | 留在 bridges | factory 参数校验、布局契约检查和调用闭包；计算已引用 kernels |
| `deepselect.py`、`deep_ep.py`、`vendor_imports.py`、`vision_attention.py` | 留在 bridges | 分别安装 top-k、通信、DeepGEMM/MLA、视觉的上游接口绑定 |
| `lightop_gemm.py`（桥层） | 留在 bridges | 安装 DeepInfer Linear 接口；不是已删除的 kernels 旧名壳 |
| `bootstrap.py`、`worker_runtime.py`、`native_extension.py`、`platform_guard.py` | 留在 bridges | 安装顺序、worker 生命周期、扩展加载与平台契约 |
| `configuration.py`、`hcu_env.py`、`runtime_dump.py` | 留在 bridges | 宿主配置翻译、宿主环境规则和记录；公共 registry/config 直接引用 |

Plugin 还须同步公共 registry 引用、MLA 计数模块名，以及稀疏索引覆盖目标
`index_ops_emu`；其余模型逻辑不随目录调整改变。

融合变换门面的 LightOp 量化及 UE8M0 scale 解码由 `quant_lightop.quantize_transform` 承担；
门面只保留后端条件、阶段顺序和输出写回。该抽取保留原有运算及转换顺序，
最终提交只通过本地静态检查，不能把此前冻结组合的 GPU 验收直接视为新提交的验收。
