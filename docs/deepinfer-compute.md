# DeepInfer 契约的公共计算实现

本批提供可独立导入的计算实现，不安装宿主桥。公共指实现不依赖宿主模型类，
不表示 SGLang、vLLM、DeepInfer 的同名接口或张量布局可以直接互换。

| 分类 | 内容 |
|---|---|
| GEMM | BF16、block-scaled FP8/FP4、einsum、contiguous grouped GEMM |
| 量化与激活 | per-token cast、SwiGLU、scale 打包和转置 |
| Norm / RoPE | norm、rotary、融合 norm/RoPE/Hadamard/KV store |
| MoE / mHC | 路由 gate、top-k、mHC mix/Sinkhorn/norm 计算 |
| Attention | MQA logits、MLA prefill/decode、投影布局转换 |
| 索引和 KV | 稀疏候选、索引转换、KV 读写、压缩与状态更新 |

接口与 provider 清单在 `hcu_adapter/ops/deepinfer_compute_catalog.py`，详细
shape/dtype/scale 条件以实现为准。后续 DeepInfer bridge 负责将该目录并入共同
catalog 和注册具体 provider；本批没有自动替换任何模型或启用后端。
旧模块名是兼容转接，指向同一份实现。这里不包括新增 channel-FP8 模型方案。

本分支做静态检查；尚未进行迁移后的节点数值、Graph 或性能验收。
