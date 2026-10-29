# DeepInfer 通信与输入／状态接口适配

本批主要承接 DeepInfer 上游的调用契约，不宣称这些接口在不同框架间直接通用。
实现位于 `hcu_adapter/kernels/`，接入清单位于
`hcu_adapter/ops/deepinfer_aux_catalog.py`；后续宿主 bridge 负责注册与安装。

| 分类 | 内容 |
|---|---|
| 通信 | EPBuffer dispatch/combine、BucketBuffer/Session CP 通信、域与对齐查询 |
| Engram | GPU 表访问、hash、gate、权重和 offset 准备 |
| 输入 metadata | position、block table、sequence length、token/Engram 输入 gather |
| 采样和约束 | fused/random sampling、bitmask、异常值检查 |
| draft / verify | 候选映射、接受判定、置信度、DFlash/MTP 输入和状态更新 |
| 视觉 | 固定 ViT varlen attention、embedding 替换、patch 和下采样 |

EP 桥目前限制单机最多 8 rank；CPU RDMA Engram 与部分通信参数尚未接通。
既有 handle 可用于 combine，cached dispatch 的 handle 复用仍拒绝。
固定 shape/dtype 与 Graph 条件以 provider 的参数检查为准。本分支未完成节点验收。
