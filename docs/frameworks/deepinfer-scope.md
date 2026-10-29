# DeepInfer 首版接入范围

本次提交定位为 **DeepInfer 上游接口的 HCU 适配接入**。参照 DeepInfer
`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`，不包含完整 DeepSeek V4.1 优化方案。
SGLang、vLLM、DeepInfer 的模型扩展遵循相同边界：兼容上游已有行为归 adapter；
新增量化、模型和执行方案归独立 plugin。代码可复用并不自动成为迁入 adapter 的理由。

## 本次保留与排除

| 接口或行为 | 本次归属 | 处理方式 |
|---|---|---|
| 设备可见性、worker 启动、CPU affinity、原生扩展加载 | adapter / DeepInfer bridge | 对接共同 bootstrap；不导入其他宿主模型 |
| DeepGEMM BF16、原有 block-scaled GEMM、grouped GEMM | adapter | 保留上游参数与 scale 契约；不是 channel-FP8 新后端 |
| TileKernels / TileLang norm、RoPE、量化、mHC、采样 | adapter | 实现固定上游入口；不重新选择模型各层的精度 |
| MQA/MLA、稀疏索引、top-k、KV 读写 | adapter | 保留 DeepInfer 原有语义；不强制与 SGLang/vLLM 共享缓存布局 |
| position、block table、sequence length、draft/verify metadata | adapter | 实现上游已有 factory，包括同一接口内的 Triton 实现 |
| DeepEP / CP、Engram GPU 表 | adapter | 对接通信库；未覆盖模式显式报错 |
| channel-FP8 checkpoint 加载及 dense/grouped/masked GEMM 新方案 | plugin | 不随此 PR 导入 |
| SGLang 数值边界、原生 KV 页及新增 bounded replay 策略 | plugin | 不随此 PR 导入 |
| 新方案下的 normal prefill / low-latency decode 编排、DSpark 模型绑定 | plugin | 不随此 PR 导入；上游已有 draft/verify 算子兼容仍属于 adapter |
| plugin 新增的 block32 到 token/channel 量化转换 | plugin | 不因辅助函数可复用就扩大首版范围 |

## 目录与依赖

- `hcu_adapter/bridges/deepinfer.py`：轻量宿主入口。
- `hcu_adapter/bridges/_deepinfer/`：固定上游接口的 factory、别名、配置和生命周期适配。
- `hcu_adapter/kernels/`：不依赖宿主的 DeepInfer 契约实现；具体实现按档位命名。
  旧文件名仅作模块转接，共用同一份实现和状态。此命名空间避免导入
  `sgl_kernel.__init__` 时提前加载与 DeepInfer 无关的 AOT/stub 注册。
- `hcu_adapter/hcu_env.py`、`hcu_config.py`、`registry.py`：唯一配置和注册状态。
  直接使用公共 `hcu_env`、`hcu_config`、`registry` 与 `kill_switch`，不经跨层转发。
- `hcu_adapter.registry.replace`：登记显式安装的外部 plugin 覆盖及原提供者；
  adapter 不导入、不自动启用 plugin。

相同算子名字不保证相同 ABI。DeepInfer 的完整限定接口单独登记，只有 dtype、scale、
stride、mask、输出及缓存语义一致时才能复用别的宿主的 provider。
一个进程只安装一个宿主桥；不会先安装 SGLang 桥再叠加 DeepInfer。

## 当前限制

- EP 桥主动限制单机最多 8 rank；跨机和自定义 QP 参数尚未接入。
- 当前 dispatch 产生的 handle 可供 combine 使用；拒绝把旧 handle 再送入 dispatch。
  参照版本的模型主路径并未使用这种 cached dispatch，不能据此称模型缺少该功能。
- 非默认 `num_sms` 等参数受桥的契约限制；这不是底层 DeepEP 不支持调优配置的结论。
- Engram 支持 GPU 表，CPU RDMA 表未接入；batch-invariant 模式显式拒绝。
- 部分接口仅覆盖参照模型使用的 shape/dtype；catalog 中 candidate 不等于全部功能已验证。
- 原生扩展需在配套 Linux/DTK/PyTorch 环境构建，本仓不分发二进制或权重。

## 验收与迁移

本次本地整理只进行静态检查。旧工作区的算子或 adapter + plugin 组合结果仅是历史
证据，不证明本分支在新上游基线的运行结果。正式切换默认部署前，仍需在配套节点
核验 adapter 的上游路径，以及独立 plugin 组合的加载、精度、Graph 和性能回归。

不将 plugin 专属优化纳入首版，也不以补全多机、CPU RDMA、batch-invariant 为首版
前置条件。保留旧部署直到新组合通过验证；旧路径转接不是第二套 adapter。
