# vLLM 0.32 HCU Adapter 部署与职责边界

## 1. 支持范围

本页保留 vLLM `0.32.x` 的部署与验证记录；v0.33 迁移见
[vllm033.md](vllm033.md)。当前公共版本门接受 epoch 为 0 的 0.32 与 0.33
patch line，包括 prerelease、dev、post 和 local build 后缀，但拒绝 `0.31.x`、`0.34.x`、非零
epoch 和无效版本。版本门通过后仍由 `RocmPlatform`、平台选择与 runtime 接口
检查能力；缺少必需 seam 时响亮失败。

已登记的 HCU 架构为 `gfx928`、`gfx936` 和 `gfx938`。平台选择前先通过 Hygon
PCI vendor、Torch HIP runtime、设备可用性和架构检查；非 HCU 主机不选择本平台。

## 2. 三组件结构

```text
official vLLM v0.32.0
        │
        │ vllm.platform_plugins / vllm.general_plugins
        ▼
adapter-das
  ├─ HcuPlatform
  ├─ vLLM 版本、HCU 硬件和 DTK runtime 门禁
  ├─ 官方已有行为的 HCU 兼容
  └─ hcu_adapter.extensions API v1
        │
        │ hcu_adapter.extensions
        ▼
optional vllm-plugin-das
  └─ DAS 可选优化实现
```

### vLLM 0.32

vLLM 负责引擎、调度、Worker、模型执行以及正式扩展接口。当前验证基线仍是官方
`v0.32.0`；HCU vLLM wheel 只保留无法经公开 seam 完成的构建或底层兼容修改。
Adapter 接受 0.32.x 不代表任意 patch wheel 已验证，交付仍须锁定并记录实际 commit。

### Adapter

Adapter 是唯一的平台和启动裁决点，独占：

```toml
[project.entry-points."vllm.platform_plugins"]
hcu_adapter = "hcu_adapter.bridges.vllm:hcu_platform_plugin"

[project.entry-points."vllm.general_plugins"]
hcu_adapter = "hcu_adapter.bridges.vllm:hcu_runtime_plugin"
```

Adapter 负责：

- 识别 HCU 并返回 `hcu_adapter.bridges.vllm.HcuPlatform`；
- 执行 vLLM 0.32、平台选择和 DTK/HIP runtime 门禁；
- 复用官方 `RocmPlatform` 行为；普通 MHA/GQA 在未显式选择 attention
  backend 时使用 vLLM 官方 Triton attention，MLA 与 Sparse MLA 则保留
  `None`，交由官方 ROCm backend priority 按配置选型；
- 为官方已有、但 HCU ABI 不一致的行为提供兼容桥；
- 在基线门禁完成后发现可选 DAS Extension；
- 保存进程内 Extension 激活报告。

### Plugin

独立 `vllm-plugin-das` 只声明 Adapter Extension：

```toml
[project.entry-points."hcu_adapter.extensions"]
das_hcu = "hcu_vllm_das.extension:create_extension"
```

Plugin 交付 `ExtensionSpec` 和可选优化实现，不得重复声明两个官方 vLLM Plugin
组，也不得复制 Adapter 的版本、平台或 DTK 门禁。上游没有的新量化、新后端、
新融合或模型专项机制归 Plugin；官方已有行为在 HCU 上的缺口归 Adapter。

## 3. 启动时序

```text
Python 启动
  -> sitecustomize.py
  -> hcu_adapter.bootstrap.install()
  -> import vllm
  -> vllm.platform_plugins 选择 HcuPlatform
  -> vllm.general_plugins 执行 Adapter runtime guards
  -> Adapter 发现 hcu_adapter.extensions
  -> 可选 Plugin 激活
  -> Engine 启动
```

必须在宿主 import 前生效的平台 seam 和 import 期常量只放 Adapter。Plugin
激活始终晚于 Adapter 的版本、平台与 DTK runtime 门禁。

## 4. 基线与失败语义

- **未安装 Plugin 且未显式请求 required 项**：`vLLM + Adapter` 保持官方基线路径；
- **可选 Plugin 干净准备失败**：Adapter 记录失败报告，基线继续启动；
- **显式 commit-fatal 失败**：已发布状态后不能安全继续时，owner 抛出公开
  `ExtensionCommitError`；宿主停止后续激活并锁存失败，即使该 owner 是 optional。
  详见[公开失败/健康 ABI](../vllm-commit-fatal.md)；
- **Extension API 调用方显式传入 required Extension/capability**：缺失时响亮
  阻断；默认 vLLM runtime context 当前不请求任何必选项；
- **不支持的 vLLM 版本或 HCU 架构**：不得选择 `HcuPlatform`；
- **进程内必需动作失败**：失败被锁存，必须使用新进程重试，避免继续使用半安装状态。

## 5. 安装与版本锁定

三件套必须以兼容锁固定准确 commit 和 wheel，而不是只记录分支名：

1. vLLM `0.32.0` HCU wheel；
2. 本仓 Adapter wheel（distribution 当前为 `sgl-kernel`）；
3. 可选 `vllm-plugin-das` Extension wheel。

建议安装顺序：

```bash
python3 -m pip install --no-deps /wheels/vllm-0.32.0+das*.whl
python3 -m pip install --no-deps /wheels/sgl_kernel-*.whl
python3 -m pip install --no-deps /wheels/hcu_vllm_das_plugin-*.whl  # 可选
```

兼容锁至少记录：

- vLLM、Adapter、Plugin 仓库、分支和 commit；
- wheel 文件名和 SHA256；
- Python、Torch、Triton 和 DTK 版本；
- HCU 架构与卡数；
- eager/graph 配置；
- 模型、长上下文、并发和精度结果；
- HumanEval 任务范围和结果。

不要把旧 Adapter 或旧 Plugin 组合的模型结果继承到新 commit；任一组件 SHA
变化都必须重新构建 wheel，并按受影响切片复验。

## 6. 验证

### Dense channel-FP8 fallback（gfx936）

DTK Torch 2.11 的 `torch._scaled_mm` 在 gfx936 上会由自身架构检查拒绝，
即使放宽 TorchFP8 linear helper 也不能执行。Adapter 不改变该判定、
`get_cdna_version` 或任何其他架构身份，而是在官方 general-plugin runtime
阶段通过 `register_linear_kernel(..., kernel_type="fp8")` 将官方已有
`TritonFP8ScaledMMLinearKernel` **追加到最后**。既有 AITER/tuned/native
优先级以及官方 `is_supported`、`can_implement`、禁用名单仍生效；不复制
kernel、模型或 QuantConfig。该已有行为兼容属于 Adapter，不是新优化 Plugin。

只有 Hygon PCI+HIP+gfx936 门通过才注册；gfx938/gfx928 和其他架构保持原逻辑。
gfx938 尚无本次实机证据，不继承 gfx936 的通过结论。注册和 provider 签名变更
会响亮失败。官方 native support 返回真时不再注册；官方 ROCm 候选列表将来
纳入该 fallback 后应删除本桥。可用已有
`VLLM_DISABLED_KERNELS=TritonFP8ScaledMMLinearKernel` 禁用，诊断登记可从
`hcu_adapter.registry.records()["vllm_fp8_linear"]` 查看。

独立数值/Graph探针（无模型，GPU3，allocator上限4GiB，非单位尺度与bias）：

```bash
python tests/tools/_probe_vllm_fp8_linear.py --output /path/to/new-provider-evidence
python tests/tools/_probe_vllm_fp8_family.py --output /path/to/new-family-evidence
```

使用 `env -i` 明确 DTK动态库、`HIP_VISIBLE_DEVICES=3`、`TMPDIR`/Triton cache
放 `/dev/shm`。第一个命令使用官方已安装环境、空PYTHONPATH直接验证原provider；
第二个仅将此worktree设为明确的PYTHONPATH，验证真实runtime注册、官方选型、
官方QuantFP8和apply_weights完整链路。二者都按解码FP8的FP32 GEMM+scale+bias
对拍，要求cosine≥.9999、NRMSE≤.01、max-abs/ref-max-abs≤.01；Graph至少三波
更换inputs/scales/bias。完整family参考复用官方quantizer，未重新实现量化。
模型加载/质量和性能另行验证，不将micro通过等同于模型通过。

### 6.1 任意开发机

```bash
HCU_ADAPTER_HOST=none python ci/l0_gate.py
HCU_ADAPTER_HOST=none pytest -q tests/test_ops_cpu.py tests/adapter
```

vLLM 聚焦契约：

```bash
HCU_ADAPTER_HOST=none pytest -q \
  tests/adapter/test_vllm_bridge.py \
  tests/adapter/test_vllm_extensions.py \
  tests/adapter/test_packaging.py
```

这些测试覆盖 0.32 patch-line 版本门、HCU 探测、平台与 runtime Plugin 顺序、进程失败锁存、
Extension API v1、可选/必需失败语义和 wheel entry-point 所有权。

### 6.2 HCU 节点

至少验证：

1. 只安装 vLLM + Adapter 时服务可启动；
2. Plugin 未安装时没有 Extension 依赖；
3. Plugin 安装后只有 Adapter 拥有官方 vLLM 两个 Plugin 组；
4. Plugin ON/OFF 数值 A/B；
5. 非 eager CUDA Graph capture/replay 变输入；
6. 多模型短文本、长上下文、并发和服务 API；
7. HumanEval 64 或本次发布要求的完整精度集合。

Attention 路由还须覆盖三类配置：普通 MHA/GQA 的默认值仍为
`TRITON_ATTN`；MLA 与 Sparse MLA 的默认值不得被 Adapter 改写为普通
attention backend。`fp8_e5m2` 等特定 KV dtype 是否可用，继续由最终选中的
官方 MLA backend 自己校验，Adapter 不伪造设备代际或 dtype 能力。

验收顺序固定为：

```text
vLLM + Adapter 正确
  -> vLLM + Adapter + Plugin 正确
  -> Plugin 组合相对基线有可量化收益
```

Plugin 跑通不能反向证明 Adapter 基线正确，性能提升也不能替代数值与 Graph
门禁。

## 7. 维护原则

通用成功加载回调、worker 可用性判定以及 compile-policy freeze 的集成边界见
[post-model-load 接口](vllm032-post-model-load.md)。此接口不激活 Plugin；foreign
worker 仍由操作者配置，依赖 freeze 的 Plugin 选型必须显式检查接口可用性。

- 官方 vLLM 已支持且能在 HCU 正确运行的能力直接复用，不重复建设；
- 官方已有行为的 HCU 缺口留在 Adapter；
- 上游没有的新能力留在独立 Plugin；
- 必须修改 vLLM 核心时先证明没有可用 seam，并推动最小修改上游化；
- Adapter 补丁数和官方单测 skip 率是长期治理指标。
