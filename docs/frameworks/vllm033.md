# vLLM 0.33 接入与验收边界

本轮目标是官方 `v0.33.0`，提交
`4ba787ec9959e566d3b587a796dff07673324da5`。Core 与
HYGON-AI/vllm-plugin-das 分别使用 `v0.33.0` 发布线；本仓继续在主线维护
公共机制，版本准入同时接受 epoch 为 0 的 `0.32.x` 和 `0.33.x`，包括
RC/dev/post/local 后缀。准入通过不表示 native ABI、算子或模型验收通过。

HCU 身份清单为 `gfx928`、`gfx92a`、`gfx936`、`gfx938`；设备属性中的
`:sramecc+:xnack-` 等后缀在匹配前移除。Core 的 DTK 构建清单须同步维护。
识别 `gfx92a` 不代表其数值、Graph 或模型验证已完成，也不扩大仅在
`gfx936` 验证的 FP8 linear fallback 等 provider 准入范围。

## 三仓定位

| 仓库 | 负责内容 | 接入位置 |
| --- | --- | --- |
| OpenDAS/vllm | 官方模型、调度、runner、通信及公共接口；必要的 DTK 构建兼容 | 官方 release 起步，逐文件说明构建差异 |
| adapter-das | 真实 HCU 探测、版本及 runtime ABI 准入、公共快关、扩展报告与进程健康 | 唯一 `vllm.platform_plugins` / `vllm.general_plugins` owner |
| vllm-plugin-das | 官方没有的 HCU 格式、算法与可选优化 | `hcu_adapter.extensions`，继而使用官方公开 registry |

本仓不得为通过一个 provider 检查而伪造 NVIDIA/CDNA 架构，也不得硬编码
Plugin 类路径。普通 MHA/GQA 的默认 Triton 选型不覆盖 MLA/Sparse MLA；
后两者仍由官方 ROCm priority 决定。显式 Flash 后端依赖当前 PID 的唯一
`hcu-flash-attention` owner 与健康检查，不能用另一个扩展的 ACTIVE 报告替代。

## v0.33 源码审计结论

官方保留 `AttentionBackendEnum` / `register_backend`、`AttentionImpl.forward`
和 `Platform.get_device_communicator_cls`。官方 Flash 实现没有旧 HCU fork 的
三个私有 hook：`_get_kv_cache_views`、`_get_kv_cache_write_views`、
`_run_flash_attention`。Plugin 应通过公开接口自有 provider glue，不能仅
修改版本门或复制官方 forward。构建缺少 gfx936 与 DTK Torch ABI 的问题应
在 Core 构建层处理，不加入本仓 sitecustomize 或设备身份逻辑。

官方 AITER 专家在 v0.33 按量化格式拆为具体子类；Plugin 的 INT8 provider
须自行声明精确的 channel-weight/token-activation 支持，不继承为全格式准入。
普通 AR 可审计公开 communicator 类选择入口；融合 AR 的注册、pass 与
Graph 生命周期仍需单独证明，不能由普通 AR 的可达性推定支持。

## 验证

BF16/FP16 AITER 的可选公开平台 ABI 候选与限制见
[unquantized AITER compatibility](vllm033-aiter-moe.md)。组件通过不代表模型验收。

CPU 快速回归：

```bash
HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/python -m pytest -q tests/adapter/test_vllm_bridge.py
```

真实宿主测试必须安装或显式指向本次 v0.33 产物，并核对父/子进程的模块
来源、distribution 与 entry points。不要让系统旧版 vLLM 或遗留平台插件
进入测试栈；共享机制的 portable 回归与真实宿主测试分别记录。

设备验收需覆盖 BF16/E5M2、实际 page/两种布局、非单位 scale、变输入同址
Graph、prefix、适用 MTP/TP、模型精度与预热性能。遵守 skill 的四个质量
维度与 vendor 快关流程；旧版结果不转授。当前文档记录源码准入设计，
不宣称 v0.33 整体可部署。

2026-11-03 安装态补充：当前 MR wheel 在独立 Core+Adapter 环境中完成真实
metadata、官方 general-plugin loader 与 config spawn/unpickle 共 17 项检查；
耗时约 397 秒。此前 240 秒总时限的中断结果保留，不计为通过。required
启动策略与 runtime ABI 共 30 项安装态检查通过；CPU 替身显式隔离无关
前置条件，runtime ABI 的真实官方 loader/无效库负控继续执行。


mHC 行为适配候选、启动配置、缓存/生命周期及未闭合门禁见 [v0.33 mHC](vllm033-mhc.md)。默认仍为原始官方，不代表已知 gfx936 TileLang 分支通过。
