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
| vllm-plugin-das | 官方没有的 HCU 格式、算法与可选优化 | `hcu_adapter.infra.extensions`，继而使用官方公开 registry |

本仓不得为通过一个 provider 检查而伪造 NVIDIA/CDNA 架构，也不得硬编码
Plugin 类路径。普通 MHA/GQA 的默认 Triton 选型不覆盖 MLA/Sparse MLA；
后两者仍由官方 ROCm priority 决定。显式 Flash 后端依赖当前 PID 的唯一
`hcu-flash-attention` owner 与健康检查，不能用另一个扩展的 ACTIVE 报告替代。

`infra/` 与 `features/` 的布局迁移须同步目标 Plugin 的静态导入、动态
`_seams` 模块表和 provider 身份检查。Adapter 自身单测不能替代组合验证：
使用目标 Plugin wheel 的真实 metadata 加载扩展 factory，并检查 RMS 等延迟
导入；保持 discovery 不导入宿主或 Torch。延后某个模型的 checkpoint 测试，
不意味着可跳过它的扩展发现契约。目录归属规则不以旧路径转发壳绕过。

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

## 可选 Decode 算子的能力声明

包的文件摘要只用于缓存失效和验证记录。实现方可在负责该算子 ABI 的模块上公开 `HCU_OPERATOR_CONTRACTS` 映射，以算子名和整数版本声明兼容范围。兼容重编译沿用契约版本。改变布局、调用语义或数值边界时升级契约版本。适配层仍校验实际入口、设备和张量布局。缺少声明或有效预检结论的路径使用原有实现并记录原因。

| 契约 v1 | 声明模块 | 声明前应核对的行为 |
| --- | --- | --- |
| `sparse_decode_plan_h64` | FlashMLA 原生扩展 | H64 OCP FP8 sparse decode 支持外部 `int32[64,8]` 调度表与 `int32[B+1]` split 前缀和。核对分块坐标、空分区、跨行标志和 SWA/extra counts 的解释。 |
| `mla_bf16_inverse_rope` | vLLM ROCm MLA ops | 512 维输出分为 448 维 NoPE 与 64 维 GPT-J RoPE。校验 BF16 写回、FMA 舍入、零行及有符号零。 |
| `dsv41_q_rope_and_padded_kv_insert` | vLLM 原生扩展 | 分拆调用的 Q RoPE 与融合入口一致，原生 KV insert 保留 584 字节页、padding 与写回语义。 |
| `dsv41_wo_a_bf16_consumer_layout` | vLLM ROCm MLA ops | WO-A 与 WO-B 之间接受 contiguous token-major BF16，权重缓存和 INT8 下游消费布局一致。 |
| `rocm_bf16_bmm_einsum_parity` | Torch | 支持该 WO-A 形状和 stride 的 BF16 `bmm(out=...)` 与 `einsum` 逐字节一致。 |
| `dsv41_greedy_markov` | vLLM DSpark draft 模型及 speculator | 核对逐步 bias、greedy tie-break、anchor 类型及 token 映射，覆盖同一 Graph 下变更输入。 |
| `dsv41_draft_final_mhc_post` | vLLM DSpark draft 模型 | 核对 final Post 的调用位置、参数语义和 BF16 舍入。 |

生产包发布能力声明前须用实际安装构建完成对应的原生对照及变输入 replay。镜像缺少声明时，启动预检可以确认实际行为。FlashMLA Worker 在模型加载后比较调度字段、split 前缀和、attention 输出和 inverse RoPE 写回。插件在构造期比较 Q/KV 输出、WO-A 下游消费张量和 draft final Post 数值。在 draft 权重加载后，Markov 预检使用实装模型的 Markov head 和原生逐步采样比较五步 token。LightOp clamped SwiGLU INT8 在构造期比较有效行的量化值与 FP32 scale。预检通过结果供同一 Worker 的对应设备或模型使用，Graph 捕获和回放复用该结果。DSpark LL receive 复用 AG/RS 的公开 per-token INT8 量化，借助原有 counts 保持 masked GEMM 的有效行语义。公开量化入口会处理缓冲区的 padding 行，这条路径的吞吐和 TPOT 需要重新测量。

## 普通 FP8/INT8 Linear 的 LightOp 接入

通用 scaled-MM 执行桥见 [LightOp Linear](vllm-scaled-mm-lightop.md)。保留官方
FP8 fallback 的公开注册、内核优先级、权重加载、量化和 INT8 非对称修正；
仅将选中的 Triton provider 的共享 GEMM 调用点路由至 LightOp。
`ops.vllm_scaled_mm` 支持 `auto`（默认）、`lightop`（要求 wheel）和 `triton`。
`HCU_DISABLE_OPS=vllm_scaled_mm` 在新进程中跳过接入。INT8 覆盖 gfx936/gfx938，
E4M3FN 仅 gfx938；不覆盖布局/尺度/对齐时保留官方 Triton 行为。候选状态不表示
DSV41 整模型或性能验收完成。该后端不处理 CT W4A8 MoE 的 INT4 packing/HIPC。

## 验证

### PCP 逆 RoPE

配置 `ops.vllm_inverse_rope: hcu` 选择 Plugin 的 BF16 输出逆 RoPE，默认 `stock`。
Adapter 在构造 PCP 注意力类前接收 Plugin 的 `forward_mqa` 实现。
该配置要求启用 PCP 模型扩展。`HCU_DISABLE_OPS=vllm_inverse_rope` 选择原实现，
策略变化须重启进程。Prefill 的压缩 KV 和 SWA KV 均沿用 Core 的反量化路径。

HCU 算子处理尚未旋转的 BF16 输出尾部，内核资源位于 Plugin wheel 中。
显式选择后的缺失依赖在模型构造前拒绝，形状不适配时使用 Core 逆 RoPE。

FlashMLA prefill/decode 的隐式 `native` 默认只在实际 vendor wheel 与调用 ABI
均可用时接入；缺包、缺符号或签名漂移保留官方 stock，且不锁存启动失败，
不应让普通 BF16 模型依赖可选 V4.1 provider。环境或配置中显式选择 `native`
仍严格拒绝不可用的 provider；发布后健康检查与失败锁存保持不变。

原生 packed decode 测试从有限 BF16 KV 经官方 cache writer 生成 page-flat
FP8/UE8M0 数据，并在每次改输入 Graph replay 后对拍独立 CPU FP64 attention。
CPU recorder 使用合法的固定尺度 packed 夹具，另验实际被选中的 KV 为有限值。
这些 CPU 契约与测试代码不等于原生 decode、Graph 或 V4.1 模型验收。

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

## KV cache placement extension seam

The HCU platform forwards the official `get_kv_cache_config_builder_cls`
selection through `hcu_adapter.bridges.vllm_kv_cache`. An extension registers
one named pair of callbacks with `register_kv_cache_layout`. The first
selects a qualified builder class or returns None for the native builder.
The second transforms the worker-local KVCacheConfig before the Adapter's
existing Worker delegates to official `initialize_from_config`.

The registry rejects conflicting owners and malformed callback results. An
extension checks its registered ownership through `require_kv_cache_layout`
in its existing health callback. No provider preserves the normal platform
selection and config object. The registry imports no framework, owns no
placement algorithm and contains no Plugin class paths. Tests are in
`tests/adapter/test_vllm_kv_cache.py`, with real platform/worker entry-point
coverage in Plugin's `tests/vllm033/test_dsa_cache_layout.py`.
# Layer Split migration (2026-11-17)

The generic KV builder/localizer registry is replayed on official main. The
current PCP attention remains in `hcu_adapter.kernels.pcp_fork`; Layer Split's
attention subclass, independent pools, scheduler and transfer protocol remain in
Plugin. `set_kv_transfer_validator` binds one optional validator for the extension
contract. Without one, PCP transfer is refused as before; the callback must
return None or raise, and does not bypass the other native PCP checks.
The Plugin validates its exact owner-aware connector and rejects unqualified
FP4/stock/kill-switch combinations. This interface migration is CPU-validated;
historical GPU results do not qualify the migrated commits.
