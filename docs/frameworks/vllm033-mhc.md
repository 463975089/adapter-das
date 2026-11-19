# vLLM 0.33 mHC 的 HCU 行为适配候选

该候选恢复官方已有的 mHC pre/post-pre 语义，归 Adapter；Plugin 不增加重复
owner。Core 运行时代码、原始 `MHCPostOp` 和 `HCHeadOp` 保持官方实现。

当前默认配置：

```yaml
version: 1
ops:
  vllm_mhc: aiter
```

调试覆盖 `HCU_VLLM_MHC_IMPL=aiter|torch|stock`，默认 `aiter`。
`HCU_VLLM_MHC_IMPL=stock` 或 `HCU_DISABLE_OPS=vllm_mhc` 仍可显式回退到原始官方路径。
`HCU_DISABLE_OPS=vllm_mhc` 在启动时优先恢复原始官方路径；它不证明已知失败的
原始 TileLang pre/fused 正确。注册后不能在原进程切换，须启动新进程。
仅准入 gfx936/gfx938、vLLM 0.33 release line；不扩大平台/DTK 通用门禁。

AITER 模式的 delayed Pre 保持两段式：stage1 使用 `mhc_pre_gemm_sqrsum`，stage2
默认使用 TileLang `pre_big_fuse_tilelang`
（`VLLM_HCU_USE_AITER_TILELANG_MHC=1`）。显式设为 `0` 仅用于诊断旧
`mhc_pre_big_fuse`；默认开启但 TileLang 不可用时回退官方 Torch，不落回旧实现。
支持条件内 `hidden=5120` 的 Post 已可由 AITER provider 执行；与 Torch
存在已知 BF16 舍入差异，不能称逐位一致。`HCHeadOp` 保持官方实现。

## 实验性 DeepGEMM shifted 接缝

`HCU_VLLM_MHC_MEGA` 接受显式 `0` 和 `1`。在 gfx938 的 AITER 模式下，已验证的
vLLM 构建 `g7364da` 和 DeepGEMM 构建 `g793351` 默认开启，其他构建保持原路径。
显式 `1` 需要 aiter 模式、gfx938、vLLM 0.33 和包含 Mega mHC 的 DeepGEMM 构建。
核对过的供应商源码为
`090ed77f489644c36388dc0bbc348011cfa18a04`（`publish/ys_develop_Mega_MHC`），
包含公开的 `mega_mhc`、`mega_mhc_prepare_fn` 和对应 gfx938 原生实现。
历史安装包为 `deepgemm 2.1.0+dtk2610.torch2130.2611122142`（DTK 26.10 / Torch 2.13）。
供应商 main 的 `fc246fc4e4104c55a16b88d2f5682be54690469c` 缺少这两个入口。
显式启用时，平台、导出或 ABI 不匹配会停止启动。`HCU_DISABLE_OPS=vllm_mhc`
仍可在启动时强制使用官方路径。

只替换带 sublayer_out、carried pre、`x=None`、`norm_weight=None` 的 post+pre
接缝。Mega 的 x 是 sublayer_out，投影对象为更新后的 full-width residual。
参数沿用 SGLang `a389ea5c7f38c18e09927eaf7b4f4eb35e229a42` 中
`mhc_mega_boundary` 的映射，post 倍数仅传入一次。Mega 的 y 包含 decoder
RMSNorm，因此丢弃 y，使用原有 collapse 生成未归一化 layer。不符合条件的调用
继续执行原有 post 和 delayed Pre，并保留 `HCU_VLLM_MHC_DELAYED_FUSION`
的独立门控。已选中的 Mega 算子执行失败后锁存错误。
全 1 norm 权重、rmsnorm_eps=1e-6、rmsnorm_scale=1.0 仅因 y 一律丢弃才无害；
将来若使用 y，三者必须重新定值并重跑对拍。y scratch 按 (device,tokens,hidden)
缓存，同形状再按 stream 隔离，避免并发写竞争；缓存保留到 runtime 销毁，
不逐出仍可能被 Graph 引用的存储。多形状/多 stream 会增加常驻 scratch 内存。
缓存只去掉重复分配，mega 内 mandatory norm 计算仍存在。

Provider 的 Python/C++ 绑定、ABI、实现摘要及开关进入发布检查和编译缓存因子。
Fn 预打包缓存绑定张量身份和 mutation version；没有版本计数的 inference
权重每次重新打包，避免静默复用陈旧数据。记录 repack/reuse/versionless 计数，
首次 versionless 给一次告警，第 128 次 Python 下发输出一次 warmup 汇总，
可用 cache_stats() 取得当前值；计数不含 Graph replay，应结合 trace 的打包 kernel
核实是否把重打包捕获进图。加载/更新静态权重后若需 Graph
必须重新预热和捕获；不提供已捕获 Graph 的热更新保证。

历史候选 `abe53ac` 在 gfx938 上用十五种合成形状验证了 opaque op 和变输入 Graph
回放。Mega 与 AITER 的 BF16 post 写出存在舍入差异，会传播到下游系数。
本补丁仍需独立验证 GPU、完整 Graph、服务精度及同配置性能。
启用 delayed fusion 时，旧路径的 pre 为 gemm_sqrsum/big_fuse/pre_mix_collapse
三颗，Mega 路径为 main/reduce/collapse 三颗。关闭 delayed fusion 时，旧路径
的 gate 和 collapse 分开执行。应按相同接缝范围统计，并通过同节点、同配置的
服务 A/B 评估性能。

## 调用与生命周期

唯一 general-plugin 在规范版本、真实平台与 native ABI guard 后安装。
通过官方 `CustomOp.register_oot` 发布 pre/fused 子类，opaque Torch op
调用原始 `mhc_pre_torch` / `mhc_post_torch`，不复制数学或重绑 dispatcher。
官方 native 方法虽接受 `norm_weight/norm_eps`，本身不执行 RMS，故显式委托
公开 `ir.ops.rms_norm`。fused 返回 BF16 post，再执行 pre，保持舍入边界；
这是语义组合回退，不声称执行快速 fused kernel 或获得性能提升。

启动检查完整 HIP 参数/默认值、官方函数 ABI 和原 constructor 选择。
公共 constructor allocation 检查真实 OOT 类，无需读取/修补私有注册表。
健康检查发生在宿主 config/选型/load 边界和 opaque op 执行时；constructor
替换、cache getter 替换、启动策略变化均拒绝且锁存，恢复原对象不自动解锁。
发布及部分导入失败先锁存，再安全格式化异常，重试不修复已污染进程。
直接重试安装时也须检查配置解析失败：已注册后遇到非法策略先锁存再抛错，
恢复原策略仍拒绝；首次注册前的纯配置错误不发布注册，也不污染进程。

既有 framework-free `ProcessLocalLifecycle` 管理注册与 fork 边界；宿主桥
保留 vLLM 导入与身份准入。成功 fork 后经 child runtime 激活重新确认继承
注册；spawn 重新注册。未完成/失败激活继承到子进程时拒绝。
共享生命周期模块没有新增宿主依赖或第二份实现。

派生 getter `VLLM_HCU_ADAPTER_MHC_IMPLEMENTATION` 将实现摘要纳入普通和
模型 AOT 缓存：本桥、官方 mHC Torch、IR RMS 源码与 Torch/HIP ABI 身份，
路径不参与。完整 wheel/SHA 作证据，不作精准运行绑定。stock/启动快关保持
相同无注册因子，torch 改变两层 hash；开启 env cache 后也保留 getter。
该派生键不是操作员配置项。

## 已有证据与剩余门禁

2026-11-10，gfx938、DeepSeek-V4.1-Flash W4A8、`--enforce-eager`：

- 原 AITER `mhc_pre_big_fuse` 在单 token 和奇数 token 的 Sinkhorn 阶段输出约
  `0.25` 的均匀值；真实 dump 输入下 `comb` 最大相对误差约 `0.75`。
  替换为 TileLang 后，m=1/2/3/5/7/16/76/128 及真实 seq320 输入均通过，
  `comb` 最大相对误差不超过 `2.75e-7`，stage2 耗时约 13.0–13.7 us。
- HumanEval 全量 164 题官方报告为 `0.9756`（160/164）；空输出、连续重复、
  非 ASCII、超长及请求超时均为 0。平均端到端延迟 58.96 s；同配置仅将
  big_fuse 回退 Torch 的诊断臂为 83.68 s。该性能对比只适用于本次 eager 配方。
- 本轮证据不覆盖 CUDA Graph、其他量化/模型或 gfx936 的 TileLang 复验；
  catalog 保持 candidate，不据此标 validated/production。

2026-11-04，真实 gfx936、Flash H4096/Pro H7168 layer0 attn/FFN 权重：

- 原始官方 post/head 可复用。pre 投影/平方和约为独立参考的2倍；normless
  pre 有非 warp-uniform barrier 编译失败，normed pre/fused 数值失败。
- vendor/公开 OOT 小批次通过不替代扩展形状：1024-token Flash norm-off
  fused 有2个元素/状态超原 `.01/.01`，新进程/缓存复现。未舍入 post
  参考通过、存储 BF16 post 参考仍失败；不改门槛或自动接入 vendor。
- 官方 Torch OOT pre/fused 各32 cases/128 states，tokens0/4/128/1024，
  共144次非空变输入 Graph、48次无 kernel 空 Graph通过。完整图编译、
  compiled/eager 与 Graph/eager 字节一致、FP64/native 对拍、输入/权重
  不变。官方 head20/80通过；MTP head 权重不代表 MTP 实测。
- 首次候选 wheel 的真实 general-plugin/native guard 与上述64 cases通过。
  CPU 安装态 spawn/fork、两层 hash、env-cache、constructor drift通过；
  CPU lifecycle 使用显式 hardware double，不是设备准入证据。

仓内回归 `tests/adapter/test_vllm_mhc_bridge.py`；安装态手动探针
`tests/tools/_probe_vllm_mhc_lifecycle.py`。独立目录保留失败及源码/权重/hash。
capture 不等于执行 Graph：先 replay 原输入确认基线，再同地址换输入。
空 Graph 单列，不计为变输入计算覆盖。

最终安装包复验（2026-11-04）：源码、wheel 与隔离安装目录的6个生产
Python 文件逐字节一致。pre/fused 各32 cases/128 states通过；每个case
先实际 replay 原输入并与独立参考及 eager 对比，再执行3次变输入 replay。
共64个首次 replay 基线、144次非空变输入 replay、48次空 Graph replay，
256次 compiled/eager 对比。原始 post/head 类保持官方。空 Graph 不计为
变输入计算覆盖。首个 pre 探针漏掉首次 replay 的记录保留，由修正后的
独立 attempt2 补证，不改写原结果。

最终包的 CPU 安装态 spawn/fork、两层缓存、env-cache 和 constructor
drift 拒绝复验通过。离线 mHC 回归22项通过，包含部分导入时的
`BaseException`、快关优先级、异常格式化失败及格式化期间真实 fork。
vLLM 0.33 draft-update 会遍历 draft module；CPU 测试夹具须返回空
`torch.nn.Module`，不能用无 `modules()` 的 namespace，亦不能替换官方
LM-head 遍历来掩盖该前置条件。

仍需完整仓内门禁、真实 V4 编译/Graph、其他量化与模型、生产配置性能及
gfx936 的 TileLang 复验。gfx938 W4A8 eager HumanEval164 已完成，见本节开头。
catalog 为 candidate，不标 validated/production，不提升 last-good。

宿主回归补证：vLLM 专项335项通过。混合节点门禁必须用完整 SGLang
依赖和 `HCU_ADAPTER_HOST=sglang`，不能在 vLLM 进程安装六参数 RMS 桥。
正确宿主下 rope/RMS/GEMM/bridge Graph 可通过，但完整门禁仍未通过：
SGLang 0.5.22 的 kpool 测试夹具缺 `index_page_size`，FP8 计算链在真实
gfx936 被官方 `_scaled_mm` 能力检查拒绝；两者在未加 mHC 的基线上复现。
INT8 17-token 计算链曾等待 GPU 超时，保留原始失败，不按跳过或通过计数。
这些 SGLang 路径不调用本 vLLM mHC 桥，仍须独立修复及重跑完整门禁。

实际 V4 Flash TP4 原始权重已完成加载，显式官方 MXFP4 emulation 与 CPU
offload 后在首次 profile 被官方 sparse-attention indexer 的 gfx936 能力
检查拒绝，尚未 ready，也没有模型精度结果。该检查不能通过伪装 gfx950
或 CDNA 代际绕过；继续按官方 provider 接口寻找窄适配。官方 AMD V4 类
没有声明 torch.compile 支持，CLI 编译参数也不能单独证明执行了编译。
上述集成限制不改变本桥的 opt-in/candidate 状态或原始质量门槛。

官方 gfx936 路径修复并过同门禁后，在同 MR 删除适配并恢复官方 dispatch；
仅升级版本号或 import 成功不是退出证据。

策略重试补证（2026-11-04）：注册后的非法策略直接重试先锁存失败，恢复
配置仍拒绝；注册前的纯配置错误可修正后首次激活。新增两项离线回归，
mHC 合计24项通过；受影响的 mHC/启动策略/native ABI/共享生命周期/
bootstrap 合计101项通过。独立 mHC 候选包的全部206个 Adapter 与2991个 Core
Python 文件均匹配源码和 wheel；安装态独立 spawn 验证直接非法策略及
恢复后拒绝。真实 gfx936 再跑 pre/fused 共64 cases/256 states，保留
原始容差、64次初始 replay、144次非空变输入及48次空 Graph replay。
完整节点、layer/e2e、模型和性能门禁仍保持上述限制。
