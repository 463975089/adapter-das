# vLLM 0.33 mHC 的 HCU 行为适配候选

该候选恢复官方已有的 mHC pre/post-pre 语义，归 Adapter；Plugin 不增加重复
owner。Core 运行时代码、原始 `MHCPostOp` 和 `HCHeadOp` 保持官方实现。

显式 opt-in 配置（尚未晋级默认）：

```yaml
version: 1
ops:
  vllm_mhc: torch
```

调试覆盖 `HCU_VLLM_MHC_IMPL=torch|stock`，默认 `stock`。
`HCU_DISABLE_OPS=vllm_mhc` 在启动时优先恢复原始官方路径；它不证明已知失败的
原始 TileLang pre/fused 正确。注册后不能在原进程切换，须启动新进程。
仅准入已测 gfx936、vLLM 0.33 release line；不扩大平台/DTK 通用门禁。

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

仍需完整仓内门禁、四质量维度（cos/maxrel、layer/e2e golden）、
真实 V4 编译/Graph/HumanEval32、性能及其他架构验证。
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
