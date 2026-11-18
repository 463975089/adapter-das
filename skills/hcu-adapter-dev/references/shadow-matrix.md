# 影子库矩阵（shadow-library matrix）

## 边界定义

以**社区代码的 import 面**为边界：sglang / vllm 源码里 `import` 到的算子
库名，本仓库都要让 `import` 成功。不锚定任何库的内部实现（内部实现随上游
版本漂移，import 面稳定得多）。

DeepInfer 契约实现按算子族和档位放入 `hcu_adapter/kernels/`，文件位置及入口职责见[算子文件导航](../../../docs/frameworks/deepinfer-kernels.md)。
拆分实现时同步 bridge/catalog 与 plugin 的导入和能力检查；删除迁移后的旧算子转发路径。

当前覆盖的 import 面（发现新的就加进矩阵）：

| 社区 import 的包 | 我们的形态 | 位置 | 状态 |
|---|---|---|---|
| `sgl_kernel` | 完整同名包（triton/torch） | `sgl_kernel/` | 生产 |
| `vllm._custom_ops` | SGLang 启动链使用的 RMS 属性级兼容绑定；不是 vLLM 宿主的默认重绑定 | `hcu_adapter/bridges/vllm.py::install_rms_norm_bridge` | SGLang 兼容路径 |
| `aiter` | 真包优先 + 缺口填空 + 高危守卫 | `hcu_adapter/aiter/` | 生产 |
| `flashinfer` | 填空桥 + 响亮 stub | `hcu_adapter/flashinfer/` | 生产 |
| `flash_attn` / `flash_attn_interface` | 响亮 stub；装 HCU flash_attn wheel 后真包自动胜出（零改动切 vendor 档） | `hcu_adapter/flash_attn/` | 生产 |
| `lightop` | vendor 直调档（不出现在 import 面，按 ops 路由调用；见下盘点） | 系统包直调 | silu/MQA 已入 catalog |

**RMS 宿主 ABI 分开核对**：上述 helper 的生产调用点在
`hcu_adapter/bridges/sglang.py`；vLLM runtime plugin 不调用它。helper 的
`fused_add_rms_norm(out, input, residual_out, residual, weight, epsilon)` 是
六参数、独立输出目标的兼容签名；vLLM 0.32 官方
`fused_add_rms_norm(input, residual, weight, epsilon)` 是四参数原位接口，
修改 input/residual 并返回 `None`。不能因同名把前者装进 vLLM 启动链。
该属性绑定也不是 `vllm._custom_ops` / `_C` 的完整影子实现；缺失的原生符号
须逐项验证。vendor 的响亮 stub 可让 import / `hasattr` / `callable` 成功，
这些检查不能证明所需 provider 的 ABI 或计算可用。

FlashInfer shadow 的 norm 转发在调用时导入既有 `sgl_kernel.layernorm`
实现；不能在模块顶层先导入它，否则 shadow-first 顺序会经
`sgl_kernel` 的 stub 注册重入尚未定义 `register()` 的模块。
注册仍在启动期完成，不延迟换绑。新解释器回归覆盖 shadow-first、
kernel-first、公开入口及原 `out=`/原位 ABI；这是 import/转发证据，
不能替代 GPU 数值、Graph 或模型门禁。

**flash_attn wheel 兼容性现状**（2026-10-10 实测）：共享盘
`/public/share/dtk_fa_whl/*/flash_attn-2.6.1+das.opt1.dtk2604-cp310`
只匹配 torch2.5 栈；dtk26041/torch2.11 上 `c10::hip` 符号不匹配，需
vendor 出新轮。轮子就位后 `tests/vendor/test_flash_attn.py` 的探针会自动
切到真包跑全场景矩阵（sglang-das 侵入仓同款回退链
`kernels → sgl_kernel.flash_attn → flash_attn 包`）。

SGLang 的随机 EAGLE 验收在 CUDA 语义分支导入
`flashinfer.sampling.top_k_renorm_probs/top_p_renorm_probs`。宿主桥将这两个
名字绑定到固定官方版本已有的 HIP Triton renorm，保留阈值处全部同值概率；
不复用旧 torch 采样面“恰好 k 个”的不同 tie 语义。导入延迟到调用时，
不把 SGLang 依赖加到其他宿主。此绑定不表示 FlashInfer 全采样 API 已覆盖。

## 三档实现策略

SGLang 的 `deep_gemm.tf32_hc_prenorm_gemm` 缺失出口由独立的
`hcu_adapter/kernels/sglang_mhc_prenorm_emu.py` 承接，保持上游 split-K
部分和及平方和布局。等长 split 批量计算，不在 Graph 内展开逐 split
的小算子链；不整除时分为两组，空 split 写零。数值与变输入重放证据见
[模型记录](../../../docs/models.md)。此实现不转接 DeepInfer 的 mHC 内核。

SGLang DSA MQA 的 paged/ragged 两种布局均须先对每个 head 的点积做
ReLU，再乘 head 权重并求和，最后应用 K scale。不能只拿两个省略
ReLU 的实现互相对拍；应包含负点积反例和官方 torch 有效位置对照。

SGLang 的单卡子进程分配通过其 `CudaDeviceMixin.reindex_device_id`
同步 CUDA/HIP 可见掩码；父进程上下文退出必须恢复原环境。仅缩窄
CUDA_VISIBLE_DEVICES 而保留更宽的 HIP_VISIBLE_DEVICES，会令多个
rank 在同一张物理卡上初始化；应验证非连续顺序和真实通信。

每个算子出口按优先级路由，出口签名逐字对齐社区版（含 `out=`、
destination-passing、占位参数）：

1. **triton（生产主力）**：自研移植，fp32 累加、graph-safe、无主机同步。
2. **vendor（转发档）**：DTK wheel 里真包已有且经过我们的压测+守卫的
   （如 aiter 部分 kernel、lightop）。接入条件 = 过场景压测 + 注册快关。
   arch 探测失败自动降级到 triton/torch 并打日志。
3. **torch（参考档）**：语义金标 + 功能兜底（慢但永远正确），也是
   契约测试的参考实现来源。

**档位状态机与演进（catalog 管制）**：每算子
出口的可用实现在 BACKEND_CATALOG 挂状态——
`planned → candidate（离线三段式绿+对拍+bench）→ validated（serving A/B
过）→ production（默认档，连续稳定后）`，定罪不可用标 `broken`（选配即
拒，bug 签名与复现探针在册；哨兵意外转绿=轮子修好，重跑门禁即晋级）。
演进**单向**：**emu（torch 调试图稳/金标准）→ triton → vendor**；
emu/triton 换下不删、转 ref 角色只留离线测试对照。校准方向不变：emu 校准
triton，emu/triton 暴露 vendor bug。晋级证据门与登记要求见
`docs/governance.md`。

路由已落地（env + yaml 双层）：`HCU_SILU_MUL_IMPL` / `HCU_RMS_NORM_IMPL` /
`HCU_FUSED_MOE_IMPL` 等已在册（hcu_env REGISTRY），正式机制是
`hcu_adapter.yaml` 的 ops 键（见 docs/config.md 与 BACKEND_CATALOG），
`auto` = 探测+压测结论决定；新算子接入时同步登记 catalog。

## lightop 接入盘点（DTK 0.6.0，validation-host-26 dtk26041 实测 export 面与 aiter 交集）

gfx938 有 fp8（gfx936 只有 int8）。**与 aiter 同用法的交集算子**（同一语义
两边都有实现 → 同一份契约测试跑双 backend，正是 73-297 要的"默认打开、
排精度时可切"）：

| 语义 | lightop 出口 | aiter 出口 | 接入优先级 |
|---|---|---|---|
| silu_and_mul(+fp8 quant 融合) | `activation.existing.fuse_silu_and_mul` / `fuse_silu_mul_fp8_quant` 系列（含 _ep） | `silu_and_mul` / fused_clamp_act_mul 融合 | P1（契约最简单，跑通双 backend 模式） |
| MQA logits（DSA） | `attention.mqa.paged_mqa_logits` / `mqa_logits` / `_fp4` 变体 | `aiter.ops.triton(.attention).pa_mqa_logits` | P1（我们有 mqa_logits_emu torch 金标可直接对拍） |
| MoE group gemm | `moe.groupgemm_permute / fuse_groupgemm_permute_quant / gemm_ops.quantized.gemm_w8a8_*`（marlin w4a8/w4a16 配置齐） | aiter 项目前 moe 出口少 | P2 |
| RMSnorm+RoPE 融合 | `fuse_rmsnorm_rope.rms_rotary_embedding_fuse(_with_kv_store)` | 无对应（aiter 分开） | P2（性能档） |
| KV cache 反量化 | `kvcache.dequantize_k_cache_paged` / `gather_and_maybe_dequant_cache` | `cp_gather_indexer_k_quant_cache`（我们已桥） | P3 |

接入顺序：P1 两个先做（silu 融合→mqa），每个出口注册 `HCU_<OP>_IMPL`
env 档位（auto=vendor 过压测则用/triton/torch），四个维度测试各跑一遍。
 أصل轻盘点记录：`lightop._lmslim_native.quantize` 顶层无导出（空）、
marlin repack 只在 lightop 有（aiter 无交）。

## 新增一个算子垫片的流程

SGLang NextN 的 fused_eh_norm 保留上游 JIT 计算：`bridges/_sglang/jit_eh_norm.py` 只为该模块补 HIP 最大线程数编译参数，避免 4096 宽度的 512-thread launch 超过 DTK 默认 256-thread 编译界限。归 rms_norm 的运行时绑定记录；不与 DeepInfer/vLLM 复用。验收需独立 FP32 RMSNorm+cat 对照、变输入 Graph 重放及实际 MTP，单纯编译通过不能算数值通过。

1. **定契约**：找到社区调用点，抄下完整签名/布局/返回语义（读上游源码，
   不要猜；rope 教训：同一个函数名在 cache 布局上还有隐含约定）。
2. **写参考**：`torch` 档参考实现（fp32），这就是测试金标。
3. **写生产档**：triton 或 vendor 转发；遵守黄金法则（fp32 累加、无
   主机同步、import 不失败）。
4. **写契约测试**：按 `test-suite.md` 四维度；vendor 高危面同时写场景
   压测 + 注册快关（`kill-switches.md`）。
5. **更新矩阵**：本文件表格 + README 覆盖范围表。

SGLang sparse FlashMLA 的 DTK 候选支持 2048–2112 列；用 -1 补齐到 64 的倍数，并将少于 64 的 query heads 补零后裁回。输入索引不原地排序。KPool 历史索引后的尾部 token 由 plugin 保留，不用历史有效长度裁掉尾列；整模型验收独立记录。

MLA HiCache 的 `transfer_kv_all_layer_mla` 接收 GPU int64/uint64 原始地址表，per-layer 接收连续 device 或 pinned host 张量。`*_lf_pf`/`*_pf_lf` 的 layout_dim 和 item_size 单位都是字节；禁止按普通张量逐层索引地址表。candidate Triton 实现覆盖这些四个 MLA API，其他 direct/非 MLA 接口不在本轮验证结论中。定向字节/Graph 诊断通过不代表整模型 HiCache 已验收。

Ragged MQA 长 prefill 采用 BF16 矩阵分块：E4M3 的有限值可被 BF16 精确表示，逐 head 点积后仍先 ReLU、乘权重再求和；不能把 ReLU 移到 head 归约之后。每行窗口外保持负无穷，完全无效的 tile 只写 mask。N<32 继续既有向量路径；数值、变输入 Graph、热态算子计时与完整模型 profile 分开记录。

`sgl_kernel.int8_scaled_mm` 可用 `ops.int8_scaled_mm=lightop` 选择候选原生 prefill 路径（BF16 输出、M≥32、K 为128倍数、N为64倍数、列主序权重且无 bias）；其他布局、短 batch、bias 和 vendor 不支持的形状继续 Triton。默认保持 Triton，显式选择和实际 provider 记入 runtime registry；`HCU_DISABLE_OPS=int8_scaled_mm` 可拒绝该接口，改回配置 `triton` 可恢复旧路由。数值、变输入 Graph 与整模型结果各自验收。

`ops.sglang_grouped_topk=triton` 显式复用官方融合路由，仅接管 FP32、288 experts、top8、单组、无融合共享专家、M≥32 的连续输入。保留官方 renormalize epsilon、仅归一化后施加 scaling 的语义及原后处理；短 batch、动态 bias 和其他形状仍调用已有 provider。旧模块别名同步绑定，不修改平台判定或官方源码。当前 GPU 数值与变输入 Graph 对拍通过，整模型验证待完成。

paged MQA 的 Triton/vendor 分发必须先于 emu 缓存展开；选中原生 provider 后不能先创建整池 BF16/interleave 副本。保留原来的输入规范化和参数传递，优化只消除未使用的准备结果，数值/Graph 逐位对照及模型性能分别验证。

SGLang target-only 随机树采样的 JIT 路径要求未安装的 FlashInfer C++ 头文件。adapter 候选以 Triton 保留树遍历、拒绝草稿概率写回、有效接受索引及最终残差分布采样；不修改随机硬币、接受阈值或转为贪心。阈值同原生 AOT 入口将累计阈值下限设为1e-9；输入和树验收概率为FP32，最终残差CDF采用FP64累加，避免大词表FP32分块求和在硬币边界选中相邻token；不承诺与原生FP32累加逐位一致。完整随机模型评测仍待验证。
