# 跑通模型记录表（HCU adapter × 官方 sglang 零修改）

每行 = 一次完整 e2e 验证。判定列的四个位：`启动/生成/精度/长文`。
新跑一个模型先加行（待验证），测完更新。环境三原则：零显式 env。

| 日期 | 模型 | 规模/卡 | 节点/栈 | 参数要点 | 启动 | 生成 | 精度 | 长文 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-08 | Qwen3-0.6B (bf16) | 1.2G/1卡 | nmz26 dtk2604 py2.5-era | TP1, triton attn | ✅ | ✅ | ✅ 对话正确 | ✅ | 首个 dense e2e；需 --attention-backend triton |
| 2026-10-08 | Qwen3.6-35B-A3B (GDN+MoE bf16) | 67G/1卡 | nmz26 | TP1 144G, mem 0.80 | ✅ | ✅ | ✅ GSM8K-100 69% | ✅ | MoE+线性注意力；graph capture 经 moe_align triton 化修复 (527ad97) |
| 2026-10-10 | GLM-5-W4A8_V2_6 (slimquant int4-int8 MoE) | ~G/8卡 | nmz18 dtk26041 torch2.11 | TP8 graph ON 零旗标 | ✅ | ✅ | ✅ 语义/长文锚点 | ✅ | 两 bug 破案（fbc540c rope / 1eb038d aiter AR）；NSA emu 短路验证 |
| 2026-10-10 | Qwen3-4B (bf16) | 7.6G/1卡 | nmz26 dtk26041 torch2.11 + **官方 v0.5.21 release** | TP1, triton attn, graph ON | ✅ | ✅ | ✅ GSM8K-100 68%（16并发46s，无算子错例） | ✅ | 首个官方 release 版；需 `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`（rust TreeCore 无预编译扩展，见下） |
| 2026-10-10 | Qwen3-4B-Channel-FP8 | 7.6G/1卡 | nmz26 dtk26041 + 官方 v0.5.21 | TP1, graph ON/OFF 双试 | ✅ | ❌ | ❌ token 全 0 | — | **quant=compressed-tensors**（format=float-quantized, act=strategy=token）——adapter 未覆盖的路径（README 路线图已知项）；fp8 trace 证实 apply_fp8_linear 从未被调，推理走 compressed-tensors 自己的 kernel 链。待办：compressed-tensors 垫片或换 Fp8LinearMethod 格式的 checkpoint 对照 |
| 2026-10-10 | Qwen2.5-0.5B-Instruct | 954M/1卡 | nmz26 dtk26041 | TP1, triton attn, graph ON | ✅ | ✅ | ✅ SHORT 语义锚点 | ✅ | qwen2 家族（attn/rope 走 Qwen2 分支）三门全过 |
| 2026-10-10 | Qwen3-32B (bf16) | 62G/1卡 | nmz26 dtk26041 | TP1 144G, mem 0.85 | 待验证 | | | | 中型 dense 64 层 |
| 2026-10-10 | Qwen3-30B-A3B (MoE bf16) | 57G/1卡 | nmz26 dtk26041 官方 v0.5.21 | TP1, mem 0.85, triton attn, graph ON | ✅ | ✅ | ✅ SHORT 数学锚点+CHAT thinking 连贯 | ✅ LONG after-think 连贯 | MoE 128 专家独立验证；thinking 模型注意 max_new_tokens 给足 |
| 2026-10-10 | Qwen3.8-27B-FP8 | 29G/1卡 | nmz26 dtk26041 官方 v0.5.21 | TP1, mem 0.85, triton attn, graph ON | ✅ | ✅ | ✅ CHAT=Paris | ✅ LONG after-think=Paris | **fp8 全链首验通过**：fp8_scaled_mm triton 后端 + capability HIP 桥（v0.5.21 Fp8ScaledMMOp AOT capacity=CUDA 拒 HIP）；长文注意 thinking 包裹（看 `</think>` 后文本） |
| 2026-10-10 | Qwen3.6-27B-Channel-INT8-w8a8 | 55G/1卡 | nmz26 dtk26041 官方 v0.5.21 | TP1, triton attn, graph ON | ✅ | ✅ | ✅ SHORT 数学锚点 1397km；GSM8K-mini 11/12 | ✅ LONG 识别重复欺骗+总结 | **CT int-quantized 首验通过**：int8_scaled_mm triton 后端 + capability/int8 双桥；曾乱码——根因是 HCU_SLIMQUANT 的 safetensors shim 全局误伤（dequant→bf16 copy_ 进 int8 参数全零），已改为按 checkpoint format 门控（仅 slimquant_w4a8 格式生效） |
| 2026-10-11 | DeepSeek-R1-L4（R1 截 4 层 fp8 MoE-MLA） | 17G/1卡 | nmz26 dtk26041 官方 v0.5.21 | TP1, **decode=flashmla** + prefill=triton 混合, page 64, graph OFF | ✅ | ✅ | ✅ 内核 6/6 单测（fp32 参考 diff≤4e-3）；e2e 对拍 triton 2/6 条 48 token 逐字全同，其余分叉点全部落在 bf16 平局（top2gap≤0.0625 且两侧候选对相同）；triton 自噪声 4e-4 为参照 | ✅ | **flashmla attention backend 首验通过**：emu 内核 + 三个新桥——wrapper 构造桩（FlashInferMLA 基类）、块表 -1 padding clamp、`--prefill-attention-backend triton` 混合（上游 FlashMLA prefill 本就依赖 flashinfer）。减层物料：163 shard 抽前 4 层+embed/lm_head 重打包单文件（config num_hidden_layers=4） |
| 2026-10-11 | Qwen3-0.6B + **fp8_e4m3 KV cache**（triton attn） | 1.2G/1卡 | nmz26 dtk26041 官方 v0.5.21 | TP1, `--kv-cache-dtype fp8_e4m3`, graph ON | ✅ | ✅ | ✅ 6/6 语义锚点流畅（数学/常识/英文） | ✅ | **fp8 KV cache 首验通过——零 adapter 改动**：上游 triton 内核自带 fp8 dot + k_scale 折叠（gfx1250 HIP 分支注释在），pool 侧 uint8 存储/fp8 view 读全兼容；HCU 判 fnuz=False 用标准 e4m3fn |
| 2026-10-11 | Qwen3-0.6B + **fa3 backend**（bf16 KV） | 1.2G/1卡 | nmz26 dtk26041 官方 v0.5.21 | TP1, `--attention-backend fa3`, graph ON | ✅ | ✅ | ✅ 6/6 语义锚点；内核 9/9 单测（fp32 参考 diff≤8e-3，含 GQA/fp8 KV/window/softcap/varlen/merge） | ✅ | **fa3 attention backend 首验通过**：`sgl_kernel.flash_attn` emu（分块在线 softmax 防 capture OOM、全静态形状图安全、页粒 gather）；sitecustomize 桥设 `SGLANG_USE_SGL_FA3_KERNEL=1`+放行 `_is_fa3_supported`（HCU_FA3_EMU=1 默认）；`merge_state_v2`/`get_scheduler_metadata` 桩 |
| 2026-10-11 | Qwen3-0.6B + fa3 + **fp8 KV + TP2** | 1.2G/2卡 | nmz26 dtk26041 官方 v0.5.21 | TP2, fa3, fp8_e4m3 KV, **eager**（graph OFF） | ✅ | ✅ | ✅ 6/6 语义锚点（fa3+量化+两卡 NCCL 三合一） | ✅ | 多卡组合验证；**nmz26 平台发现：TP>1 + CUDA graph 输出乱码（NCCL 与 aiter AR 均坏，triton/fa3 均坏，TP1 graph 正常、TP2 eager 正常）→ nmz26 多卡暂需 `--disable-cuda-graph`**（nmz18 TP8 graph 曾通过，两机栈差异待深挖） |
| 2026-10-11 | **Qwen3.5-122B-A10B-AWQ**（awq gemm 4bit/g128/zp，MoE 专家量化；GDN 线性注意力+MoE+MTP，fp16 底模） | 77G/4卡 | nmz26 dtk26041 官方 v0.5.21 | TP4, `--quantization moe_wna16`, triton attn, **eager**（多卡 graph 禁）, `SGLANG_MAMBA_CONV_DTYPE=float16`, mem 0.85 | ✅ | ✅ | ✅ 6/6 锚点连贯；3 题终答精确（14000米/北京/40） | ✅ 1848 tok prompt 三问全对（28段/T28/M6 含模运算检索） | **AWQ 路径首验通过**。关键链路：①`--quantization moe_wna16` 是 HCU 正确入口——上游把 FusedMoE 路由到 triton W4A16 runner、LinearBase 委托回 plain AWQ（HIP 分支自带 `awq_dequantize_triton`）；**默认无旗标会被 capability=(9,3)≥80 骗去自动转 awq_marlin，而 HIP 分支 `awq_marlin_repack=None` 必炸**。②adapter 三桥：MoeWNA16 `get_quant_method` RadixAttention isinstance 守卫（上游把 modules_to_not_convert 检查放在 isinstance 前，`"self_attn"` 子串误中 `...self_attn.attn` → 6 参 create_weights 被单参调用）；selector 消歧（我们的 capability HIP 桥放行了 JIT/_CUDA spec 造成二义性，现多后端时优先原生可用的 spec——本例 AOT/_HIP，与真 ROCm 一致）；fp16 底模需 `SGLANG_MAMBA_CONV_DTYPE=float16`（上游 conv 池硬编码 bf16 不看模型 dtype）。留证 30078/GPU0-3 |
| 2026-10-11 | **Qwen3.5-4B + MTP 投机解码**（EAGLE/NEXTN，32 层 dense GDN 混合；checkpoint 自带 `mtp.*` 单层全注意力 draft） | 8.8G/1卡 | nmz26 dtk26041 官方 v0.5.21 | TP1, `--speculative-algorithm EAGLE --speculative-draft-model-path <同路径>`, steps3/topk1/draft4, **fa3 attn**, eager, mem 0.80 | ✅ | ✅ | ✅ **8/8 语义锚点**（Paris/北京/算术/代码补全/Jupiter 含 `<think>`）；**accept len 2.35–3.85/4（均值 ~3.0，accept rate 至 0.81）** | ✅ | **MTP 精度验证通过**。要点：①draft 无需重打包——sglang 对 is_draft 的 Qwen3.5 自动改写 `Qwen3_5ForCausalLMMTP`（单层 full_attention，正配 fa3）；GDN 线性注意力层走 `linear_attn_backend=triton`（上游唯一实现，与全注意力 backend 无关）。②吞吐注意：eager 模式 MTP 1.8 tok/s vs 无 MTP 基线 15.1 tok/s（同卡同配置）——verify 批(4 token)在 python-emulation 栈上每步开销 ~25×，接受率本身正常；graph ON 版见下。③修复 `_adapters/_base.py` namespace 包 loader=None 崩（MTP graph capture 懒加载 aiter_backend 触发：`_OverlayLoader.exec_module` 缺 None 守卫）。R1-L4 减层模型词汤已破案为截断固有（昨日 flashmla PASS 判据是后端一致性非语义），MTP 精度标尺须用全量模型 |
| 2026-10-11 | Qwen3.5-4B + MTP + **cuda graph ON** | 8.8G/1卡 | nmz26 dtk26041 官方 v0.5.21 | 同上去掉 `--disable-cuda-graph` | ✅ | ✅ | ✅ 8/8 锚点与 eager/基线**逐字一致**（graph 正确性交叉验证）；accept len 同分布（~3.0/4） | ✅ | MTP+graph 组合验证：target 23 档 + draft 23 档 capture 均成功（链式 spec 形状静态：3×1 token draft、4 token verify）；修 `_base.py` 后 warmup 不再崩。吞吐 **4.6–7.6 tok/s（essay 256tok 55.4s）= eager MTP 的 2.6×，但仍低于无 MTP 基线 15.1**——emu 栈每步 4-token verify 的 python 页 gather × 33 层开销主导（每步 3 draft + 1 verify forward 换 ~3 token），真 FA3 硬件内核下预期翻正。留证 30084/GPU6 |
| 规划 | Qwen3.5-35B-A3B-W8A8 | 36G/1卡 | nmz26 | TP1 | 风险：config 无 quantization_config（裸 w8a8 打包，需 --quantization 显式或权重名自识别） | | | | 36G/14 分片无 quant 声明，格式待定 |

## 组合特性清单（每跑通一个打勾）

- [x] MTP（投机解码）（Qwen3.5-4B EAGLE：8/8 锚点 + accept len ~3.0/4；R1-L4 NEXTN 链路 rig）
- [ ] hicache（分层 KV cache）
- [x] 并发多请求（GSM8K-100 c=16 通过，图内批处理正常）
- [ ] EP 高吞吐（expert parallel）
- [ ] EP 低延迟
- [ ] PD 分离
- [ ] dspark
- [ ] spec vllm-NV 路径降级验证

## 官方 v0.5.21 在 HCU 上的已知调整（零修改前提下的 env 级适配）

| 现象 | 原因 | 解法 | 去向 |
|---|---|---|---|
| rust TreeCore 源码构建失败（Cargo） | 镜像无预编译 rust 扩展、cargo 半残；v0.5.21 默认后端=rust | `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python` | 官方自带 python 后端回退，低频写路径够用；确认后可进 hcu_env 自动规则 |
| `--attention-backend flashmla` 构造即 NameError（BatchPrefillWithRaggedKVCacheWrapper） | FlashMLABackend 继承 FlashInferMLA 基类，`__init__` 无条件构造 flashinfer wrapper，而 flashinfer 在 HCU 不可得（import 被 is_flashinfer_available() 门控，名字从未绑定） | sitecustomize 构造桩桥（可构造、任何 plan/run 调用大声报错） | 已内置 adapter；wrapper 实际不被 flashmla 前向使用 |
| flashmla 的 EXTEND prefill 依赖 flashinfer | 上游设计：FlashMLA 内核只管 decode/verify，prefill 走 flashinfer paged prefill | `--prefill-attention-backend triton --decode-attention-backend flashmla` 混合 | v0.5.21 原生支持混合后端（实验特性警告可忽略） |
| **nmz26：TP>1 + CUDA graph 输出乱码**（!!!/词汤） | 图内集合通信损坏：NCCL 与 aiter AR 均复现、triton/fa3 后端均复现——与 attention 后端和 AR 实现都无关；TP1 graph 正常、TP2 eager 正常；nmz18 TP8 graph 曾通过（两机驱动/NCCL 栈差异待查） | 多卡加 `--disable-cuda-graph` | 平台级问题待深挖（NCCL graph capture / hipGraph 支持）；单卡不受影响 |
| `--attention-backend fa3` 默认不可用 | `_is_fa3_supported()` 在 HIP 上恒 False（torch.version.cuda=None）→ with_kvcache raise、varlen 落坏 wheel | adapter 已内置：`sgl_kernel.flash_attn` emu + sitecustomize 桥（HCU_FA3_EMU=1 默认开） | 已内置 adapter；emu 为正确性路径，生产性能走 triton |
| AWQ checkpoint 默认必炸（两层） | ①capability=(9,3)≥80 使 awq 自动转 awq_marlin，但 HIP 分支 `awq_marlin_repack=None`；②moe_wna16 的 get_quant_method 缺 isinstance 守卫，RadixAttention 被 `"self_attn"` 子串误中拿 6 参 method | `--quantization moe_wna16` + adapter sitecustomize 守卫桥（已内置） | 上游出口：MoE→triton W4A16、Linear→`awq_dequantize_triton`（HIP 分支自带） |
| fp16 底模 + GDN 线性注意力首 forward dtype 崩 | conv-state 池 dtype 只由 `SGLANG_MAMBA_CONV_DTYPE` 决定（默认硬编码 bf16），不看模型 dtype；fp16 模型激活是 Half → index put 失配 | 启动 env `SGLANG_MAMBA_CONV_DTYPE=float16` | 上游 env 出口，零代码 |

## 验证口径

启动 = server 起来 /health OX；生成 = 非 thinking 截断下 16 token 连贯；
精度 = 语义锚点（London->Paris 类）+ 与 eager/graph 双变体一致；
长文 = ~2700 token prompt（>DSA index_topk 覆盖 sparse 变体）。
