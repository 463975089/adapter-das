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
| 2026-10-10 | Qwen2.5-0.5B-Instruct | 954M/1卡 | nmz26 dtk26041 | TP1 | 待验证 | | | | 极小回归锚点；qwen2 家族（attn/rope 走 Qwen2 分支） |
| 2026-10-10 | Qwen3-32B (bf16) | 62G/1卡 | nmz26 dtk26041 | TP1 144G, mem 0.85 | 待验证 | | | | 中型 dense 64 层 |
| 2026-10-10 | Qwen3-30B-A3B (MoE bf16) | 57G/1卡 | nmz26 dtk26041 | TP1, mem 0.85 | 待验证 | | | | MoE 128 专家；独立于 GDN 验证 align/gate 链 |
| 2026-10-10 | Qwen3.8-27B-FP8 | 29G/1卡 | nmz26 dtk26041 | TP1, mem 0.85, trust-remote-code | 待验证 | | | | **quant_method=fp8 正确格式**（dynamic block[128,128]，Qwen3_5 arch）——fp8 全链首个真验证（Channel-FP8 是 CT 路径不算） |
| 2026-10-10 | Qwen3.5-122B-A10B-AWQ | 77G/4卡 | nmz26 dtk26041 | TP4, trust-remote-code | 待验证 | | | | awq bits=4（Qwen3_5Moe arch）；AWQ 路径首验 |
| 规划 | Qwen3.5-35B-A3B-W8A8 | 36G/1卡 | nmz26 | TP1 | 风险：config 无 quantization_config（裸 w8a8 打包，需 --quantization 显式或权重名自识别） | | | | 36G/14 分片无 quant 声明，格式待定 |

## 组合特性清单（每跑通一个打勾）

- [ ] MTP（投机解码）
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

## 验证口径

启动 = server 起来 /health OX；生成 = 非 thinking 截断下 16 token 连贯；
精度 = 语义锚点（London->Paris 类）+ 与 eager/graph 双变体一致；
长文 = ~2700 token prompt（>DSA index_topk 覆盖 sparse 变体）。
