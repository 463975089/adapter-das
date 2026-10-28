# 端到端性能优化手册（E2E Performance）

适用场景：把 HCU 上 sglang 的端到端性能做起来——跑通→开箱基线→并行调参→profiler 归因→算子优化需求→回归。
交付物（同团队 SOP）：开箱性能、根因分析数据、优化项提升过程数据及最优启动脚本（cookbook）。

> 本手册是流程骨架与仓内资产映射；**每个环节的知识细节存个人知识库（kb）**，
> 引用格式 `kb:<id>`。检索入口：
> `curl http://127.0.0.1:8765/api/health`（不通则 `python E:\GITS\KnowledgeSet\kb_server.py`），
> `curl "http://127.0.0.1:8765/api/search?q=<词>"`；人看图谱逛 `http://127.0.0.1:8765/`。

## 三个指标先定靶（每次优化只改一个目标变量）

| 指标 | 含义 | 观察阶段 |
|---|---|---|
| TTFT | 首 token 延迟 | prefill、长输入、PD 分离 prefill 侧 |
| TPOT/ITL | decode 每 token 耗时 | decode 侧、单请求体验 |
| Throughput | tokens/s 或 req/s | batch/concurrency 扩展、服务容量 |

不得同时改多个变量再一起看结果——归因会废。kb:e9352b

## 九步流程（每步的仓内钩子）

| # | 步骤 | 仓内/工具钩子 | 细节 |
|---|---|---|---|
| 1 | 拿量化模型与启动命令（六项信息） | 权重格式对照 `docs/env-vars.md`；精度验证方式对照 `references/precision-triage.md` | kb:de8fc0 |
| 2 | 最简并行跑通（单机 TP 起步，勿先叠 CP/EP/PP/PD） | 容器先按黄金法则 8 灭镜像包 | kb:de8fc0 |
| 3 | 出开箱性能（PD 分离模拟要求按团队表格） | 基线记录进 `docs/models.md` 对应模型行（含环境戳） | kb:4182a4, kb:f9261a |
| 4 | 补各并行方式数据（TP/EP/CP 各影响什么） | 快关 A/B 用 `HCU_DISABLE_OPS`（kill_switch 注册表） | kb:a860f2 |
| 5 | 抓 profiler（避开 warmup，prefill/decode 分开抓） | 下方 curl；trace 与启动/bench 参数同存 | kb:dd1d00 |
| 6 | 分析：算子八字段表→**先对齐端到端**→竞品同口径 | 与 `tests/bench`（gsm8k_eval）结果互证 | kb:ff528d |
| 7 | 大头算子归因（Attention/MoE/GEMM/小算子族） | GEMM 必带 size；融合提议走 kb:a4fab3 原则 | kb:5a71fa, kb:145a0a |
| 8 | 回归端到端 + profiler + 精度 | 精度回归走 `run_triage.py`；**启动验证≠精度验证** kb:ad1e07 | — |
| 9 | 写结论与风险；优化落地走治理闭环 | catalog 晋级需对拍+bench+A/B 证据，见 SKILL.md 治理闭环 | — |

## profiler 快速抓取

```bash
port=${1:-30000}
curl -X POST http://127.0.0.1:$port/start_profile \
  -H "Content-Type: application/json" \
  -d '{"output_dir":"/prof/","start_step":5,"num_steps":5,"activities":["CPU","GPU"]}'
```

- benchmark 先 warmup 再抓；trace 里只出现一次的 size/kernel 先怀疑 warmup/编译路径。
- trace 路径、rank、step、启动参数、benchmark 参数同存，缺一个就不可复现。

## 与本仓黄金法则的接口（违反即翻车）

- **bench 三分离**（法则 11）：GPU 空闲检查 → 强制预热段（冷编译曾测出假 0.0×）→ 结果带环境戳；断言用相对阈值，不吃绝对 tok/s。
- 性能"变快"不是合入依据：**catalog 档位证据门**说了算；无 A/B+对拍数据的提速一律按传闻处理。
- 新 kernel 反馈到算子供侧时，走治理闭环登记链（catalog/env 手册/UI 快照/registry 同 commit）。
- 精度回归用 `tests/tools/run_triage.py`；性能问题疑似 kernel 级再上 SQTT（kb:3790f5、kb:ae148f）。

## 硬件预期速查（竞品对比前先看口径）

| 项 | 值 | 备注 |
|---|---|---|
| H20 显存带宽 | 4.0 TB/s（公开口径） | 算力口径（FP16/BF16/FP8）需与竞品约定固定 |
| BW1100 带宽 | 1.8 TB/s | 勿用其他 DCU 型号替代 |
| 大 size GEMM 目标 | ≈ H20 的 1.2~1.5 倍 | 内部经验值 |
| 访存算子预期 | X/4.0 × H20 上限（X=BW1100 实际带宽） | 再按实际访存模式修正 |
| GEMM 实测效率 | 2mnk/时间（GFX906 峰值公式） | kb:676d1c |

细节与失效判定：kb:9e3335。融合决策与实证（+105.5% 勿外推）：kb:a4fab3、kb:39340d。

## 提优化需求的最小数据包

给算子同事的需求必须附：trace+场景+kernel 名+占比+竞品同口径数据；
GEMM 类另附 M/N/K、dtype（BF16/FP8 cw/bw、INT8、FP4）、layout、tuning 目标；
MoE 另附 batch/topk/expert 数/EP 配置、grouped GEMM size、dispatch/combine 耗时。缺项的需求按可被打回处理。

## 原文与本地快照

- 飞书原文（活文档，冲突以原文为准）：`kb:408d32`（含 PD 分离测试表格入口）
- 本地快照：`E:\GITS\KnowledgeSet\.ingest\_feishu_performance_guide.txt`（2026-08-08 版）
