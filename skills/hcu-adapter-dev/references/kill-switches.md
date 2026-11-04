# 快关（kill switches）与调试设施

高危算子（vendor AR / FA / 未来 vendor 档）必须配"一翻开关就绕开"的快关：
精度出问题时，禁用疑似算子 → 复跑 → 症状消失即锁定。这是最快的二分手段。

**现状定位（2026-10-19）**：AR 层默认 aiter GraphSafeAllreduce（压测
15/15 + serving 连续无恙转 production），**不再需要日常快关兜底**；快关
语义降级为**应急隔离工具**。历史前科：adapter 自研 custom AR 因跨 block
写序竞态定罪（R-01/A 定罪，160 臂 A/B），已除役（修复评估被
"aiter 顶上后无生产位"否决，"修→废"）——教训沉淀：高危 vendor 面转正
靠压测+serving 双证据，出问题先快关二分再定罪。

## HCU_DISABLE_OPS（快关总闸）

```bash
export HCU_DISABLE_OPS=aiter_ar          # 单个
export HCU_DISABLE_OPS=aiter_ar,aiter_fa # 多个（逗号/空格/分号分隔）
```

注册名（`hcu_adapter/kill_switch.py::SWITCHES`）：

| 名字 | 拦截物 | 拦截方式 |
|---|---|---|
| `vllm_sparse_indexer` | vLLM 0.33/gfx936 C4 public OOT | 启动时优先 stock；注册后改变须新进程，不能撤销 dispatcher 或清除失败 |
| `vllm_mhc` | vLLM 0.33 mHC Torch/OOT 适配候选 | 启动时跳过注册，保留原始官方；已注册后改变策略拒绝，须新进程。原始已知 TileLang 失败仍保留，不是功能回退通过。 || `aiter_ar` | AiterCustomAllreduce dispatch | 拒绝 → sglang 回退 NCCL（优雅降级） |
| `aiter_fa` | aiter FA 入口（真包已有属性也拦） | 调用时响亮报错（换 `--attention-backend triton`） |

优先级：`HCU_DISABLE_OPS` > arch 自动规则 > 单算子 opt-in env。
`aiter_ar` 的自动规则（hcu_env.py）区分新旧 aiter 布局：gfx93\* +
**旧版** `aiter.ops.triton.custom_all_reduce` → 自动拒绝（图重放损坏，
1eb038d）；**新版** `aiter.dist` GraphSafeAiterCustomAllreduce → 自动允许
（2026-10-10 nmz26 实测 15/15 压测全过，含 graph-replay 换输入）。
`HCU_AITER_AR=1` 可显式夺回（压测先行），但跨不过显式 disable。
变量手册与自动适应总注册表：`docs/env-vars.md` / `hcu_env.py`。

## 新增一个快关（三处，一个 PR 内）

1. `kill_switch.py::SWITCHES` 注册名字 + 拦截物说明 + opt-in env；
2. 拦截动作：优雅降级型→在 `sitecustomize.py` 对应 patch 处判
   `op_disabled(name)`；调时报错型→在对应 `hcu_adapter/<lib>/__init__.py`
   的 `_guard_attrs()` 加 `{attr: 名字}`（真包属性也会被包一层）；
3. `tests/vendor/test_allreduce.py` 的 gate 测试加一态（禁用生效），
   并在本文件登记。

## 其它运行开关（非快关，行为选择）

| env | 作用 |
|---|---|
| `HCU_SLIMQUANT=1` | 启用 slimquant_w4a8 量化方法桥 |
| `HCU_MQA_EMU=1` | paged/ragged MQA logits 用 torch 模拟（gfx936 无 fp8 时） |
| `HCU_DSA_PRESHUFFLE=0` | 回退 DSA legacy page_size=1 路径 |
| `HCU_DSA_PAGE_SIZE` | DSA page_size 覆盖（须为 index_kpool 倍数） |
| `HCU_AITER_AR=1` | 跨过 gfx93* 自动拒绝，强行启用 aiter AR（压测用） |

## 调试 dump 设施（排精度时抓数据）

| env | 产出 |
|---|---|
| `HCU_DUMP_LAYERS=1`（`_PATH`/`_T`/`_TOTAL`） | 每层 input_layernorm 前 residual 流（TP8 唯一可靠捕获点）→ `run_triage.py --stage layers` 消费 |
| `HCU_DUMP_INNER=1` | L0/L1 的 xa/xb 内部张量 |
| `HCU_DUMP_TOPK=1` | DSA topk 输入输出样本 |
| `HCU_DUMP_MQA=1` | MQA emu 调用与范数日志 |

注意：graph 模式下 python hook 不触发，dump 只在 eager/截层验证时用。
