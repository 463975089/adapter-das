# --autotune：目标机自动探测与选档

> 一条命令在新机器上产出可用的 `hcu_adapter.yaml`：平台画像 → 逐算子三门判定（状态/正确性/性能）→ 阶梯降档带理由。机制随 `tools/hcu_config_wizard.py --autotune` 交付，本文是用户与机制手册。

## 用法

```bash
# 全自动（推荐）：探测 + 逐算子裁判 + A/B 微基准，产两件
python tools/hcu_config_wizard.py --autotune
#   -> hcu_adapter.yaml        仅非默认选档（生成时即过 hcu_config._validate 闭环）
#   -> autotune_report.json    完整判定链（平台画像/每算子各门读数/降档原因/耗时）

# 干跑（无 subprocess 裁判）：只走目录与阶梯，秒级，无卡机器可跑
python tools/hcu_config_wizard.py --autotune --dry -o /tmp/preview.yaml

# 常用变体
python tools/hcu_config_wizard.py --autotune -o /etc/hcu/hcu_adapter.yaml \
    --autotune-out /var/log/autotune_report.json
```

产出 yaml 放进配置搜索路径（`$HCU_CONFIG` → cwd → adapter 根，见 [config.md](config.md)），启动日志 `HCU adapter config:` 行确认生效。

## 选档三门（每算子，阶梯自高到低）

1. **状态门**：`BACKEND_CATALOG` status ∈ {production, validated, candidate} 才进候选；**broken/planned 永不自动选**（broken 选配即 `_validate` 拒，工具在状态门就跳过）。
2. **正确性门**：跑该 op×档位注册的判据测试（`tools/_autotune/choices.py` 映射表）。pytest 单文件或 standalone 直跑（退出码判据）；子进程 env 按档位显式注入（显露 env 赢 adapter 裁决链——**路由是被证明的，不是被假设的**）。红=记录原因（rc+日志尾 200 字符）降下一档。
3. **性能门**：过门档位之间同窗微基准（统一 harness：cuda.Event，warmup5+iters20；≥2 档可测才 A/B，快者胜）。

**降档语义**：判据全红 → 保底默认档 + 报告写明 "all ladder tiers failed"（**诚实降级，不假绿**）；无判据的 op（不在 choices.OPS）→ 直接默认档 + "no-judge fallback" 标注；与 catalog auto 规则结论冲突时以实测为准、报告注明分歧。

## 覆盖面（首版）

- **可测 10 op**：silu_and_mul / rms_norm / rotary_embedding / moe_gate / paged_mqa_logits / flash_attention / per_token_quant / conv1d / kda_prefill / linear_attn_gdn。
- **显式跳过 10 op**（延续 auto 规则）：platform_detect / model_env / kv_pool_strides / dist_broadcast / triton_pdl / triton_libdevice / allreduce（需双机世界做有意义裁判）/ sampling / fused_moe / w4a16_moe（bench 接线 v2）。
- 扩覆盖 = 在 `choices.py` 加条目（judge 已存在的测试文件 + env 键值），目录交叉一致性由门禁 `_test_autotune.py` 钉死（判据文件不存在/env 键不在 hcu_env.REGISTRY 即红）。

## 平台画像与已知坑快捷规则（probe）

画像项：arch（`hcu_env.arch_name`）/ fp8 能力 / torch(hip?) 与 dtk 版本尾 / aiter / tilelang / deep_gemm / sglang 版本 / hyhal 挂载 / rocm-smi 显存表 / **验活二连**（device_count + 真张量 matmul，子进程）。

硬编码坑规则（bootstrap §5 账本机械化，level=FAIL/WARN/ok/note）：

| 检查 | 处置 |
|---|---|
| torch 真 init 失败/超时（hyhal 缺挂载、驱动错配"半配对"） | **FAIL → 全轮中止**（先修再测） |
| tilelang ≠ 0.1.12 | WARN（mHC 融合 ThreadSync 回归，钉版提示） |
| NCCL_TOPO_FILE 存在（异构盒拓扑） | WARN（unset 配方） |
| ROCR_VISIBLE_DEVICES 与 CUDA 语义双设 | WARN（假死配方） |
| HIP torch 但 rocm_version 不可读 | WARN（deepgemm 桥/JIT 会拒） |

## 报告与审计

`autotune_report.json`：`platform`（完整画像+traps）+ `decisions[]`（每 op：default/ladder/gates[]逐门读数/chosen/source/bench 读数）+ `summary`（by_source 计数、非默认清单）。`source ∈ {measured, ladder-order, ladder-order(dry), fallback-default}`——**measured 才是实测选档**，其余两类是降档/无判据保守值，接手的人一眼可辨证据等级。

## 设计边界（首版非目标）

- serving 级端到端评估不做（catalog 正式晋级仍走人工同窗 A/B 流程，判据见 governance §5.2 / models.md 台账）；工具产出是**建议配置**，不是 production 判定。
- 多机通信档（allreduce/网卡/RDMA/EP）不做——v2 列表；首版 allreduce 沿用 auto 规则（aiter 布局嗅探）。
- 性能数字**仅本机有效**（仓内共识：数字不跨机迁移）；同 op 跨机对比请各自跑 autotune 后对报告。
- bench 工厂首版注册 mock 类（harness/门禁可跑无卡机）；真算子工厂随 on-card 验证逐个落 `bench.py::SHAPES`（键 `kind:tier`），未注册=报告标 unbenched，不编造数字。

## 门禁

`python tests/tools/_test_autotune.py`（24 钉，无卡可跑）：阶梯语义（broken 拒/降档/穷竭保底/无判据回退）、产物闭环（yaml 仅非默认+re-parse+validate、报告计数守恒）、choices 卫生（目录全覆盖/判据文件与 env 键真实）、bench 诚实性（未注册 not-ok/mock anywhere）。
