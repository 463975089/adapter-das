# 测试与路由治理工程实施计划（implementation plan）

> 本文把 [test-route-governance.md](test-route-governance.md) 的目标态拆成可合并、
> 可回滚、可验收的工程任务。后续实施以本计划的 MR 编号、依赖和 DoD 为准；
> 方案原则变更先改治理文档，任务依赖或实现细节变化改本文。
>
> **基线锚**：社区 sglang v0.5.22 与对应 adapter 分支。执行环境必须遵守
> “adapter 分支 ↔ 上游源码树 ↔ 测试容器/镜像”三绑定；具体节点、容器名和挂载目录
> 属于部署侧配置，不写入公共仓库。本文统一使用 `$ADAPTER_ROOT`、`$UP_TREE`、
> `$RESULT_ROOT` 和 `<GPU_ID>`；耐久产物写入 `$RESULT_ROOT`，不得依赖临时目录。
> 运行前由执行者确认所选设备空闲且不会影响已有服务。

## 0. 实施边界与成功标准

### 0.1 首期范围

首期只闭环 10 个高频 op，不追求一次覆盖整个 catalog：

```text
silu_and_mul / rms_norm / rotary_embedding / per_token_quant /
kv_cache_store / flash_attention / paged_mqa_logits /
moe_gate / moe_align / conv1d
```

其中 `kv_cache_store` 若进入 `BACKEND_CATALOG` 时命名不同，MR-01A 先定 canonical
key；此后 manifest、telemetry 和报告只使用 canonical key，别名只在入口归一化。

首期不做：全量多机性能平台、在线时序数据库、Web dashboard、生产常开逐调用计时、
自动删除代码、自动把 candidate 晋升 production。报告先落 JSON/Markdown 文件，
晋级和删除仍经 review。

### 0.2 首期完成定义（Program DoD）

同时满足才算首期完成：

1. L0 基线为绿，且新增规则无未解释红项；
2. L1.5 v2 使用版本化固定 manifest，除现有 Attention/Mechanism 外覆盖基础 ABI、
   Quant/MoE、Attention kernels、Mamba/Conv1D 和模型契约；新增/消失 nodeid 与错误
   指纹漂移可见；
3. 首批 10 op 每个都有独立参照、至少一个 active 测试和 route evidence；
4. 核心测试中 `unknown route = 0`，每个 stub/fallback 都有 reason；
5. 可对同一 op/shape 强制 emu（或 torch）、triton、vendor 可用档并先精度后性能；
6. 一次运行产 `report.json` 和 `report.md`，能合并 pytest、路由与 benchmark 数据；
7. 完成一次 candidate→validated 的证据演练和一次 deprecated→retired 的全链演练；
8. MR 快车保持单卡、分段超时和可中断；更重的 contract/diagnostic/性能测试留扩展验证。

## 1. 现状基线与前置清债

### 1.1 已有可复用机制

| 资产 | 当前能力 | 本计划复用方式 |
|---|---|---|
| `BACKEND_CATALOG` | op、backend、状态、配置裁决 | canonical op/backend 与生命周期事实源 |
| `registry.py` | 安装态 binding、unbound、env dump | 保持安装态职责；运行态数据单独模块，dump 时汇总 |
| `HCU_ADAPTER_DUMP` | 启动时输出安装态 JSON | 向后兼容；schema version 后增加 `runtime` 可选区 |
| `hcu_config_wizard --autotune` | 正确性门、梯级选择、微基准报告 | 性能 harness 与 shape catalog 复用，不另造一套裁判 |
| `ci/l1_gpu_gate.sh` | 树绑定、GPU 选择、超时、分段日志 | 新 manifest runner/report 段接入现有框架 |
| `ci/cache_export/import.sh` | sglang JIT + Triton 缓存种子 | 冷/暖结果分账，缩短扩展验证墙钟时间 |

### 1.2 当前已知阻塞基线

当前 L0 已知 `2 fail / 56 warn`：

- `HCU_MOE_ALIGN_FUSED` 未登记 `docs/env-vars.md`；
- `wna16_channel_moe.py` 注册 `moe`，但 catalog 无该 key。

这两个问题先用独立 **MR-00** 修到 L0 绿，不能把历史红项带进新机制后再判断“是否
由 telemetry 引入”。同时把 `tests/adapter/test_registry.py` 的断言改为正常 pytest
可收集函数；当前断言只在 `main()` 中执行，`pytest tests/adapter` 不会真正跑到。

代码审阅还确认了四个必须早于 telemetry 的契约缺口：

1. catalog 没有显式 `default`，向导靠“第一个 production”推断，机器治理不能长期
   依赖字典插入顺序；
2. catalog 的 `var=None` 不代表 backend 可强制切换，必须显式登记 selector/
   `switchable`，runner 不能从 backend 名猜；
3. wizard 当前把 `broken` 列为 selectable，但 config validator 会拒绝，选择面和
   生效面不一致；
4. registry 的 `_RECORDS[op]` 会被同 op 后续 target 覆盖，且 `ROLE_REF` 目前只打印
   debug 日志；安装态必须先能表达“一 op 多 binding、一个 active implementation”。

这些问题分别由 MR-01A/MR-01B 修复，且只镜像当前行为，不在同一 MR 翻默认档。

### 1.3 基线冻结产物

在与分支绑定的测试环境中做一次不改行为的基线运行，产物目录：

```text
$RESULT_ROOT/test-governance-baseline/<run-id>-<upstream-version>/
  environment.json
  l0.log
  l1.log
  l15_upstream.log
  l15_mech.log
  collect-core.txt
  collect-mech.txt
  registry-dump.json
  README.md
```

记录冷/暖缓存、GPU、耗时和现有 known-fail；该目录不提交仓库，只在实施期间作为
A/B 参照。

## 2. 目标代码布局

首期建议布局如下；具体文件名允许在 MR review 时微调，但职责不得重新混进 bridge：

```text
sgl_kernel/_adapters/
  registry.py                 # 安装态；只做兼容扩展
  route_telemetry.py          # 运行态计数、shape bucket、flush、合并
  route_schema.py             # schema version、backend/reason 枚举、校验

tests/manifests/
  l15_v0522.json              # core/contract/diagnostic/skip 精确 nodeid
  route_coverage_v1.json      # 首批 op 的预期 route/shape/phase

tests/route/
  test_route_schema.py        # CPU
  test_route_counter.py       # CPU，多进程/flush/禁用态
  test_route_report.py        # CPU，聚合与漂移

tools/
  l15_manifest.py             # collect/verify/run；不改上游测试
  route_report.py             # 合并 pytest/route/bench，产 JSON+Markdown
  route_assets.py             # catalog/registry/code/test/last-hit 对账

tools/_autotune/
  choices.py / bench.py       # 复用现有 autotune 实际模块位置，补 shape/档位

docs/schemas/
  route-event-v1.json         # 数据契约
  test-report-v1.json

ci/
  l1_gpu_gate.sh              # 接 runner，不承载解析业务
  extended_route_gate.sh      # 扩展功能/性能/资产扫描入口
```

`route_telemetry.py` 必须 framework-light：模块 import 不得 eager import sglang；CPU
单测无 torch 也能覆盖 schema/聚合核心。算子 wrapper 可传 tensor metadata，但聚合层
只接收已归一化的标量/字符串。

## 3. 数据契约先行

### 3.1 Route event v1

必填字段：

```json
{
  "schema_version": 1,
  "op": "silu_and_mul",
  "backend": "triton",
  "implementation": "sgl_kernel.elementwise.silu_and_mul",
  "phase": "decode",
  "graph_mode": "eager",
  "shape_key": "bf16:m=1,h=7168",
  "count": 128,
  "reason": "catalog-default"
}
```

可选字段：`dtype`、结构化 `shape`、`capture_count`、`replay_count`、`latency`、
`vendor_version`、`nodeid/request_id`、`rank/pid`。未知字段允许 reader 忽略，以保证
v1 小版本前向兼容；缺必填字段或未知 enum 必须响亮拒绝。

backend 固定为：

```text
vendor / triton / emu / torch / upstream / stub / unknown
```

reason 首版至少包含：

```text
catalog-default / config / explicit-debug / auto / unsupported-shape /
backend-unavailable / correctness-fallback / exception-fallback /
platform-skip / loud-stub / unknown
```

### 3.2 Test report v1

顶层固定：

```text
schema_version / run / environment / tests / accuracy / features /
routes / benchmarks / governance / verdict
```

所有分段工具只写自己的 fragment；`route_report.py merge` 最后生成总报告。任何分段
失败也要留下 fragment 和退出码，禁止因为 pytest 红而没有报告。

### 3.3 文件写入规则

- 每进程写独立 `route.<run_id>.<rank>.<pid>.json`，避免并发覆盖；
- 先写 `.tmp`，flush+close 后原子 rename；
- 聚合器按 schema/version/run_id 校验后合并；
- 默认输出根由 CLI `--out-dir` 给定；CI 使用仓外耐久目录；
- 禁止多进程共享一个可变 JSON 文件。

## 4. MR 路线图（小步、可回滚）

### MR-00：修绿基线与 registry 测试收集

**范围**：只修两项既有 L0 红；把 registry `main()` 断言拆成 pytest tests，保留
standalone 入口可选。

**主要文件**：`docs/env-vars.md`、`hcu_config.py` 或注册 op 的归一化修复、
`tests/adapter/test_registry.py`。

**验收**：

```bash
python ci/l0_gate.py
python -m pytest tests/adapter/test_registry.py -q
```

**DoD**：L0 零 fail；不得顺手加入 telemetry。

---

### MR-01A：catalog/config 契约冻结（CPU-only）

**依赖**：MR-00。

**范围**：新增两个 JSON schema、枚举/校验器、空报告骨架；定下首批 10 op 的
canonical key。同时给 catalog 增加显式 `default` 和 selector 能力描述：

- `default` 只镜像当前第一个 production/实际自动规则，不在本 MR 翻默认；
- `switchable` 或 `selector` 明确 `config/env/auto/fixed`，`var=None` 不再被误判为可强制；
- default 必须存在、不可指向 broken/planned；
- wizard 不展示 broken/planned 为 selectable，写盘前再走 `_validate`；
- status/default/vars/selector 的纯结构校验进入 CPU 测试。

**验收**：

```bash
python -m pytest tests/route/test_route_schema.py tests/route/test_route_report.py -q
python tests/tools/_test_autotune.py
python ci/l0_gate.py
python tools/route_report.py init --out-dir <dir>
python tools/route_report.py validate <dir>/report.json
```

**DoD**：schema round-trip、未知 enum/缺字段拒绝、空分段不伪造“通过”；显式 default
与改动前解析结果逐 op 一致；broken 不可被向导选中。

**回滚**：删除新增独立文件和 catalog 元数据即可；无生产默认变化。

---

### MR-01B：安装态 registry 加法增强（CPU-only）

**依赖**：MR-01A。

**范围**：先修安装态事实，再做 runtime telemetry：

- registry 用 `op → bindings[]` 表达同一 logical op 的多个 target；
- 禁止的是多个 active implementation 冲突，不是同实现绑定多个 target；
- ref 使用独立记录或显式拒绝生产 binding，不进入 active `_BOUND`；
- `dump()` 增加 `schema_version`/`installed`，但旧 `ops/unbound/env` 至少保留一个
  保留到旧消费者迁移验证全部通过，旧消费者不改即可继续工作；
- 文件 dump 使用临时文件+原子 replace。

**验收**：

```bash
python -m pytest tests/adapter/test_registry.py \
  tests/adapter/test_sitecustomize_sequence.py -q
python ci/l0_gate.py
```

**DoD**：多 target 不丢记录；不同 active implementation 冲突响亮失败；ref 不能被
生产路由；旧三键和 `HCU_ADAPTER_DUMP` 契约仍可读。

**回滚**：旧键始终保留；runtime 尚未接入，可独立回滚。

---

### MR-02A：L1.5 v2 候选盘点与分类（observe-only）

**依赖**：MR-01A。

**范围**：以 [l15-test-inventory.md](l15-test-inventory.md) 的逐文件候选账为输入，建立
候选扫描器，对社区 v0.5.22 的 Attention、kernels/ops、rotary、MoE、Mamba 和无权重
模型契约测试执行 collect-only，并输出机器清单。每个候选记录：

```text
nodeid / domain / logical ops / expected backend / route evidence /
device_count / needs_server / needs_model / external deps /
platform guard / timeout / current result / owner
```

首批候选域：

```text
现有 attention core/mechanism
activation + layernorm
rotary + sampling
kvcache
quantization + moe
attention lower-level kernels
mamba/conv1d
model contract tests without weights
```

本 MR 不改现有门禁、不宣称候选已经支持；不能自动判断的条目标 `unclassified`。

**验收**：

```bash
PYTHONPATH="$UP/python:$ROOT" python tools/l15_manifest.py inventory \
  --up-tree "$UP" --manifest tests/manifests/l15_v0522.json \
  --out <dir>/l15-inventory.json
python tools/l15_manifest.py validate-artifact <dir>/l15-inventory.json
```

**DoD**：同一树两次 inventory 除时间字段外稳定；server/model/multi-GPU/external-dep
用例可被过滤；所有条目有来源文件和分类理由。

---

### MR-02B：现有 L1.5 精确 manifest（observe-only）

**依赖**：MR-02A。

**范围**：先把当前已运行的 attention core/mechanism 转为
`core/contract/diagnostic/skip/unclassified` 精确 nodeid；known-fail 记录错误正则、owner、
复审条件和转绿条件。manifest 按上游版本/commit 分文件，v0.5.21 与 v0.5.22 不共用
nodeid。现有目录门和 `l15_mech` 总数预算继续裁决，新 manifest 只产生漂移报告。

**验收**：

```bash
python tools/l15_manifest.py validate tests/manifests/l15_v0522.json
PYTHONPATH="$UP/python:$ROOT" python tools/l15_manifest.py collect \
  --up-tree "$UP" --manifest tests/manifests/l15_v0522.json --out-dir <dir>
python tools/l15_manifest.py verify --mode observe \
  --manifest tests/manifests/l15_v0522.json --collect <dir>/collect.json
```

**DoD**：两轮 collect 稳定；新增/消失/skip/fingerprint 漂移可见；树或版本错配 loud
fail；现有 235 个左右收集实例全部有分类或明确 `unclassified`。`run` 只接受 manifest
内的精确 nodeid，并要求显式选择 classification，不透传任意 pytest 参数。

---

### MR-02C：基础 ABI Batch 1 影子运行

**依赖**：MR-02B。

**范围**：把 [l15-test-inventory.md](l15-test-inventory.md) §4.1 的基础 ABI 文件加入
v2 manifest 的 observe 集，覆盖：

```text
activation fused-op parity
layernorm fused-op parity / rmsnorm / fused-add rmsnorm
rotary ROCm
sampling renorm / top-p
kvcache store-cache / set-MLA-KV-buffer
```

先完成收集和单卡实跑分诊。文件内 CPU-only、GPU parity、平台 skip 等参数实例必须逐
nodeid 分类，不能整文件标绿。收集错误进入 diagnostic，合法平台 skip 单列，只有真实
执行且稳定通过的 nodeid 才标 core-green。已知 layernorm parity 收集问题必须留下精确
错误证据，不能整文件静默剔除。

**DoD**：每个 nodeid 有 logical op、当前结果、路由要求和耗时；连续两跑绿色集合稳定；
影子运行不阻塞 MR，但新增 import/ABI 缺口进入报告。

---

### MR-02D：Quant/MoE/Attention/Mamba 分批纳入

**依赖**：MR-02C；可拆 D1–D4 小 MR。逐文件范围与初始分类以
[l15-test-inventory.md](l15-test-inventory.md) §4.2–§4.4 为准。

| 子批 | 范围 | 重点 |
|---|---|---|
| D1 | FP8 aiter utils、per-token/group quant | 顶层 ABI、scale/layout、bit/dequant；`sgl_per_token_quant_fp8` 缺口先 diagnostic |
| D2 | MoE TopK padded/degenerate、gate/align | expert 集合、padding、renorm、graph replay |
| D3 | KV indices、decode metadata、Triton attention、FP8 KV、verify KV | prefill/extend/decode 底层调用面 |
| D4 | causal-conv、SConv cache/metadata/tracking/replay | Mamba/Conv1D state 和页边界 |

每个子批最多新增一个测试域，先 observe、后修 ABI/精度、再把稳定 nodeid 提升到
core-green。完整服务、多卡、模型权重、外部存储和纯性能测试保持排除。

**DoD**：每个新增 core nodeid 实际执行而非平台 skip；对应 adapter seam 可证明；失败
按 adapter/upstream/vendor/DTK 归因并有 owner/转绿条件。

---

### MR-02E：core-green 转硬门与模型契约扩展段

**依赖**：MR-02B 的 inventory/manifest 连续两轮无非预期漂移；MR-02C/D 的目标批
连续两跑稳定；MR-05 报告可用。

**范围**：

- 只把稳定 `core-green` 切为 MR 阻塞；
- `contract-green` 建独立扩展验证段，按 [l15-test-inventory.md](l15-test-inventory.md)
  §4.5–§4.6 逐 nodeid 收无权重的模型构造、quant config、packed weight mapping/loader、
  pipeline/embedding registration；不得按 `unit/models` 整目录纳入，也不得计入 kernel
  route coverage；
- diagnostic 以 nodeid+fingerprint 检查，错误指纹变化打红；
- platform-skip 检查 skip reason；
- 保留旧目录门直到新旧裁决结果连续一致，确认后删除总数预算。

**验收**：构造“旧失败修复 1 个、新回归增加 1 个、总数不变”的 fixture，新门必须
判红；删除或改名 core nodeid 必须判集合漂移；满足复审条件但未处理的 diagnostic 至少 WARN。

**DoD**：MR core 目标 250–350 个稳定实例，保持单卡、可分段和可配置超时；
contract/diagnostic 扩展段在扩展验证入口运行；有效覆盖分母不含合法 skip。

---

### MR-03：运行态计数核心（CPU-only、默认关闭）

**依赖**：MR-01A。

**范围**：实现 `route_telemetry.py` 的 enable/record/snapshot/flush/reset/merge；不接
任何 op。新增 `HCU_ROUTE_TRACE=off|count` 和 `HCU_ROUTE_OUT` 时，同 MR 登记
`hcu_env.REGISTRY` 与 `docs/env-vars.md`；默认 `off`。

**关键约束**：

- 禁用态快速返回；
- 不 import sglang、不做 GPU 同步；
- shape 从调用者传入，不在聚合层读 tensor；
- 线程安全；fork/spawn 后 pid 隔离；
- 计数失败不得改变算子输出，但在测试模式可配置 fail-loud。

**验收**：

```bash
python -m pytest tests/route/test_route_counter.py -q
python ci/l0_gate.py
```

**DoD**：10 万次禁用态/计数态 CPU microbench 留基线；并发计数守恒；原子落盘。

---

### MR-04：两个试点 op 接线与 route evidence

**依赖**：MR-03。

**范围**：只接 `silu_and_mul` 和 `paged_mqa_logits`：前者覆盖 config/多 backend/
现有数值基础，后者覆盖 emu/triton 与真实上游门禁。首个埋点 MR 不接 10 个 op；
验证 schema、开销和 graph 口径后再按 2→5→10 扩展。记录最终 dispatch 结果，
不在 config lookup 处提前记账，也不给每颗底层 kernel 重复记账。

**接线规则**：

- 一个 logical 调用只记一条主 route；组合 emu 可在 `children`/detail 中列子路径，
  不重复计入主调用总数；
- fallback 必须记录原 backend、最终 backend 和 reason；
- 异常后改走 fallback 时只在成功返回后记最终 route，同时计 fallback reason；
- graph capture 首版标 `graph_mode=capture`，普通 replay 逻辑计数留 MR-08。

**验收**：对应 `silu_and_mul` 与 paged-MQA logits 的 L1 精度测试和上游候选；报告
断言每个强制档位实际命中且 count>0，非目标 backend=0，禁用 trace 时输出一致。
Graph 首版只能报告 eager/capture/unknown，`logical_call_count` 必须是 `null` 或
`not_collected`，不得用 wrapper 次数冒充 replay。

**DoD**：两个试点 op 的 route coverage 100%；`unknown=0`；trace off 不改变测试结果。

---

### MR-05：安装态+运行态报告合并

**依赖**：MR-01B、MR-02B、MR-03；可与 MR-04 后半并行开发。

**范围**：扩展 registry dump 为带 `schema_version` 的兼容结构；保留现有
`ops/unbound/env` 键，新增可选 `runtime` 或 fragment 引用。实现 `route_report.py
merge/render/diff`。

**输出**：

- `report.json`：完整机器数据；
- `report.md`：环境、测试、精度、路由、性能、治理七段；
- `summary.txt`：供 MR watcher 粘贴的短摘要。

**验收**：旧 reader 仍能读 `ops/unbound/env`；多 rank 合并计数守恒；无 route fragment
时明确写 `not_collected`，不能写 0 假装没有调用。

---

### MR-06：L1/L1.5 门禁接报告，v2 manifest 影子运行

**依赖**：MR-02B、MR-05；MR-02C/D 可在该框架上逐批增加影子段。

**范围**：`l1_gpu_gate.sh` 新增 `--out-dir`/run id，把日志从固定 `/tmp/l1_*` 迁到
指定结果目录（保留默认兼容）；MR-02B/C/D manifest runner 作为非阻塞影子段，现有
目录段仍裁决。MR-02E 单独负责把稳定 core 转硬门，不在本 MR 提前切换。

**验收命令**：

```bash
bash ci/l1_gpu_gate.sh <GPU_ID> \
  --up-tree "$UP_TREE" \
  --out-dir "$RESULT_ROOT/<run-id>"
```

**DoD**：成功、pytest 失败、超时三种情况下都产报告；不会覆盖另一并发 run；树、GPU、
cache seed 和各段耗时进入环境信息。

---

### MR-07A/B/C：按 2→5→10 扩展算子精度与路由覆盖

**依赖**：MR-04、MR-06。它与 MR-02C/D 的上游用例扩充共享 logical-op 映射，但职责
不同：MR-02 扩社区测试面，MR-07 补 adapter 自有精度与运行态证据。每个 MR 最多接
2–3 个 op，避免横切大爆炸。

| MR | 新增 op | 累计 | 重点 |
|---|---|---:|---|
| 07A | rms_norm / rotary_embedding / per_token_quant | 5 | fused norm、RoPE native/vendor、FP8 scale/bit/ABI |
| 07B | kv_cache_store / moe_gate / moe_align | 8 | 页表/索引、TopK/padding、graph replay |
| 07C | flash_attention / conv1d | 10 | causal/window/GQA/varlen、state/cache/页边界；paged-MQA 已在试点 |

每个 op 的 DoD：

```text
独立 reference ≥1
active backend test ≥1
fallback/ref test ≥1
force-route evidence ≥1
graph test（decode 路径适用时）≥1
shape source 与 tolerance 有登记
```

同时解决上游 v0.5.22 已发现的 ABI 缺口（如 `sgl_per_token_quant_fp8`），但 ABI 修复
必须有独立回归，不在 route wrapper 中偷偷 alias。

---

### MR-08：Graph logical replay 计数

**依赖**：MR-07。

**范围**：在 graph runner/scheduler 官方 seam 记录 capture id 与 replay 乘数，把
`python_call_count/capture_count/replay_count/logical_call_count` 分开。不得从 GPU
回读统计值。

**验收**：构造 capture 1 次、replay N 次、每 replay 含 K 个 logical op 的测试，断言
逻辑次数为 N×K；换新输入后结果跟随；trace off 完全不接 seam。

**DoD**：报告不会把 capture 一次误报成模型只调用一次；多 graph bucket 分账。

---

### MR-09：性能 harness 与三档同窗 A/B

**依赖**：MR-07；MR-08 对 graph benchmark 必需。

**范围**：扩展现有 autotune `choices/bench`，不另建互相冲突的 benchmark 框架；加入
shape catalog、强制档位、精度前门、Event 采样、冷/暖分账和 JSON fragment。

执行协议：

```text
固定输入/seed → 强制 backend → route evidence → 精度门 → warmup → Event iters
```

每个结果记录：p50/p95/max、吞吐、显存/workspace（可得时）、首次编译、暖态、
production 相对加速比、失败/fallback 原因。

**验收**：首批 10 op 中至少 6 个有两档以上可比数据；只有一档的诚实标
`single-tier`，无 factory 的标 `unbenched`，禁止造 0。

**DoD**：连续两轮排序稳定；route evidence 与被测档一致；性能模式不进 MR 快车。

---

### MR-10：L2-F 小模型功能矩阵

**依赖**：MR-06、MR-07。

**范围**：从历史可用模型中选最小覆盖集，首版建议：

- Qwen3-0.6B：dense/基础 attention/rope/KV；
- 一个小型 MoE 或可控 MoE fixture：MoE gate/align；
- 一个机制模型/fixture：Mamba/conv1d 或 DSA/MLA。

复用 `tests/tools/run_model_suite.sh` 和 cookbook，不把启动逻辑复制进报告工具。输出
launch/generation/accuracy/long-context/graph 与 route fragment。

**DoD**：失败报告能回答“在哪个阶段失败、此前命中过哪些 backend”；单卡 smoke
进入扩展验证，多卡组合留后续专项入口。

---

### MR-11：资产扫描器（先 WARN）

**依赖**：MR-05、MR-07。

**范围**：实现 `route_assets.py`，输出：catalog-unbound、code-not-catalogued、
registered-not-hit、hit-not-tested、duplicate-active、ref-routed、target-missing、
env-not-read、deprecated-expired。

**约束**：首版只输出证据与 WARN；静态扫描无法确定的项标 `needs-review`，不自动删。
稳定运行且误报归零后，只有 deterministic 规则升 L0 HARD。

**DoD**：每个发现包含 file/line 或 registry key、证据类型、owner（可空）和建议动作；
同一基线两跑结果稳定。

---

### MR-12：生命周期与两次治理演练

**依赖**：MR-09、MR-11。

**范围**：catalog/schema 增加 `deprecated`/`retired` 语义；完成：

1. 一个有性能证据的 candidate→validated 演练；
2. 一个确证无调用、无测试需求、无回退价值的 deprecated→retired 全链演练。

退役 MR 必须同时处理：实现、import、registry、catalog、env、示例配置、文档、测试和
known-fail/manifest。`retired` 历史记录放治理账，不作为可选 backend 留在运行 catalog。

**DoD**：L0 能抓住故意遗留的幽灵 env/import；回滚仅需恢复该 MR，不影响其他 op。

## 5. 依赖关系与并行条件

```text
MR-00
  └─ MR-01A ── MR-01B
       ├─ MR-02A ─ MR-02B ─ MR-02C ─ MR-02D ───────── MR-02E（硬门切换）
       │              │                  │                 ▲
       └─ MR-03 ── MR-04 ────────────── MR-05 ── MR-06 ──┘
                                             ├─ MR-07A ─ MR-07B ─ MR-07C
                                             │              ├─ MR-08
                                             │              ├─ MR-09
                                             │              └─ MR-10
                                             └──────────── MR-11
                                                            └─ MR-12（收口）
```

卡不可用时可优先推进 CPU-only 的 01A/01B/02A/02B/03/05/11；02C/02D 的收集与
schema 可先完成，结果分类和转绿必须使用真实 HCU。任务是否进入下一阶段由前置 DoD
决定。

## 6. 分支、MR 和评审纪律

- 一个 MR 只做一个编号；02D 按 D1–D4、07 按 A/B/C 拆分；
- 不修改社区 sglang 源码；上游测试只通过 manifest 引用；
- 每个 MR 描述必须列：行为变化、schema 变化、测试、GPU 证据、性能影响、回滚；
- 新 env 与 `hcu_env.REGISTRY`、`docs/env-vars.md` 同 MR；
- 新 catalog key/backend 与 config 文档、测试同 MR；
- 数据 schema 只做向后兼容新增；破坏性变化升 `schema_version`；
- `route_telemetry` 接线不能顺带重写算子数学；发现精度 bug 单独修复和回归；
- 所有长期产物使用 run id，不使用固定日志名，不覆盖历史结果。

每个 MR 的通用最小检查：

```bash
python ci/l0_gate.py
python -m pytest tests/adapter tests/route -q
# 涉及 kernel/路由时：
bash ci/l1_gpu_gate.sh <GPU> --up-tree <sglang-v0.5.22> --out-dir <run-dir>
```

## 7. CI 与资源预算

### MR 快车

- 1 卡；L0、CPU、L1、L1.5 core；
- telemetry 使用 count 模式，只对首批 op；
- 不跑性能采样和完整 L2；
- 各段使用可配置超时，超时即留报告并失败；
- manifest 迁移期间“旧段裁决+新段影子”，满足 MR-02E 的稳定性条件后切换。

### 扩展验证

- 1–4 卡分域运行；先预热/导入匹配 cache seed；
- L1/L1.5 全量、小模型 L2-F、三档 benchmark、资产扫描；
- 生成本次报告与上一次成功扩展验证报告的 diff；
- GPU 并行时每个 worker 独立 route fragment 和 Triton cache 目录，结束后合并。

### 发版/专项验证窗口

- 多卡 TP/EP/PD、长上下文、压力和大模型；
- deprecated 审核；
- 更新 cache seed 和环境 manifest；
- 审核性能数字仅在相同环境指纹下比较。

## 8. 报告门槛与判定策略

首期门槛：

| 指标 | MR | 扩展验证 |
|---|---:|---:|
| L0 fail | 0 | 0 |
| core-green pass | 100% | 100% |
| unknown route | 0 | 0 |
| active 无测试 | 0（首批 op） | 0（已纳入 op） |
| 新 stub | 0 | 0；例外需登记 |
| manifest 漂移 | 阻塞 | 阻塞并出 diff |
| 性能回退 | 不常规执行 | 按 op 阈值；默认模板 >5% 需复验 |
| stale/orphan | 初期 WARN | 有 owner/复审条件；确定性规则成熟后阻塞 |

性能回退须连续复验，且环境指纹一致；单次 p95 抖动不直接改 production。精度失败
永不因性能更优豁免。

## 9. 风险、缓解和回滚

| 风险 | 表现 | 缓解/回滚 |
|---|---|---|
| 观测影响热路径 | 延迟上升、Graph 失败 | 默认 off；MR-03 做开销基线；trace off 一键回滚 |
| Graph 次数失真 | capture=1 被当真实调用数 | MR-08 前明确标 `capture-only`，不计算 replay 覆盖 |
| 多进程文件踩踏 | JSON 损坏/丢计数 | pid/rank 独立文件+原子 rename+离线合并 |
| shape 基数爆炸 | 报告巨大、内存增长 | 归一化 bucket、每 op 上限、overflow bucket |
| nodeid 上游漂移 | 全量误报 | manifest 绑定 upstream 版本；collect diff 人工审核更新 |
| known-fail 掩盖回归 | 同数量失败互换 | nodeid+指纹+复审条件，不用总数终态 |
| 性能数据不可比 | 错误晋级 | 环境指纹、route evidence、同窗 protocol；不符即 invalid |
| vendor 不稳定 | import/ABI/数值失败 | vendor 档独立失败，production 保留 triton/emu 回退 |
| stale 扫描误删 | 删除仍有隐藏场景代码 | 首版 WARN；route/test/upstream 三证与 owner 复核后才删除 |
| schema 锁死 | 后续字段难扩 | versioned schema、未知字段容忍、必填最小化 |

任何 MR 导致 serving 回归时优先关闭 `HCU_ROUTE_TRACE`；telemetry 与报告层不得成为
模型可用性的强依赖。算子本身的 kill switch 沿用现有机制，不为统计系统创造第二套路由。

## 10. 开工前检查清单

- [ ] 确认 adapter 分支、`$UP_TREE` 与测试容器/镜像满足三绑定；
- [ ] 确认 `<GPU_ID>` 空闲且测试不会影响已有服务；
- [ ] 建立基线结果目录并记录环境指纹；
- [ ] MR-00 把 L0 修绿；
- [ ] 为 MR-01A 冻结首批 op canonical key、显式 default 与 selector；
- [ ] 确认 report/schema 评审人和各 op owner；
- [ ] 约定扩展验证结果的保留策略和磁盘配额；
- [ ] 确认 manifest 切换所需的稳定性与裁决一致性条件；
- [ ] 不在社区 sglang 树中写任何实现或测试修改。

## 11. 文档维护与状态回写

每个 MR 合并后在本文对应标题追加状态标记：`未开始 / 进行中 / 已合并(<hash>) /
阻塞(<原因>)`，并回写验收证据与实际偏差。MR-06 后由报告自动生成完成度摘要，
但人工验收结论仍回写本文。

以下变化要求同 MR 更新本文：目录布局、schema、MR 依赖、首批 op、门槛、资源预算或
回滚开关。目标原则变化则先改 [test-route-governance.md](test-route-governance.md)。
