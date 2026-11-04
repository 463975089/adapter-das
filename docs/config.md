# HCU Adapter 后端配置（hcu_adapter.yaml）

## 是什么

同一算子往往有多个可用来源：adapter 自带 triton 内核（`sgl_kernel`）、DTK vendor 轮
（aiter / lightop）、以及 torch 参考实现。**选哪个后端是部署决策，写进配置文件**
（用户口径 2026-10-12），不再靠环境变量。

## 配置文件

```yaml
version: 1
ops:
  silu_and_mul: lightop     # triton | lightop | torch
  rms_norm: aiter           # triton | aiter | torch
  allreduce: aiter          # aiter（默认落点）| nccl；custom AR 已除役删除（R-01 定罪+2026-10-19 R-13 #1 改判废弃）
  paged_mqa_logits: triton  # triton（R-13 #4 晋级默认）| emu | vendor（broken 选配即拒）
  fused_moe: auto           # auto（aiter 先试/无调优形状回 triton）| aiter | triton
  w4a16_moe: auto           # auto（现答 triton）| deepgemm（R-05 桥，候选 opt-in）
```

- **查找顺序**（先中先得）：`$HCU_CONFIG` 指定路径 → `./hcu_adapter.yaml|.yml|.json`
  → adapter 根目录（sitecustomize.py 旁）同名文件。
- **未写条目 = 生产默认**；文件里只写要覆盖的算子。
- **校验是严格的**：未知算子名、未知后端值、选择 `planned` 状态的后端，都会在
  加载时**响亮报错**（防手滑静默回退默认）。启动日志出现
  `HCU adapter config: <路径> (ops: ...)` 一行即为生效。
- 部署样例见仓根 `hcu_adapter.example.yaml`；交互式生成：
  `python tools/hcu_config_wizard.py`（逐算子提示选择，只写入非默认项）。
  `--list` 看后端目录表，`--show` 看当前生效配置，`--check 文件` 校验。

## 优先级（裁决链）

```
显式环境变量（调试层，仅排查用）  >  配置文件 ops.<op>  >  平台自动规则  >  内置默认
```

环境变量降级为**调试覆盖**：临时一键回退无需改文件；正式部署**零环境变量**，
全部选择落在配置文件里。启动汇总行标注每项来源（`explicit`/`config`/`auto`/`default`）。

## 后端目录（BACKEND_CATALOG）

vLLM 0.33/gfx936 的 C4 indexer 候选通过 `ops.vllm_sparse_indexer: native`
显式选择；默认 `stock` 保留官方行为。配置文件、调试 env 与快关遵循同一
裁决链；注册后改变策略须新进程。限制与验证见
[vLLM C4 indexer](frameworks/vllm033-sparse-indexer.md)，不表示完整 V4 已通过。

`hcu_adapter/hcu_config.py::BACKEND_CATALOG` 是唯一事实源：算子 → 可选
后端 → 状态与备注。向导和未来的 Web UI 都从它渲染，新后端接线后自动出现在选择项里。
**实现归籍与角色（active/ref）规约见 [governance.md](governance.md)**：算子数学
只在 `sgl_kernel/`，行为桥放 `hcu_adapter/<域>.py` 并登记运行时注册表——启动日志
`HCU adapter runtime:` 行 + `HCU_ADAPTER_DUMP` JSON 可查每个算子当前实际谁在跑
（plugin 仓的后端同样要登记，含直调 lightop/aiter 的路径）。

状态语义：

| 状态 | 含义 |
|---|---|
| production | 生产默认，全套门禁通过 |
| validated | 门禁全过，opt-in（或按 wheel 自动规则放行） |
| candidate | 已接线，门禁矩阵待跑 |
| planned | 仅入目录（UI 可见置灰），选择即报错 |

当前目录（2026-10-12 实拍三家轮盘后的交集）：

| 算子 | 可选后端 | 说明 |
|---|---|---|
| `silu_and_mul` | triton（默认）/ lightop / torch | aiter 是融合变体（silu+quant），非本算子后端，另行入目录 |
| `rms_norm` | triton（默认）/ aiter / torch | 仅 plain rmsnorm；fused_add/gemma 恒 triton/torch |
| `allreduce` | aiter（默认）/ nccl | 复合键：aiter=aiter.dist GraphSafe（新镜像压测 4/8-rank 全绿）；nccl=回退；custom 自制 AR 已于 2026-10-19 除役删除（R-01 定罪后 aiter 顶上，修复无意义） |
| `paged_mqa_logits` | triton（默认，R-13 #4）/ emu（opt-out）/ vendor（broken） | DSA 线 |
| `rotary_embedding` | triton（在用，无开关）；aiter / lightop（planned） | rope_fwd / rotary_embedding_fuse 家族待接 |
| `moe_gate` | triton（在用，contract=torch）；aiter / lightop（planned） | topk_softmax/sigmoid/moe_fused_gate |
| `fused_moe` | triton（默认回退）/ aiter（validated，auto 先试） | w8a8 专家层：aiter 块量化走 ASM 调优表、通道量化走 moe_c；未调优形状自动回 triton |
| `w4a16_moe` | triton（auto 现答）/ deepgemm（candidate，显式 opt-in） | W4A16 专家层（moe_wna16 AWQ/GPTQ-int4 线）：deepgemm=R-05 桥（DTK 轮 masked w4a16 C++ 分组 GEMM+自管 dispatch/silu/combine，eager-only v1；对称无零点+g32/64/128+对齐守卫） |
| `per_token_quant` | torch（在用）；aiter / lightop（planned） | per_token_quant_hip / lightop 量化族 |

## 与环境变量的关系（迁移口径）

`docs/env-vars.md` 仍列全部变量，但从本文件起定位为**调试覆盖手册**。新功能一律
只加配置项，不加环境变量；既有环境变量行为不变（仍高于配置文件），便于已部署环境
平滑过渡。

## Web 配置台

配置文件 schema 即 UI 的数据契约：`version: 1` + `ops` 平面映射。浏览器直开
`tools/hcu_config_ui.html`（自包含单文件，无需服务端）可视化完成三件事：

1. **算子后端选择**：按 BACKEND_CATALOG 快照渲染下拉（planned 置灰），只导出
   非默认项成 `hcu_adapter.yaml`（复制/下载）；
2. **cookbook 启动命令生成**：按并行方案（TP/EP/DP/CP/PD/MTP/AWQ…）套用
   [cookbook.md](cookbook.md) 已验证配方生成完整启动命令与警示项；
3. **每卡权重显存评估**：读模型 config.json（或预设），按并行切分规则估算
   每卡权重占用并判定可行性。

CLI 向导（`tools/hcu_config_wizard.py`）是其命令行形态；`--autotune` 变体=目标机自动探测+实测选档（平台画像→逐算子三门→阶梯降档带理由，见 [autotune.md](autotune.md)）。注意 UI 内嵌的
目录/配方是代码快照，BACKEND_CATALOG 或 cookbook 变更后需同步（纯函数段
在文件内 `/*==PURE-BEGIN*/ ... /*==PURE-END*/` 标记之间）。
