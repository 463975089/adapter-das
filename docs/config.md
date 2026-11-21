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
  `infra/` 模块迁移不改变这个部署位置；从其他目录启动宿主也使用同一规则。
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

`hcu_adapter/infra/hcu_config.py::BACKEND_CATALOG` 是唯一事实源：算子 → 可选
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
| `minimax_h3_rotary_embedding` | triton / torch（candidate） | H3 DiT128/96及VAE64/48，详见 [H3候选](minimax-h3.md)；BF16分量通过，模型待验证 |
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

MiniMax PCP8 prefill 的 Plugin 新能力通过 `ops.vllm_minimax_prefill` 管理：
默认 `triton`，显式 `stock` 为未融合对照，均保留 candidate 状态。
Adapter 只提供配置/catalog/快关，数学实现位于独立 `vllm-plugin-das`。

## SGLang sparse MLA 候选后端

`ops.sglang_mla_sparse: dtk` 显式选择 SGLang 专属 BF16 sparse FlashMLA 实现；默认 `emu`，没有新增环境变量。当前原生范围为 BF16、H1–64、D512、top-k 2048–2112（掩码补齐到 64 的倍数）、无 attention sink，其他形状保留兼容路径。实现独立于 vLLM/DeepInfer。候选仅完成小规模数值和变输入 Graph 探针，整模型与性能验收尚未完成。

### MLA HiCache 地址表传输

`hicache_mla_transfer` 当前使用 candidate Triton 实现：按字节处理 device pointer table、pinned host buffer、layer-first/page-first 布局，不将地址值当作缓存数据索引。GPU 路径不读取地址表到 CPU。2026-11-01 定向诊断覆盖 528/8448/33792/65537 字节行、1/7/65 行、三层、双向拷贝与三次变输入 Graph 回放，逐字节一致；完整模型的驱逐回载与性能仍待验收。

### HY3 prefill Graph 兼容开关

`HCU_DISABLE_OPS=prefill_graph_driver` 在新进程中跳过流状态查询适配，保留安装前的绑定。默认启用适配；只改变驱动查询，保留 prefill Graph，不改变硬件身份。

### NEXTN 空闲批次字段开关

`HCU_DISABLE_OPS=nextn_idle_metadata` 在新进程中跳过缺失字段补齐。默认只为没有 `kv_indptr` 的类型提供 `None`，不改已有字段、实例值或实际 attention 后端。注册失败时不发布字段。两个停用项可用逗号合并。

### FA E5M2 显式选择

`HCU_FA_IMPL=vendor` 才安装 HCU E5M2 启动检查适配；auto/triton/emu、非 HIP 和未支持架构保留当前检查。支持范围为 gfx928/gfx936/gfx938，真机证据目前仅 gfx936。每个实际消费者先登记再替换，重复安装不叠加。与 Plugin 的分页布局配套运行时，在外层部署环境显式设置 `SGLANG_KV_LAYOUT_HCU_FA=1`，不改用户原启动脚本，也不扩大布局默认值。
