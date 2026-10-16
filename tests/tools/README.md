# tests/tools — 门禁 / 探针 / 物料 / 调试

本目录是适配工作的物料库：启动配方、能力探针、CI 门禁脚本、一次性调试取证。
**不归 pytest 收集**（那是 `tests/test_ops_cpu.py` / `tests/kernels` /
`tests/vendor` 的职责），本目录文件用 `python tests/tools/_xxx.py` /
`bash tests/tools/_launch_xxx.sh` 直接跑。

## 命名语法（新文件必须遵守）

| 前缀 | 语义 | 生命周期与纪律 |
|---|---|---|
| `_test_` | **门禁**：可被 CI / 回归清单引用的判定脚本 | 长期维护。被 `ci/l1_gpu_gate.sh` GATES 引用的五个改动要跑 L1 验证 |
| `_probe_` | **探针**：环境/能力盘点、e2e 行为探测 | 随特性存活；结论应回写到 docs（models.md / cookbook.md） |
| `_launch_` | **物料**：生产启动配方（模型 + 旗标 + 端口固化） | 随配方存活；env 变化时更新 |
| `_dbg_` | **调试**：一次性取证脚本 | 用完可删；**禁止被其他脚本 import / 引用** |
| 无前缀 | **工具**：跨任务长期工具（`run_triage.py` 精度四级定位、`run_model_suite.sh` 模型套件） | 文档化于 skills/hcu-adapter-dev/SKILL.md |

每个新文件顶部必须有 docstring：干什么、怎么跑（含 HIP_VISIBLE_DEVICES /
PYTHONPATH 等前置）。历史文件未补齐的，顺手改到哪个补哪个。

## CI 引用关系

`ci/l1_gpu_gate.sh` 的 GATES 引用（MR 门禁，改这些要本地过一遍 L1）：

- `_test_hcu_config.py` — 配置裁决链（env > yaml > auto > default）
- `_test_fa3_emu_cp.py` — fa3 emu 的 CP 分块数值（zigzag 多窗口）
- `_test_rope_fallback_vs_fused.py` — rope 融合写 vs 纯 rope 三路对拍
- `_test_aiter_moe_gate.py` — aiter fused-MoE 三路线 vs torch 参照
- `_test_contig_triton.py` — deep_gemm contig 的 triton 重写块 GEMM

e2e 级（不在 L1，回归时手工跑）：`_test_hcu_ar2.py`（triton AR + 30197 锚点）、
`_test_cp_allgather.py`（torchrun 2 卡 CP all-gather stamp）、
`_test_triton_ar.py` / `_test_dtk_deepgemm.py`。

## 常用速查

- 精度问题四级定位：`python tests/tools/run_triage.py`（容器内）。
- 三源算子面盘点（triton 自研 / aiter / lightop 交集矩阵）：
  `python tests/tools/_probe_vendor_inventory.py`。
- 生产配方按模型名搜 `_launch_<model>*.sh`；配方里的端口/env 是实测拍板值，
  复用时先核 `docs/env-vars.md` 是否仍一致。
- vendored sglang-das 内核 A/B 对拍+微基准：`python tests/tools/_probe_fla_hcu_ab.py`
  （standalone 内核级证据；serving 门在 `tests/kernels/test_fla_hcu_fork_gfx938.py`
  + e2e 锚点，二者区别见 skills/hcu-adapter-dev/references/precision-triage.md）。
- 性能定性三件套：`_probe_moe_decode_bench.py`（E=288 stock triton MoE decode）、
  `_probe_mhc_prepost_bench.py`（unfuse 后 mHC 链耗时+dispatch 计数，standalone
  坑见文件头）、`_probe_graph_launch_bench.py`（裸图发射开销，换 DTK 后复测）。
- graph 提交路径诊断：`_probe_graph_submit_bench.py`（节点阶梯/多线程并行性/
  阻塞判别——2026-10-16 结论：纯提交 6.8µs/节点、进程内串行、跨进程独立）；
  NUMA 绑核配方 `_launch_g53f_tp4_pinned.sh`（实测无收益，留作复测模板）。
