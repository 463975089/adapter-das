---
name: hcu-adapter-dev
description: hcu-sglang-adapter 仓库开发规约：算子垫片/影子库新增、三档实现策略、自动化测试要求、高危算子快关与精度定位流程、sglang cookbook (NVIDIA 参数) 直接支持的降级矩阵。改这个仓库的任何代码前先读本 skill。
---

# hcu-sglang-adapter 开发规约

不修改 sglang 官方代码，让社区版 sglang 直接跑在海光 HCU（DCU/DTK/HIP）上。
本仓库的价值主张：**上游 import 什么算子名，我们就提供什么算子名，内部可
路由到 triton / vendor / torch 三档实现，并对全部出口 API 做统一质量验证**。
"adapter 测试全绿 ⇒ 官方 sglang 无大问题" 是本仓库的验收标准。

## 黄金法则（历史 bug 换来的，违反必踩坑）

1. **同名 ≠ 同语义**：每个垫片必须对拍参考实现（cos/rel 断言），不能只测
   "能跑不崩"。案例：rope GPT-J cache 恒为半分 [cos|sin] 布局（fbc540c）。
2. **eager 对 ≠ graph 对**：凡进 decode 链路的算子必须过 capture-replay
   变输入探针。案例：AiterCustomAllreduce 图内重放损坏（1eb038d）。
3. **GPU 路径禁主机同步**：`.item()`/`nonzero`/布尔掩码索引会毁掉 CUDA
   graph capture。
4. **高危 vendor 算子默认不信**：custom all-reduce、flash attention 这类
   "size 灵活、场景多" 的 vendor kernel，接入即配压测 + 快关。
5. **import 永不失败**：未覆盖算子在**调用时**抛带指引的 NotImplementedError。
6. **环境变量三原则**（客户反馈"变量太多"的对策）：一个手册
   （`docs/env-vars.md`）；默认自动适应（arch/wheel 探测，正确部署零
   变量）；显式设置优先且启动日志标注来源（`hcu_env.py` 注册表，
   禁止在行为接线处裸读 `os.environ`）。
7. **平台语义走 is_cuda 分支**（governance §0.5）：上游平台判定统一回答
   CUDA——新模型/新功能上游先落 CUDA 分支，追新是卡商关键路径；HCU 缺的
   NVIDIA 库符号按四级路由接（lightop → aiter → triton 先行 → 响亮 stub）；
   fp8 fnuz 等语义护栏白名单锁定不随翻。
8. **新容器先灭镜像包**：每次创建/新用容器，先把镜像自带的 sglang(das
   定制线)和 sgl_kernel/sglang_kernel 从 dist-packages 删干净再干活；
   残留会与 PYTHONPATH 的官方源/ adapter 混栈（症状：奇怪的
   ImportError 如 hcu_alloc_decode_kernel、module 解析到镜像路径）。
   清理命令见 references/cookbook-matrix.md。

## 任务路由（做某类事之前先读对应参考文档）

> 全仓文档地图与失效信号清单见 `docs/reading-guide.md`（新会话先读它）。

| 你要做的事 | 先读 |
|---|---|
| 新增/修改一个算子垫片或影子库 | `references/shadow-matrix.md` |
| 加一座桥（改官方代码行为：换绑/断言放宽/判定词 wrap）或新能力选位 | `docs/governance.md`（双仓判别 + 裁决链 + registry 登记） |
| 写或改任何 kernel / 测试 | `references/test-suite.md` |
| 加一个快关（kill switch） | `references/kill-switches.md` |
| 排查精度问题（输出垃圾/分叉/漂移） | `references/precision-triage.md` |
| 评估 sglang cookbook NVIDIA 参数能否直接用 | `references/cookbook-matrix.md` |
| 加/改一个环境变量 | `docs/env-vars.md`（手册+维护规则）与 `hcu_env.py` 注册表 |

## 仓库结构

```
sgl_kernel/            同名影子包（sglang 的 AOT 默认内核路径）
  _adapters/           元路径影子库：aiter / flashinfer / flash_attn 填空+守卫
  _adapters/kill_switch.py   HCU_DISABLE_OPS 快关注册表
  _adapters/registry.py      运行时实现注册表（register/dump，治理 §4）
  elementwise/layernorm/rotary/moe/kvcacheio.py   triton/torch 实现
sitecustomize.py       编排器：探测 + 调各域桥 install()（不新增实现体）
tests/
  test_ops_cpu.py + tests/kernels/        sgl_kernel 自有算子契约测试
  tests/vendor/{test_allreduce,test_flash_attn}.py  高危 vendor 压测
  tests/tools/_*.py     排查探针（手动，路径可用 HCU_ADAPTER_ROOT/HCU_CKPT 覆盖）
  tests/tools/run_triage.py  精度问题四级快速定位入口
  tests/bench/          性能/并发评测（gsm8k_eval，非门禁）
docs/governance.md     算子治理：双仓边界/裁决链/角色/注册表
skills/hcu-adapter-dev/  本 skill
```

## 协作规约

- 自动化门禁 = `tests/test_ops_cpu.py` + `tests/kernels/` + `tests/vendor/`
  （pytest 可跑、断言带数值谓词）；`tests/tools/_*.py` 是排查工具
  （手动、可硬编码路径但须支持 env 覆盖）；`tests/tools/` 不进门禁。
- 改 kernel 必须附带或更新对应契约测试；涉及 vendor 高危面的同时更新压测
  与快关注册表，三件一起提交。
- 桥/plugin 选位与登记（先分仓、裁决链、registry.register、L0 三规则）见
  `docs/governance.md` 与 CONTRIBUTING §3b——sitecustomize 不新增实现体。
- push 前最低标准：`pytest tests/test_ops_cpu.py`（任何机器）+
  节点上 `pytest tests/kernels tests/vendor -k "not stress"` +
  `python tests/tools/run_triage.py --stage ops`。
- commit 信息带根因短句（参考 fbc540c / 1eb038d 风格），便于回溯 bug 类。
- 文档与代码同库同 PR：改行为必须同步改本 skill 对应参考文档。
