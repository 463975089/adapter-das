---
name: hcu-adapter-dev
description: hcu-sglang-adapter 仓库开发规约：算子垫片/影子库新增、三档实现策略、自动化测试要求、高危算子快关与精度定位流程、sglang cookbook (NVIDIA 参数) 直接支持的降级矩阵。改这个仓库的任何代码前先读本 skill。
---

# hcu-sglang-adapter 开发规约

不修改 sglang 官方代码，让社区版 sglang 直接跑在海光 HCU（HCU/DTK/HIP）上。
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

9. **测试先行顺序**：算子/数学类新实现——单测先于 e2e（ref 对拍+形状矩阵
   +容差，脚手架见 references/test-suite.md 三段式）；seam/接缝类——e2e 探路
   可以，但 "one pit, one test"：探明的坑修复时必带离线复现单测，否则不予
   合入。实证对照：contig-triton/fa3-emu 单测先行 e2e 首过 vs
   moe_align/slimquant 反序付 e2e 倒查代价（governance §7 维护闭环）。

10. **tests/ 下禁止脚本级副作用**：门禁会 pytest 收集 `tests/kernels`
   与 `tests/adapter`——模块级 `sys.exit`/裸执行会让收集期直接
   INTERNALERROR（两周内两撞，ae0698b 后仍有回潮）。脚本式测试必须
   函数化 + `__main__` 守卫；新写测试一律 pytest 兼容形态。

11. **性能 bench 三分离**：跑 bench 前 GPU 空闲检查、**预热段强制**
    （冷编译曾把 triton bench 测成 0.0×，热跑实为 1.9×）、结果带环境戳
    （卡型/驱动）。断言用**相对阈值**（臂间比/历史比），不吃绝对
    tok/s——绝对值锚在别人的机器上。性能判别与晋级依据 catalog 档位
    证据门，不做无档案的"感觉变快了"。

12. **模块命名随档位走**（governance §3b，L0 `impl-naming` 执法）：具体
    实现文件 = `<op>[_<变体>]_<tier>`——tier ∈ emu/triton/cpp/fork/ref
    或 vendor 库名（aiter/lightop/dtk）；路由器/机制/门禁裸名；平台词
    （hip/cuda/rocm）不是档位；`_ref` 只准被 tests/ 引用；同族实现共享
    op 词根（mqa_logits_{emu,triton}/mqa_vendor 一眼看全家族）。加档位
    后跑 `python tools/gen_tier_matrix.py`，`docs/tier-matrix.md` 一眼
    看全"谁有哪些档、缺哪档"（L0 freshness 钉同步）。

13. **vLLM 专项后端保留官方语义**：普通 MHA/GQA 可在已验证条件下默认
    `TRITON_ATTN`，但 `use_mla` / `use_sparse` 等专项配置必须保留 `None`
    交给官方 `RocmPlatform` priority，禁止强改成普通 attention backend；
    Plugin 的普通 FlashAttention 优化也不得接管 MLA/Sparse MLA。禁止把
    `gfx936` 等 HCU 架构伪装成 CDNA3 来让单个 AITER 检查通过——平台代际会
    同时放开 FP8/GEMM/MHA/MLA 等不相关能力。只在最窄 seam 表达真实能力，
    测试至少覆盖普通、MLA、Sparse MLA 路由以及安装态 wheel 的实际 backend 日志。

## 任务路由（做某类事之前先读对应参考文档）

> 全仓文档地图与失效信号清单见 `docs/reading-guide.md`（新会话先读它）。

| 你要做的事 | 先读 |
|---|---|
| 新增/修改一个算子垫片或影子库 | `references/shadow-matrix.md` |
| 加一座桥（改官方代码行为：换绑/断言放宽/判定词 wrap）或新能力选位 | `docs/governance.md`（双仓判别 + 裁决链 + registry 登记） |
| 写或改任何 kernel / 测试 | `references/test-suite.md` |
| 加一个快关（kill switch） | `references/kill-switches.md` |
| 排查精度问题（输出垃圾/分叉/漂移） | `references/precision-triage.md` |
| 端到端性能优化（开箱基线/profiler 归因/算子优化需求/回归） | `references/e2e-performance.md`（细节检索个人知识库 kb，手册内有 kb:<id> 引用） |
| 评估 sglang cookbook NVIDIA 参数能否直接用 | `references/cookbook-matrix.md` |
| 加/改一个环境变量 | `docs/env-vars.md`（手册+维护规则）与 `hcu_env.py` 注册表 |
| 改上游行为补丁（text patch） | `docs/governance.md` TEXT_PATCH_ALLOWLIST（L0 硬门） |
| 加/改算子档位实现文件（命名/归档） | `docs/governance.md` §3b + `python tools/gen_tier_matrix.py`（档位矩阵随 catalog 同 commit） |
| vLLM 0.32 平台、attention/MLA backend 选型 | `docs/frameworks/vllm032.md` + `docs/governance.md`（先区分普通/MLA/Sparse MLA，保持真实架构能力） |
| 需求/任务领取 | 任务跟踪不走仓内看板（board.md 已移除），归 MR 与团队现用任务系统；协作红线见 `docs/governance.md` §7 |

## 治理闭环（改代码必走的登记链，L0 机器查）

一处行为落地，多处登记同步——**同 commit 内完成**，否则 L0 红：

1. **catalog**：BACKEND_CATALOG 加/改条目（状态机 planned/candidate/
   validated/production/broken，晋级必须有对拍+bench+A/B 证据）；
2. **env 手册**：新 env → `hcu_env.py::REGISTRY` + `docs/env-vars.md`
   行（手册在册不是可选动作）；
3. **UI 快照**：`tools/hcu_config_ui.html` 与 catalog 同 commit（快照是
   治理链最后一公里，曾吃过 a983af3 亏）；
4. **运行时注册**：行为接线用 `registry.register`（验收/排障以
   `HCU_ADAPTER_DUMP` 可见为准）。

平台语义走 is_cuda 路线的推理：judgement 层翻译、硬件层保持真（见
governance §0.5）；上游行为补丁（源码 text-patch）必须进
TEXT_PATCH_ALLOWLIST 白名单，白名单外不合法。

## 仓库结构

```
hcu_adapter/           唯一适配包：bootstrap.py 宿主探测 + common→host 分发
                       （sglang/vllm/deepinfer/none）；bridges/ = common 跨框架守卫 +
                       sglang 宿主桥顺序链（按 architecture.md §9 P1-P3 向包内
                       域桥拆分中）+ vllm 平台身份/版本门/ABI 桥，安装序列有
                       专属回归钉 tests/adapter/test_sitecustomize_sequence.py；
                       deepinfer 宿主桥与契约内核独立命名空间；包内其余为按契约
                       复用的跨宿主机制层与域桥：
                       aiter / flashinfer / flash_attn 填空+守卫 + vendor 桥
                       （quant/ar-norm/mqa/w4a16）+ fla_hcu 路由；
                       hcu_env.py REGISTRY + hcu_config.py BACKEND_CATALOG；
                       kill_switch.py HCU_DISABLE_OPS 快关注册表；
                       registry.py 运行时实现注册表（register/dump，治理 §4）
sgl_kernel/            同名影子包（sglang 的 AOT 默认内核路径；宿主 import 契约，
                       包名不可改）——elementwise/layernorm/rotary/moe/kvcacheio.py
                       triton/torch 实现；fa_varlen_triton.py 等顶层自研 triton 内核
sitecustomize.py       三行自动加载入口（governance §0 终态）；整链 hip 门控
tests/
  test_ops_cpu.py        CPU 契约面（任何机器）
  tests/kernels/         自有算子+vendor 桥契约（GPU；pytest 收集面）
  tests/adapter/         seam/编排层测试（platform_env/model_env/sitecustomize
                         序列；脚本式测试须函数化+__main__ 守卫，pytest 收集）
  tests/vendor/          高危 vendor 压测
  tests/tools/_*.py      排查探针（手动，路径可用 HCU_ADAPTER_ROOT/HCU_CKPT 覆盖）
  tests/tools/run_triage.py  精度问题四级快速定位入口
  tests/bench/           性能/并发评测（gsm8k_eval，非门禁）
ci/l1_gpu_gate.sh      一键 L1 全门（l0/cpu/kernels/adapter/l15 上游单测/
                       gates/vendor 分段；上游单测零拷贝——收集列表在此维护）
docs/governance.md     算子治理：双仓边界/裁决链/角色/注册表/catalog 状态机/
                       接入时序/共建边界/维护闭环（§7）
skills/hcu-adapter-dev/  本 skill
```

## 协作规约

- 自动化门禁 = `tests/test_ops_cpu.py` + `tests/kernels tests/adapter` +
  `tests/vendor/` + ci/l1_gpu_gate.sh 分段（pytest 可跑、断言带数值谓词）；
  `tests/tools/_*.py` 是排查工具（手动、可硬编码路径但须支持 env 覆盖）；
  `tests/tools/` 不进门禁。
- 改 kernel 必须附带或更新对应契约测试；涉及 vendor 高危面的同时更新压测
  与快关注册表，三件一起提交。
- 桥/plugin 选位与登记（先分仓、裁决链、registry.register、L0 三规则、
  四处登记同 commit）见上文治理闭环与 `docs/governance.md`（§0/§3/§7）
  ——sitecustomize 不新增实现体。
- push 前最低标准：`pytest tests/test_ops_cpu.py`（任何机器）+
  节点上 `pytest tests/kernels tests/adapter tests/vendor -k "not stress"` +
  `python tests/tools/run_triage.py --stage ops`（或直接 `bash
  ci/l1_gpu_gate.sh` 一键全门）。
- commit 信息带根因短句（参考 fbc540c / 1eb038d 风格），便于回溯 bug 类。
- 文档与代码同库同 PR：发现可复用的新坑后立即在修复所在 MR 同步更新本
  skill/对应参考文档，不等最终总结；一次性环境日志只留验证证据。README
  是指针型入口，静态资产表只活在 catalog/models.md（已立规约：README
  不维护清单、不留本地任务编号）。
