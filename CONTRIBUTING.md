# 贡献指南（多人协作约定）

> 先读 [docs/architecture.md](docs/architecture.md)（分层与机制总览）与
> [skills/hcu-adapter-dev/SKILL.md](skills/hcu-adapter-dev/SKILL.md)（开发规约），
> 再读 [docs/ci.md](docs/ci.md)（MR 自动测试怎么跑）。本文只讲协作流程与红线。

## 1. 分支与 MR 模型

- `master` 为受保护主干：**不直接 push**，一律走 MR。
- 分支命名 `feat/<域>-<短名>` / `fix/<域>-<短名>`，域与 commit 前缀同词
  （如 `feat/cp-flashmla`、`fix/moe-int8-scale`）。
- MR 提交后自动过两层门禁（见 docs/ci.md）：
  - **L0 结构门禁**（Gitee Go 云端，秒级）：语法 / LF / 密钥扫描 / 注册表交叉一致；
  - **L1 GPU 数值门禁**（自托管 watcher，容器空闲卡，~10 分钟）：算子单测 +
    五个门禁脚本 + vendor 探针，结果以评论回帖到 MR。
  - L1 全绿（或仅 WARN）即可合入；改动 `sitecustomize.py` / `_base.py` /
    import 接管逻辑的 MR，另需一次 e2e 冒烟（cookbook 配方）再合。
- 本地提交前先自跑 L0：`python3 ci/l0_gate.py`（零依赖，秒级）。

## 2. Commit 风格

一行式 `域: 描述`，中文英文均可，描述讲"做了什么+为什么"，长背景进 body：

```
cp: prefill context-parallel bridge (HCU_PREFILL_CP=1) - platform gate + rope guard ...
moe: aiter bridge engagement/debug logging (HCU_AITER_MOE_DEBUG)
ci: tiered MR auto-test -- L0 + L1 + watcher
docs: architecture collaboration chapter + html diagram
```

常用域：`cp` `moe` `attn` `rope` `gemm` `quant` `ar`（allreduce）`pd`（分离部署）
`ep` `ci` `docs` `tests` `env` `config`。

## 3. 三类改动的 Checklist

### 3a. 加 / 改算子（`sgl_kernel/*.py`）

1. API 签名与官方 `sgl-kernel` **逐字对齐**（含 `out=`、占位参数、destination-passing）。
2. 按三档实现：triton 首档（fp32 累加、零主机同步）→ vendor 封装 → torch 兜底。
3. 单测三层归位：`tests/test_ops_cpu.py`（torch 兜底）、`tests/kernels/`（GPU +
   CUDA-graph capture 回归）、`tests/vendor/`（高危 vendor 压测 + 快关 gate）。
4. 走 op 级 env / yaml 开关的，同步登记：`hcu_env.py` REGISTRY +
   `docs/env-vars.md` + `hcu_adapter.yaml` 示例键 + `hcu_config.py`
   BACKEND_CATALOG——**L0 交叉一致检查会挡缺漏**。
5. 高危算子（allreduce / attention 类）接入即配：压测（尺寸×dtype×graph-replay
   变输入）+ `kill_switch` 快关。

0. **平台语义前置**（governance §0.5）：接入点的平台语义走 is_cuda 分支——
   上游新符号（含 flashinfer/trtllm 等专有库调用）按四级路由找归宿：
   lightop → aiter → **triton 先行**（先跑通）→ 响亮 stub；fp8 fnuz 等
   语义护栏白名单锁定不随翻。triton 先行产物在 vendor 就位后转 ref 档。

### 3a. 加 / 改算子（`sgl_kernel/*.py`）

1. API 签名与官方 `sgl-kernel` **逐字对齐**（含 `out=`、占位参数、destination-passing）。
2. 按三档实现：triton 首档（fp32 累加、零主机同步）→ vendor 封装 → torch 兜底。
3. 单测三层归位：`tests/test_ops_cpu.py`（torch 兜底）、`tests/kernels/`（GPU +
   CUDA-graph capture 回归）、`tests/vendor/`（高危 vendor 压测 + 快关 gate）。
4. 走 op 级 env / yaml 开关的，同步登记：`hcu_env.py` REGISTRY +
   `docs/env-vars.md` + `hcu_adapter.yaml` 示例键 + `hcu_config.py`
   BACKEND_CATALOG——**L0 交叉一致检查会挡缺漏**。
5. 高危算子（allreduce / attention 类）接入即配：压测（尺寸×dtype×graph-replay
   变输入）+ `kill_switch` 快关。

### 3b. 加桥 / 加 plugin（先分仓，再落位）

1. **先读 [docs/governance.md](docs/governance.md)**：双仓边界、sitecustomize
   收敛终态、裁决链（seam → rebind → text-patch 白名单）、active/ref 角色
   都在那里，本文是执行项。
2. **先分仓**：上游在 AMD/CUDA 上**有**而 HCU 上缺的行为 → adapter 仓的域桥；
   上游**也没有**的能力 → **独立 plugin 仓**（经上游 seam 或注册机制接入，
   可直调 lightop/aiter 等底层库），MR 不进本仓。注意 plugin 的目的同样是
   **性能优化**（算子融合、专项内核、新调度路径）——动机是性能不改变选仓，
   判据始终是"上游有没有这个行为"。plugin 有意上游化时（自研格式/新算子
   push 上游拿正式 seam），按 governance **§5.1 演进轨迹**走：接入 → 证据
   积累（格式契约提前固化为文档+契约单测）→ 上游 PR → 过渡桥退役。
3. **本仓落位**：行为还原实现体一律放 `_adapters/<域>.py`（install() 入口 +
   开关 + 安装成功 `registry.register(...)` 登记），sitecustomize 只加一行
   调用（或 `_base.py` 机制通道里的注册）——**不新增**实现体到
   sitecustomize，review 拒收。
4. **先写根因注释**：上游哪个文件哪个判定漏了什么、为什么桥是解法（现存每座
   域桥都是范本）。没有根因注释的桥不予合入。
5. 优先用 `_base.py` 的机制通道，不动上游源码：
   - `extras`（allowlist 填空）——真包缺的符号我们补；
   - `overrides`——需要整体夺回某函数时；
   - `post_hooks`——包加载后 wrap 类方法 / 判定词；
   - 模型文件级判定词才用 post-hook wrap（按名 import 的模块
     必须走这条，直接改属性不生效）。
6. 桥必须有开关（env / yaml），默认行为与上游一致或已验证更优。
7. **登记运行时注册表**：`registry.register(op, impl, target, mechanism)`——
   op 是 catalog 键，mechanism ∈ seam/rebind/text-patch（text-patch 针串需在
   `TEXT_PATCH_ALLOWLIST` 在册）。这一步换来启动日志"每算子一行谁在跑"
   和 `HCU_ADAPTER_DUMP` 机读 JSON；L0 `registry-ops`/`text-patch` 会挡漏登。
8. **域桥/编排器里都只准写 glue**（路由/签名适配/断言放宽），不准出现
   triton import、`@triton.jit`、`tl.*`——算子数学一律放 `sgl_kernel/`
   （governance §1 红线，L0 `glue-numerics`）。
9. ref（诊断）实现同样登记 `role="ref"`，但**永久留在离线测试路径**
   （`tests/kernels/`），任何桥不得把它 bind 进运行时。
10. 存量 sitecustomize 长桥的迁出按 architecture.md §9 P1-P3 节奏做，
    每次迁移容器 e2e 复验；不要在别的 MR 里顺手搬（迁移与改逻辑分离）。

## 3d. 推广边界（团队协作红线）

adapter 仓与独立 plugin 仓并存后，防止边界随时间模糊的五条（违反即 review 拒）：

0. **先分仓再写码**：上游没有的新能力（算子融合/专项内核/新后端/新机制，
   即便动机是性能优化）不进本仓，落 plugin 仓经 seam/注册接入或直调底层库；
   本仓只收行为还原（governance §0 判别问句）。
1. **选位不跳级**：行为还原需求按 governance §3 裁决链选实现手段，
   MR 描述里写明选位理由（"seam 不可行因为…"）。
2. **单 active 指针**：一个 catalog 算子同一时刻只有一个生产实现；想换实现
   = 改注册表指针 + 走门禁，不是再抄一份。
3. **登记即存在**：没有 catalog 条目、没有 register 记录的实现不存在
   ——排障只认 `HCU_ADAPTER_DUMP` 的输出（plugin 仓的接入同样要求登记）。
4. **组合归因不重跑矩阵**：集成问题先看 dump（谁在跑），再 yaml 翻单条目
   二分；禁止"全绿矩阵跑三遍"式排查（governance §4）。

### 3c. 加环境变量 / 配置键

三原则：**正确部署零环境变量**（默认值即最优解）；**显式设置永远最高优先**；
**启动日志打印每个生效值及来源**。落地 = REGISTRY 登记 + env-vars.md 手册行 +
yaml 示例键，缺一被 L0 挡。

## 4. tests/tools 命名语法

`tests/tools/` 是物料库不是垃圾抽屉，新文件必须带 docstring（用途 + 运行方式）：

| 前缀 | 语义 | 生命周期 |
|---|---|---|
| `_test_` | 门禁脚本（可被 CI / 回归引用） | 长期维护，改坏 CI 会红 |
| `_probe_` | 环境/能力/e2e 行为探针 | 随特性存活 |
| `_launch_` | 生产启动配方（模型+旗标固化） | 随配方存活 |
| `_dbg_` | 一次性调试取证 | 用完可删，禁止被引用 |
| 无前缀（`run_triage.py` 等） | 跨任务长期工具 | 文档化于 SKILL.md |

详见 [tests/tools/README.md](tests/tools/README.md)。

## 5. 共享节点与双树纪律（红线）

- **本地仓（编辑树）是唯一改码处**；HCU 容器内的仓只做运行验证，**禁止在容器里
  改代码再手拷回来**——两树漂移曾造成 API 对不上的实 bug（elementwise 事故）。
  容器树与 master 漂移时，以 Gitee master 为准：`git fetch && git reset origin/master`。
- 共享 GPU 节点上**只操作自己标识前缀的容器与进程**（每人用自己的名字
  缩写做前缀命名容器，如 `zhang3_*`、`li4_*`；只动带自己前缀的，别人的不碰）；
  杀进程按端口精杀，
  不做宽匹配 `pkill -f`。长任务必须 `timeout N` 前缀。容器内改码/部署走脚本
  文件，不拼超长内联命令。
- 任何令牌（Gitee push token、CI token）**不落仓库、不进 commit**；临时写入
  `.git/config` 用完即还原。L0 密钥扫描是兜底不是借口。

## 6. 版本升级（跟随上游 sglang）

升级 = 对齐 import 面：官方 `sgl-kernel` python API diff 一遍 → 补齐新符号的
影子实现 → 跑 L1 + cookbook 冒烟。影子库机制的价值就在这：上游模型代码不碰，
只追算子层。上游大版本（如 v0.5.21→v0.6）建议开 `feat/upgrade-vXX` 分支，
先跑 `_probe_vendor_inventory.py` 类盘点再动手。
