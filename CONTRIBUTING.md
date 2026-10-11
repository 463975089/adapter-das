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

### 3b. 加桥（`sitecustomize.py` / `sgl_kernel/_adapters/`）

1. **先写根因注释**：上游哪个文件哪个判定漏了什么、为什么桥是解法（现存每座桥
   都是范本）。没有根因注释的桥不予合入。
2. 优先用 `_base.py` 的机制通道，不动上游源码：
   - `extras`（allowlist 填空）——真包缺的符号我们补；
   - `overrides`——需要整体夺回某函数时；
   - `post_hooks`——包加载后 wrap 类方法 / 判定词；
   - 模型文件级判定词才用 sitecustomize 的 post-hook wrap（按名 import 的模块
     必须走这条，直接改属性不生效）。
3. 桥必须有开关（env / yaml），默认行为与上游一致或已验证更优。
4. 新桥文件归位 `_adapters/<域>.py`（如 `prefill_cp.py`、`slimquant.py`），
   不要再往 `sitecustomize.py` 单体里追加长桥（见 architecture.md §8 拆分计划）。

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
- 共享 GPU 节点上**只操作自己标识（lzg*）的容器与进程**；杀进程按端口精杀，
  不做宽匹配 `pkill -f`。长任务必须 `timeout N` 前缀。容器内改码/部署走脚本
  文件，不拼超长内联命令。
- 任何令牌（Gitee push token、CI token）**不落仓库、不进 commit**；临时写入
  `.git/config` 用完即还原。L0 密钥扫描是兜底不是借口。

## 6. 版本升级（跟随上游 sglang）

升级 = 对齐 import 面：官方 `sgl-kernel` python API diff 一遍 → 补齐新符号的
影子实现 → 跑 L1 + cookbook 冒烟。影子库机制的价值就在这：上游模型代码不碰，
只追算子层。上游大版本（如 v0.5.21→v0.6）建议开 `feat/upgrade-vXX` 分支，
先跑 `_probe_vendor_inventory.py` 类盘点再动手。
