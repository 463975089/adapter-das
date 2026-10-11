# CI 分层设计（MR 快速自动测试）

目标：**别人提 MR → 分钟级自动反馈**。约束：数值/内核测试必须跑在真实 HCU
（gfx938 + DTK 栈）上，而 Gitee Go 云端 runner 只有 CPU；GPU 在共享容器里
（nmz26 lzg_pr418_test），不能被 CI 无脑抢占。因此分三层，层间用不同触发面：

| 层 | 内容 | 环境 | 耗时 | 触发 |
|---|---|---|---|---|
| **L0 结构门禁** | `ci/l0_gate.py`：全仓 py 语法、.sh LF-only（CRLF 曾静默毁 bash）、密钥扫描、REGISTRY↔docs/env-vars.md↔BACKEND_CATALOG 交叉一致、sitecustomize 引用完整性、example yaml 键合法性；软警告=裸读 os.environ/CRLF py | 任何机器，零依赖 | <5s | 每次 push / MR（Gitee Go 云端，`.workflow/l0-gate.yml`） |
| **L1 GPU 数值门禁** | `ci/l1_gpu_gate.sh`：L0 + pytest（test_ops_cpu / kernels / adapter）+ 精选 tools 门禁（config 目录、fa3-emu CP 形状、rope 三路一致、aiter-MoE 分发、deepgemm contiguous triton）+ vendor wheel 探针（WARN 级） | HCU 容器，1 卡，自动挑空闲（VRAM<5%） | 5-15min | MR watcher（`ci/mr_watcher.py` 轮询开放 MR，新 head sha 即测） |
| **L2 e2e** | 0.6B 四判定（launch/生成/锚点/长文）+ 多卡项（AR/EP/PD）= `tests/tools/run_model_suite.sh` / cookbook 配方 | HCU 容器，1-2 卡 | ~1h | 夜间 / 发版 / 手动（勿挂 MR，太慢太占卡） |

## MR 流（怎么"快速自动测试"）

```
提 MR ──> Gitee Go 跑 L0（秒级，必跑）
      └─> 容器内 mr_watcher 轮询到新 head sha
            ├─ fetch 到独立 clone/worktree（~/hcu_ci，绝不碰在役的 /root/hcu_t2）
            ├─ ci/l1_gpu_gate.sh：无空闲卡 -> 状态=queued，下轮重试（评论"排队中"）
            └─ ✅/❌ + 日志尾部评论回 MR
```

### 启用步骤

1. **L0（云端，无密钥）**：Gitee 仓库设置里开通 Gitee Go，`.workflow/l0-gate.yml`
   即生效（schema 有出入时以控制台生成模板对齐 stages 块）。
2. **L1（容器侧）**：
   ```bash
   # 令牌只放这里（绝不进仓库/git 配置）：
   echo -n <gitee-token> > ~/.hcu_ci_token && chmod 600 ~/.hcu_ci_token
   # 前台试一轮（不回帖）：
   python3 ci/mr_watcher.py --once --no-comment
   # 常驻：
   nohup python3 ci/mr_watcher.py > ~/hcu_ci/watcher.log 2>&1 &
   ```
   GPU 礼让：l1 脚本只在有卡 VRAM<5% 时启动，否则排队重试——不会踩同事服务。

### 状态与排错

- watcher 状态文件 `~/hcu_ci/state.json`（PR# → sha/status/rc）；删掉即全部重测。
- L1 各段日志在容器 `/tmp/l1_<段名>.log`。
- 无令牌时 watcher 仍可对**公开仓库**匿名拉 MR 列表并测试，只是不能评论。

## 测试怎么归层（给提 MR 的人）

| 你的改动 | 必须过 | 放哪 |
|---|---|---|
| 纯 python/文档/配置 | L0 | 自动 |
| 内核/数值/桥接（sgl_kernel/**） | L0+L1 | kernels 单测进 `tests/kernels/`，门禁脚本进 `tests/tools/_test_*.py`（脚本须无服务依赖、退出码即结论） |
| 启动行为/模型精度 | L0+L1+L2 | probe 进 `tests/tools/`，配方进 cookbook |

新增 L1 门禁：把脚本加进 `ci/l1_gpu_gate.sh` 的 GATES 列表即入 CI。
新增环境变量：`hcu_env.py::REGISTRY` + `docs/env-vars.md` 行——**L0 会强制
这对成对出现**（漏文档直接 RED），catalog 引用不存在的变量同样 RED。

## 设计取舍备忘

- **L0 刻意零依赖**（不用 pytest/torch）：Gitee Go 免装环境、秒级出结果、
  笔记本也能跑 `python ci/l0_gate.py` 当 pre-push 钩子。
- **裸读 os.environ 是软警告不是硬门禁**：存量 27 处（多为 debug 旋钮），
  硬门禁会让存量全红；新增行为级读取请走 `hcu_env.resolve`，逐步清零。
- **vendor wheel 探针 WARN 级**：坏 wheel 是环境事实不是 MR 回归，不应挡合并。
- **watcher 不用 webhook**：容器无公网入口，轮询 API v5 是唯一通路；90s
  间隔对"分钟级反馈"足够。
- **令牌纪律**：`GITEE_TOKEN` 环境变量或 `~/.hcu_ci_token`（600），CI 的
  git fetch 用一次性 askpass 传内存，绝不落盘/进 remote 配置（仓库里出现
  32-hex 令牌会被 L0 secrets 检查打红）。
