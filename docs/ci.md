# CI 分层设计（MR 快速自动测试）

目标：**别人提 MR → 分钟级自动反馈**。约束：数值/内核测试必须跑在真实 HCU
（gfx938 + DTK 栈）上，而 Gitee Go 云端 runner 只有 CPU；GPU 在共享节点的
容器里（CI 用自己的容器，不影响他人），不能被 CI 无脑抢占。因此分三层，
层间用不同触发面：

| 层 | 内容 | 环境 | 耗时 | 触发 |
|---|---|---|---|---|
| **L0 结构门禁** | `ci/l0_gate.py`：全仓 py 语法、.sh LF-only（CRLF 曾静默毁 bash）、密钥扫描、REGISTRY↔docs/env-vars.md↔BACKEND_CATALOG 交叉一致、sitecustomize 引用完整性、example yaml 键合法性、治理三规则（`glue-numerics`=sitecustomize 无 triton 数学 / `text-patch`=源码补丁必在 TEXT_PATCH_ALLOWLIST / `registry-ops`=register 的 op 必须 catalog）；软警告=裸读 os.environ/CRLF py | 任何机器，零依赖 | <5s | 每次 push / MR（Gitee Go 云端，`.workflow/l0-gate.yml`） |
| **L1 GPU 数值门禁** | `ci/l1_gpu_gate.sh`：L0 + pytest（test_ops_cpu / kernels / adapter）+ 精选 tools 门禁（config 目录、fa3-emu CP 形状、rope 三路一致、aiter-MoE 分发、deepgemm contiguous triton）+ vendor wheel 探针（WARN 级） | HCU 容器，1 卡，自动挑空闲（VRAM<5%） | 5-15min | MR watcher（`ci/mr_watcher.py` 轮询开放 MR，新 head sha 即测） |
| **L1.5 上游单测精选（已落地 l15_upstream 段）** | sglang 官方 `test/registered/attention/unittests` 的 dsa+kda+dense+mla 四目录（`-k "not tilelang"`）：**88 绿/2 排除/101 合法 skip（fa4/flashinfer/trtllm 平台性）/208 子用例，暖缓存 ~1min、冷缓存 ~35min**。首役战果（2026-10-17）：五修打通——shfl `__shfl_*_sync`→非 sync（dpsk_v4_topk 编译复活）、`get_mla_metadata` 新 API、`flash_mla_with_kvcache` 输出秩随输入、flashinfer `_grouped_size` ABSENT+deny 组合、`attn_backend_gate`（fa4/flashinfer 构造期 ImportError→kit 优雅 skip）。mla 加餐（2026-10-18）：首跑 8 失败同根——flashmla 后端 unified 模式的 extend 路径整体委托 flashinfer 父类（`forward_extend`→`super()`），flashinfer 桩 `__getattr__` 记 FAIL；修法=桩内 `_hcu_stub_raise`（unittest 栈帧→SkipTest，serving 栈保持 NotImplementedError），8 失败→25 合法 skip、26 绿全保（flash_mla 移植面获正面检验）。**判定：MTP/fp8-KV/DSA 内核面全绿 → serving 层病在 srt 状态机**。**环境铁律**：容器两份 sglang（pip dist-packages vs `/sglang-v0521` 源码树，serving 与 JIT include 均指后者）；pytest 必须显式 `PYTHONPATH=/sglang-v0521/python:$ROOT`（缺一静默测错树/adapter 不加载）+ `-e HCU_DEEPGEMM_BRIDGE=1`；JIT 内核首编 ~5min/颗（ninja 编译非挂死）。tilelang 两用例=DTK 0.1.12+dtk2604 warp 划分 bug（threads=384→6 warp 不整除，编译期 InternalError），平台已知红待 DTK 修 | HCU 容器，1 卡 | 暖 ~2min / 冷 ~35min | 与 L1 同车（`ci/l1_gpu_gate.sh` l15_upstream 段自动发现上游树，无树则 SKIP） |
| **L1.5b 上游机制目录（l15_mech 段，R-14 批 A）** | unittests 六机制目录 gdn/hybrid_linear/mamba/swa/lightning/dsv4（旧树无目录自动 SKIP），env 契约同 l15_upstream（`PYTHONPATH=/sglang-v0521/python:$ROOT` + `HCU_DEEPGEMM_BRIDGE=1` + `-k "not tilelang"`），另加 `-rs` 列 skip 理由。**known-fail 预算绊线**：段判绿按计数不按 pytest rc=1——`passed≥90 且 failed≤42`（基线 38/99 的 ±10%）才 PASS，越线/超时(124)即红进 FAILS，防"登记失败"膨胀成"回归自由"；账本见下节。首跑 826s（dsv4 JIT 败三连不缓存，每跑重付编译尝试） | HCU 容器，1 卡 | 暖 ~5-14min | 与 L1 同车（`ci/l1_gpu_gate.sh`） |
| **L2 e2e** | 0.6B 四判定（launch/生成/锚点/长文）+ 多卡项（AR/EP/PD）= `tests/tools/run_model_suite.sh` / cookbook 配方 | HCU 容器，1-2 卡 | ~1h | 夜间 / 发版 / 手动（勿挂 MR，太慢太占卡） |

### L1.5b KNOWN_FAILURES 账本（l15_mech 六机制目录，2026-10-19 首跑分诊）

基线（容器 lzg-g53-tp8，新镜像 sf_b075-20261018-0048，/tmp/l1_mech_first.log）：
`38 failed / 99 passed / 15 skipped / 173 subtests passed in 826.58s`。失败共六族
逐族登记零静默；族级件数按 `--tb=line` 错误行聚合（与 pytest 总数 38 有 ±1 的
短汇总/栈重复计差，不影响归因）：

| # | 族 / 件数 | 症状（错误行在案 /tmp/l1_mech_first.log） | 归因 | 去向 |
|---|---|---|---|---|
| K1 | gdn test_gdn_cutedsl_ring_verify ×2（test_wide_vec_batch / test_ilp4_small_batch） | cuteDSL 编译错：`llvm.call op operand type mismatch: 'ptr<1>' != 'ptr'`（cutedsl_gdn_mtp_ring.py:2474，确定性、非数值） | **平台工具链**：cuteDSL→LLVM lowering 给 operand 0 加地址空间限定与被调签名失配（DTK cute-dsl 面，与 conv1d-flake 同类 codegen 嫌疑） | 随 cute-dsl/DTK 升级重跑；不修上游 |
| K2 | dsv4 JIT ×3 | `Failed to build JIT module sgl_kernel_jit_dpsk_v4_metadata`（ninja + `retry() exceed` ×3） | **平台**：DPSK v4 metadata JIT 源对 gfx938 编译栈不兼容 | 随 DTK/上游 JIT 源更新重跑 |
| K3 | dsv4 subtest ×1 | `ValueError: Memory attributes {'l1c_evict_priority'} not supported for generic -> rmem copies` | **平台**：本机 triton/ROCm 无该拷贝专用 op | 随 triton 轮更新重跑 |
| K4 | dsv4 页表形态 ×10 | `AttributeError: 'NoneType' has no attribute 'shape'` @ adapter `flash_mla.py:125`（`block_table=None` 未覆盖） | **桥失配**：emu flash_mla 只收页表形态，None 表（连续 KV）路径缺 | 候补 R-11 flash_mla/dsa_paged_mqa 面承接清单 |
| K5 | dsv4 数值 ~20 | `Tensor-likes are not close!`（dsv4_attention.py:910 ×9 / :1689 ×1） | **数值面待深查**：dsv4 native backend 或受 K2 连带走兜底后数值可疑 | 深查另立；不设晋级门 |
| K6 | mamba2 triton ×3 | decode 页界/bsz1-prefix 数值失配（backend='triton'：mamba2_decode_page_boundary / bsz1_nonzero_prefix / runner_cuda_graph 页界） | **自研 triton 档已知面**：页界形状敏感性，与 conv1d-flake 同族嫌疑未证 | R-11 / conv1d-flake 桩跟踪面 |

15 skipped 全数合理（第 2 跑 `-rs` 实列）：gdn/swa 的 test_flashinfer.py ×14
= `CUDA + flashinfer are required` 平台性 skip + 1 件 flashinfer linear GDN
SM90/SM100 门——与 L1.5 的合法 skip 同类，零非预期。

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
   # 仓库位置是部署配置不是代码（迁仓只改这里）：
   mkdir -p ~/hcu_ci && echo -n <owner>/<repo> > ~/hcu_ci/repo_slug
   #   （或 export HCU_CI_REPO=<owner>/<repo>）
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
桥/plugin 治理规则（glue-numerics / text-patch / registry-ops）的语义见
[governance.md](governance.md)；运行时归因用 `HCU_ADAPTER_DUMP`（JSON：
每算子活跃实现+机制+env 生效值），集成问题的配置级二分以它为起点。

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
