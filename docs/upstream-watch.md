# 上游追新巡检规约（upstream-watch）

> 本文回答：**如何用一套可交接的流程，让 adapter 在上游 sglang 每次发版前
> 就知道"哪些桥会被打破"，而不是等模型 e2e 失败后倒查。**
> 定位：**规则文档**——流程、判据、授权边界全部写死，**执行者默认为
> agent**（定时任务挂载的巡检 agent 或任意接手的 AI 会话），人只出现在
> 两端：制定规则、消费报告。不依赖任何个人信息或特定会话。
> 启用时机：兼容性对账工具（ci/upstream_diff.py，#349）+ 测试接入层
> （#350）落地并稳定运行一版后，由 agent 按本规约执行。

## 0. 目标与预算

- **目标**：上游 main / 新 release 的每一处变更，在影响 adapter 前 24h
  内被枚举成清单；release 落地当天即有"破损桥名单+修复方案候选"。
- **预算**：<1 小时/天（三层：静态对账为主，import smoke 次之，行为
  探针/测试增量按需）；只在有空闲 GPU（VRAM<5%）时跑验证层。
- **原则**：报告优先，自主修复为辅；一切动作可追溯（日志+commit+任务
  回填），有疑义宁可不动。

## 1. 每日流程（凌晨低峰期执行，agent 后台完成不唤醒任何人）

1. **上游变更摘要**：拉 `sgl-project/sglang` main 过去 24h 提交，按
   adapter 接触面分区摘要：`server_args` / `arg_groups` /
   `layers/attention*` / `layers/quantization*` / `platforms 判定` /
   `moe runner` / `model registry`。无相关变更则当日报告两行结束。
2. **release 动态**：检查新 release / RC tag。出现即触发 §4 全量流程。
3. **兼容性对账**（工具就位后）：跑 `ci/upstream_diff.py <上游树>`——
   以 runtime registry + `TEXT_PATCH_ALLOWLIST` 为对账表，逐项核 target
   符号存在性/签名、needle 原文、判定词 seam 源码指纹。输出=失配清单
   （**这份清单就是变更面/攻击点**），exit code 可接门禁。
4. **增量验证**（条件执行，两层各一）：
   - import smoke：PYTHONPATH 组合下 `import sglang` + sitecustomize
     静默，捕获 ImportError/AttributeError/TypeError；
   - 上游测试接入层（#350）：registered PR-suite 精选 e2e 薄壳 +
     L1 `upstream-contracts` 段，均只在有空闲卡时跑，单任务 timeout
     锁死。
5. **报告落盘**：≤20 行（变更要点/风险预警（失败点定位到桥名）/
   自行修复记录/待人工决策项），写入 `upstream_watch/log-YYYYMMDD.md`
   （gitignore 根目录目录，不进版本库）。

## 2. 自主修复授权边界（巡检 agent 的行动准则）

允许直接修复并留痕的**全部条件**（缺一不可）：

- 有空闲资源：有空闲 GPU、容器可用（只允许动**自己负责名单内**的容器
  与进程——部署共享节点时每人以名字缩写为前缀，绝不触碰他人服务）；
- 失败点明确：对账/报错给出具体桥名与错误栈，属于"已知模式的破损"
  （符号改名/签名漂移/needle 失配/断言收紧），不是新症状；
- 修复路径走既定流程：`hcu_adapter/<域>.py` 修桥 → 对应单测补离线复现
  （governance §7.3 "one pit, one test"）→ L1 绿 → commit →
  对应子任务/工单回填结论（工具/流程由团队现用任务系统承载，规约不绑定具体系统）。

**必须只写报告、留给白班协调的**：

- 所有卡全忙（无条件跑验证，自然也就无法确认修复有效）；
- 变更属于**新的破损模式**（首轮排查到不了单一桥名的）；
- 需要协调他人的事：卡窗口申请、模型权重就位、疑似上游 bug 需要向上
  游提 issue/PR 的（走 operator-requests / governance §5.1 流程）；
- 任何拿不准的时刻——报告说明即可，次日人工接手。

## 3. 节拍与责任

- 日常节拍：每日一次（建议 01:00 前后共享节点空闲期），由**巡检
  agent**（挂载本规约的定时任务）执行；维护内容变更（工具换代/流程
  修订）由团队改本文档即可，规则不因执行者变化。
- release 日加跑：0.5.x 正式版落地当天，先跑对账与 import smoke 出
  面清单，模型 e2e 按 models.md 矩阵**只测清单点名区域**，人工锚点+
  上游薄壳双判定；结论当日报群。**分支动作（R-17 双线模型）**：上游
  出新 release / RC→正式分支当天，从 main 建 `rel/v0.5.x` 线即推主仓
  （拓扑约束：主仓只留 main+rel 发版分支，个人调试在 fork 内）；rel 线适配期
  三绑定（分支↔`/sglang-v05XX` 树↔容器），HTTP 普查与门禁一律带
  `--up-tree` 指树；v0.5.22 起的前一版本线降「仅 bugfix」（N=2 窗口），
  降档/EOL 在本文 §5 末尾追加一行登记。
- 交接：本规约 + `ci/upstream_diff.py` + `upstream_watch/` 日志目录
  即全部家当；重启巡检 = 新建 agent 定时任务并在提示词中引用本文档
  路径，agent 读完本文与 docs/ci.md、docs/bootstrap.md 即可接手。

## 4. 工具清单（就位状态）

| 组件 | 状态 | 说明 |
|---|---|---|
| `ci/upstream_diff.py` | 静态首版 | 比较两个明确 commit 的注册 literal target、签名/函数体、白名单 needle 与 L1.5 nodeid；dynamic/re-export 等未解析面单列 baseline gaps，不冒称全覆盖 |
| import smoke | 待建（#349 内） | 见 ci.md L1 前半段的独立化 |
| 上游单元层挂 L1 | 待建（#350） | `upstream-contracts` segment |
| registered e2e 薄壳 | 待建（#350） | 精选 ~20 个,模型路径/端口/旗标注入 |
| 巡检执行者 | 未挂载 | 规则就绪；届时由定时任务挂载本规约（agent 执行） |

> 维护规则:本文件随工具有效性修订——对账工具出现连环误报/漏报时,先修
> 工具或在本文件登记已知盲区,不允许让巡检带着坏对账表跑。

### 手动升级验证入口

```bash
HCU_ADAPTER_HOST=none python3 ci/upstream_diff.py /path/to/sglang \
  --baseline-ref <old-commit> --candidate-ref <new-commit>

HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 HCU_ADAPTER_HOST=none \
  python3 -m tools.l15_upgrade_check --up-tree /path/to/sglang \
  --baseline-ref <old-commit> --candidate-ref <new-commit> \
  --checkpoint /path/to/local-real-checkpoint
```

静态对账读 Git commit 原文，不读 adapter 启动后已改写的工作树。结果另外记录
工作树 HEAD/dirty；验证树 HEAD 必须与 candidate 一致。工具只比较声明契约，
不能保证动态注册/导入重导出或整个仓库的语义兼容。函数体变化也会要求审查，
不会只核签名后默认兼容。相同 commit 可用于验证零漂移基线，
报告明确 `same_revision=true`，不代表完成了一次升级。

串联入口顺序执行静态对账、五家族严格组件门、五类 dummy 小模型与真实
checkpoint 的 eager/graph 冒烟。任何阶段失败停止；每次删除旧阶段产物，
拒绝错误 artifact 或其他 upstream commit 的绿色报告。只有全部实际通过才
写 `ready=true`。`--audit-only` 只生成静态报告，始终保留 `ready=false`。
版本升级后应审查并更新矩阵及其绑定 manifest，而不是跳过指纹检查。
静态漂移经审查且适配修复后，可用 `--reviewed-drift <已审查的对账JSON>`
显式接受该份差异；工具核对 baseline、candidate 和完整 changes 精确一致，
拒绝旧版本或其他变更的报告，并在 verdict 记录接受来源。该选项仍强制执行
全部组件和模型验证，不跳过源码绑定。默认不接受任何静态漂移。
该入口是手动工作流，尚未配置定时巡检。

## 5. 轮子问题登记（镜像/DTK 轮升级踩雷实录，升级镜像时先过这里）

| 日期 | 轮子 | 症状 | 定裁 | 状态 |
|---|---|---|---|---|
| 2026-10-18 | **tilelang 0.1.15+dtk2604**（sf_b075-20261018-0048 镜像内） | mhc_pre_big_fuse_with_norm tilelang 融合内核（sglang kernels 树 mHC）JIT 编译在 `ThreadSync("shared.dyn")` pass 硬错：`Cannot lower a required shared-memory sync inside the else branch...not warp-uniform. Condition: not tx < 32`；**0.1.12 同内核同条件仅 Warning hoisting**（thread_storage_sync.cc:949）——pass 把可安全提升的模式改成了硬错。触发面：DTK 是 CUDA 语义平台（is_hip 恒 False），上游 `SGLANG_OPT_USE_TILELANG_MHC_PRE(默认True) and not is_hip()` 门恒开，glm5_next/GLM-5.3 系 mHC 模型 decode graph 捕获必踩 | 临时：镜像内钉 0.1.12（vendor_tilelang_0112.tgz，bootstrap §5）；**不走 SGLANG_OPT_USE_TILELANG_MHC_*=0 回退**（torch 链性能倒退，用户定规：跟版本升走） | 待报轮子团队；**tilelang ≥0.1.16 升级验收配方（~40 分钟）**：①换包（pip install 新轮或从新镜像 docker cp，删旧 tilelang/+dist-info）②glm53f TP4 launch（cookbook §1.5）——boot ~8min 到 decode graph 捕获，坏轮必在此报 InternalError/ThreadSync（0.1.15 实证）③四门 probe 电池+单流 tok/s 对 7 基线 ④绿则更新 bootstrap §5 pin 版本+本行状态。离线 2 分钟探针已试废（mhc 需全 runtime，单进程 core dump，tmp_fetch/probe_tilelang_mhc.py 存档） |
| 2026-10-18 | tvm_ffi optional addon | 冷容器 import 链自递归 fork（87-122 层，每层 ~22s）——build 脚本 import 期再触发 optional 检查 | bootstrap §3 预热/直拷配方在案（TVM_FFI_DISABLE_TORCH_C_DLPACK=1 直构，或健康容器拷 .so） | 已有既裁决決，登记备查 |
