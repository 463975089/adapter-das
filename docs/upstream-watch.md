# 上游追新巡检规约（upstream-watch）

> 本文回答：**如何用一套可交接的流程，让 adapter 在上游 sglang 每次发版前
> 就知道"哪些桥会被打破"，而不是等模型 e2e 失败后倒查。**
> 定位：**规则文档**——流程、判据、授权边界全部写死，任何被指定的人
> （工程师或夜班 AI）拿到本文即可执行，不依赖任何个人信息或特定会话。
> 启用时机：兼容性对账工具（ci/upstream_diff.py，#349）+ 上游测试接入层（#350）落地并稳定运行
> 一版后，指派负责人按本规约执行。

## 0. 目标与预算

- **目标**：上游 main / 新 release 的每一处变更，在影响 adapter 前 24h
  内被枚举成清单；release 落地当天即有"破损桥名单+修复方案候选"。
- **预算**：<1 小时/天（三层：静态对账为主，import smoke 次之，行为
  探针/测试增量按需）；只在有空闲 GPU（VRAM<5%）时跑验证层。
- **原则**：报告优先，自主修复为辅；一切动作可追溯（日志+commit+任务
  回填），有疑义宁可不动。

## 1. 每日流程（凌晨低峰执行，只出报告不唤醒人）

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

## 2. 自主修复授权边界（夜班执行者的行动准则）

允许直接修复并留痕的**全部条件**（缺一不可）：

- 有空闲资源：有空闲 GPU、容器可用（只允许动**自己负责名单内**的容器
  与进程——部署共享节点时每人以名字缩写为前缀，绝不触碰他人服务）；
- 失败点明确：对账/报错给出具体桥名与错误栈，属于"已知模式的破损"
  （符号改名/签名漂移/needle 失配/断言收紧），不是新症状；
- 修复路径走既定流程：`_adapters/<域>.py` 修桥 → 对应单测补离线复现
  （CONTRIBUTING §3.4b "one pit, one test"）→ L1 绿 → commit →
  对应子任务/工单回填结论（工具/流程由团队现用任务系统承载，规约不绑定具体系统）。

**必须只写报告、留给白班协调的**：

- 所有卡全忙（无条件跑验证，自然也就无法确认修复有效）；
- 变更属于**新的破损模式**（首轮排查到不了单一桥名的）；
- 需要协调他人的事：卡窗口申请、模型权重就位、疑似上游 bug 需要向上
  游提 issue/PR 的（走 operator-requests / governance §5.1 流程）；
- 任何拿不准的时刻——报告说明即可，次日人工接手。

## 3. 节拍与责任

- 日常节拍：每日一次（建议 01:00 前后共享节点空闲期），由指定的维护
  人（或其挂载的定时任务）执行；.roster 按团队排班更新此文档维护者
  名单即可，规则本身不因人变化。
- release 日加跑：0.5.x 正式版落地当天，先跑对账与 import smoke 出
  面清单，模型 e2e 按 models.md 矩阵**只测清单点名区域**，人工锚点+
  上游薄壳双判定；结论当日报群。
- 交接：本规约 + `ci/upstream_diff.py` + `upstream_watch/` 日志目录
  即全部家当；新维护人读完本文与 docs/ci.md、docs/bootstrap.md 即可
  接手。

## 4. 工具清单（就位状态）

| 组件 | 状态 | 说明 |
|---|---|---|
| `ci/upstream_diff.py` | 待建（#349） | registry 驱动的静态对账；验收=对基线零误报+人为演进符号能红 |
| import smoke | 待建（#349 内） | 见 ci.md L1 前半段的独立化 |
| 上游单元层挂 L1 | 待建（#350） | `upstream-contracts` segment |
| registered e2e 薄壳 | 待建（#350） | 精选 ~20 个,模型路径/端口/旗标注入 |
| 巡检执行者 | 未指派 | 规则就绪，等团队指派（人或班表自动化） |

> 维护规则:本文件随工具有效性修订——对账工具出现连环误报/漏报时,先修
> 工具或在本文件登记已知盲区,不允许让巡检带着坏对账表跑。
