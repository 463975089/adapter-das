# GLM5.3 Flash Adapter 配套交接

## 当前配套整理：2026-11-18

当前 Plugin 的 gfx936 INT8 DeepEP 配方使用 `deepgemm==2.1.0+dtk2604.torch2110.2610301506.gedf118`
（Python 3.10 / DTK 26.04 / Torch 2.11），不是下方历史记录的 Python 3.12 镜像。
专家 masked/contiguous 权重布局、MODE=1001 与调用参数由 Plugin 管理；详细接口见
配套仓 `sglang-plugin-das/docs/configuration.md`。Adapter 不新增同名 INT8 GEMM 包装器。

`deep_gemm_dtk` 保留上游 `deep_gemm` 兼容出口与已有 BF16/FP8 档位选择；新增回归断言
四个 INT8 vendor 打包/GEMM 符号在 alias 和 raw provider 中保持同一对象，避免将
Plugin 专家计算混进 Adapter。DeepEP quant_type ABI 和 dense INT8 hipBLASLt 接入保持。

本次还为 rebase 后的 sampling/grouped-topk 模块补齐四处深层 import 的档位/替代说明，
消除 L0 的四项静态拒绝；这两处文件的 Python AST 与父提交相同，未改计算或路由。
两仓代码仍分别提交，官方 SGLang 源码不修改。本次容器定向回归 16 项测试、6 个子测试通过；修复旧 DeepEP 测试遗漏内部 helper 的装配，并覆盖 INT8 原生透传与上游参数翻译。L0 通过，保留 32 项既有警告。

最新同步基线为 HYGON-AI `master/main@1c9448e1b09514a8d1424ad0d50d70cb88a6d078`。沿用上游 AR+RMSNorm 在设备绑定、通信组初始化后安装的生命周期，删除两项旧即时安装/旧探测接口测试，覆盖交由上游 `test_ar_norm_install.py` 与 `test_hcu_env_ar_norm.py`。容器定向回归 103 项及 6 个子测试通过；真实 SGLang 宿主下 dense INT8/桥接 24 项通过，保留 eager/变输入 Graph 和舍入反例。L0 通过（30 项既有警告）。首次 HOSTnone 混跑的 6 项 Linear 失败属于桥未安装的测试环境，记录保留，不计为通过。未执行本轮数据集精度。

组合 Decode bs8 回归保存了 16 rank、每 rank 10 轮，整轮均值 79.501 ms（rebase 前 80.847 ms），未见整轮回退；单次 profile 不作为确定性加速证据。采集后发生既有 profiler SIGSEGV，按同一候选恢复。实验目录为 `$AGENT_DIR/das-glm53-rebase-20261116/perf/rebase-squash-20261118`。

下方为 2026-11-04 的历史证据，不代表当前环境或最新全集精度。

## 历史交接：2026-11-04

## 2026-11-10 Fork 交付与复现入口

| 组件 | 分支 | 交付时的提交 |
|---|---|---|
| [Plugin fork](https://github.com/Alone-wl/sglang-plugin-das/tree/feat/glm5.3-flash) | `feat/glm5.3-flash` | `fd789c1c42296d3726c9e916cdd3e90b79524944` |
| [Adapter fork](https://github.com/Alone-wl/adapter-das/tree/feat/glm5.3-flash) | `feat/glm5.3-flash` | `b1a1289d435d75ade22e9f6983604fcf16efb55f` |
| 官方 SGLang | 固定 commit | `baa9befa381511cd4d56f83980951e5a45fffc6d` |

Plugin 和 Adapter 分别 fork 自 HYGON-AI 的同名仓库，交付分支已推送到 fork，PR 尚未创建。表中 Adapter 提交固定的是交付快照。

```bash
git clone --branch feat/glm5.3-flash https://github.com/Alone-wl/sglang-plugin-das.git
git clone --branch feat/glm5.3-flash https://github.com/Alone-wl/adapter-das.git
```

复现 11 月 7 日评测时，使用下文「2026-11-07 最终 P/D 回归」记录的实验入口。P 在 a12r4n13，D 在 f14r1n15/16，使用 `plugin-dev` 容器、INT8、无 Encoder。保留 CP、HiCache、DeepEP、EAGLE 和 Graph，评测环境已去掉 `SGLANG_SCHEDULER_SKIP_ALL_GATHER`。下文的 Python 3.12 与 D INT4 启动配方属于 11 月 4 日交接快照。

## 2026-11-10 AIME25 对比

| 运行组合 | 正确数 | 错题（零基索引） |
|---|---:|---|
| Plugin P/D（11 月 7 日） | 25/30（83.33%） | 9、12、14、21、23 |
| 侵入式 sglang-das TX-on（11 月 9 日） | 28/30（93.33%） | 13、14 |

两轮各完成 30 个请求，输入消息和标准答案逐题一致。配置记录均为 temperature=1、top_p=0.95、seed=42、max_tokens=64000。TX 独有正确题为 9、12、21、23，Plugin 独有正确题为 13，两边都错的是 14。

Plugin 第 9 题的同一乘法等式整行重复 2469 次，达到 64000 tokens 后截断，最终答案为空。TX 同题在 16107 tokens 正常停止并答对（81）。TX 第 13、14 题均达到 64000 tokens，最终答案为空。按连续 50 词完全重复检查，TX 这轮未见与 Plugin 第 9 题同类的长段复读。两轮运行时及部署配置未做全量对齐，单轮 10 个百分点的差距尚无法归因到 Plugin 或某个算子。

原始 EvalScope 结果相对于容器 `/home/work/agent`：

- Plugin：`das-glm53-tx-pd/perfopt-20261105/accuracy-final-a12/aime25/outputs/20261107_130812/`
- TX-on：`sync-20261108/defer/fusion-aime25/results/tx-on/aime25/outputs/20261109_152612/`

## 2026-11-06 P/D 性能：冷启动 GPU 上下文候选

当前实验使用官方 `baa9befa381511cd4d56f83980951e5a45fffc6d`、plugin `c661691` 的运行时内容和 adapter `f80832f`。plugin 后续 `e2e9966` 仅保留路径审计文档，运行时没有性能改动。P 在 `f14r1n14`，D 在 `f14r1n15/16`；原 `a12r4n13` 曾被 Slurm 标记 DOWN，不能将更换硬件的差异归因于代码。保留 INT8、EAGLE、CP、HiCache、DeepEP 与现有 Graph 配置，仍忽略 E。

改前服务、trace 与冷启动探针位于容器 `/home/work/agent/das-glm53-tx-pd/perfopt-20261105/`；宿主 KFD 的 `kfd-node15-before.json` 已同步至该目录的 `cold-start/candidate/`：

- `d-4k-trace1/`：D 使用参考压测入口，4K 输入、256 输出、并发 128；日志确认任一 rank 的 `#running-req: 8` 后等待 30 秒，再采集 CPU/GPU 10 steps，显式 `with_stack=false`。16 rank 全部导出。rank 0 trace 跨度约 198.64 秒，其中 `hipEventSynchronize` 为 178.87 秒；rank 1 的 DeepEP dispatch 累计 146.35 秒。不同类别存在重叠，不能相加成总耗时。rank 0 的 GPU 事件时间线上有 615 个超过 100ms 的空档，累计约 96.39 秒，很多接近 0.28 秒。此轮压力在导出完成后主动结束于预热阶段，不能给出正式吞吐分数。
- 内核反复报告 GPU `0000:09:00.0` 的 `Over subscribed runlist, process_count:28,max_proc_per_quantum:8`。宿主 KFD 与 Docker PID 映射确认这些是本服务的进程：16 个 tokenizer worker、detokenizer、资源跟踪进程、控制器、主进程，以及其他 GPU 的 scheduler，均在 GPU 0 上保留约 150MB。此现象与通信等待相符，但提速因果仍待同配置 A/B。
- `p-f14-baseline1/benchmark_results.csv`：同硬件 4K 输入、1 token 请求输出、128 并发，128 请求的 Mean TTFT 17100.24ms、Median 16604.38ms、P99 39353.47ms，时长 39.44 秒。经真实 P/D router 的短输出负载偏向 Prefill，但包含传输和 Decode 首步，不能当作纯 P 算子耗时。`p-f14-trace1/` 已导出 8 rank trace。
- `cold-start/first-gpu-init-serving-env.log`：复制实际 D 主进程环境，在 CUDA 初始化前安装阻断探针，首次调用链为 `sitecustomize → bootstrap → common.install → hcu_env.log_summary → arch_name → torch.cuda.get_device_properties(0)`。探针阻止真实初始化，前后 `torch.cuda.is_initialized()` 均为 False。后续隔离还发现 weak-ref 的导入期 CUDA dummy tensor、AITER communicator 元数据探测，以及架构守卫安装晚于 `runner_utils` 导入。

候选属于 adapter 的启动兼容与能力探测，已部署并完成 D 重启，P 正在复验。冷启动时使用同构 KFD topology 的 `gfx_target_version`；混合架构、缺失或非法元数据保留既有 Torch 属性查询。AITER/框架布局探测改查包来源与源码。Torch 的冷启动 current-device/capability 查询通过 HIP 元数据 API 返回，已初始化、非 HIP 或查询失败时保留原 API；HIP 属性枚举 36/37 已用镜像头文件编译核实。gfx/CUDA 版本守卫与公共重导出前移至 runner 导入前。AR+norm 不提前导入未使用的 communicator，实际 collective 仍由原 group 管理。plugin 的架构布尔检测在 adapter 共存时复用冷启动元数据，独立运行及已初始化路径仍查原设备属性。没有修改官方源码或添加环境变量。

weak-ref 保留原 native 符号及失败时 view fallback，将 ABI dummy 探测推迟到首次非空输入的实际设备，按设备缓存；空 tensor 直接返回。实际 native 实现在空 tensor 上报 `cuda:0 versus cuda:-2`，原始失败已留存。`cold-start/weakref-gpu-probe-v5.py` 使用每设备独立 capture stream，通过 GPU 0/3、FP32/FP16/BF16、连续/转置/偏移/空 tensor 和首次 Graph capture、变输入 replay 共 42 项；native 在两卡及首次 capture 用例均实际选中。结果文件为 `weakref-gpu-result-v2.json`。不测试释放 owner 后的悬空引用，也不据此宣称支持该生命周期。

首次 P 重启暴露 NVIDIA SM90 专用 SBO 禁止检查：CUDA 语义层把 HCU capability 9.3 误认成 SM90。adapter 仅在 `initialize_moe_config` 的函数作用域使用真实 HIP 身份，并同步已加载消费者别名；其余 CUDA 语义及 SBO 参数保持。非 HIP 原调用不变。当前 CPU 回归 66 项通过，L0 GREEN（32 项既有 bare-environ 警告）；覆盖安装时序、冷/热设备查询、native 延迟分发、fallback 及 SBO 物理硬件身份。CPU mock 不能代替整模型验收。

`cold-start/cpu-audit-f14r1n14-v7.json` 与 n15 对应文件复制实际 P/D 环境，阻断任何 Torch CUDA 初始化，完整候选 bootstrap 及 plugin 检测均无初始化尝试，新 KFD 进程在全部 GPU 上 VRAM 为零。D 重启后健康检查 200，宿主 KFD 确认仅 8 个 scheduler 各占本卡，不再有 CPU worker 和其他卡 scheduler 的 GPU 0 显存。部署前后 SHA 与备份在 `context-fix-deployment.json` 和 `context-fix-backup/`，SBO 追加修复也记录前后 SHA。

D 持续负载复采 `d-4k-contextfix-trace2/`：同为 940 次 dispatch，rank 1 累计 0.286 秒（改前 146.354 秒），rank 0 GPU 时间跨度 1.981 秒（改前 178.344 秒）；这是 trace 等待改善，输入状态和实际接受长度并非逐条锁定，不能称纯算子提速倍数。新轮 4K/256/128、2048 请求正式结果：平均 ITL 36.41ms，Mean TTFT 777.76ms，时长 175.83 秒，11.65 req/s。两次 profile 可能影响该轮时延；旧轮仅完成预热，没有正式吞吐 A/B。待完成 P 原参数启动、同硬件 P A/B 与新 profile、真实 P/D 请求及 AIME25/HumanEval 回归。SCNet 公网 SSH 间歇失败不代表服务退出，不重复加载模型。内网 `zz-login03/04` 已解析为 `10.13.17.103/104`，login03 已经跳板登录；未发现公网直连入口。

P 后续原参数启动通过，完整 P/D 算术请求返回 2，`finish_reason=stop`。同硬件 P 首轮 4K/1/128、128 请求 Mean TTFT 16208.13ms、Median 16289.76ms、P99 27663.54ms、时长 27.75 秒；改前为 17100.24/16604.38/39353.47ms、39.44 秒。单轮有波动，不承诺稳定提速倍数。`p-f14-contextfix-trace1/` 已导出 8 rank，rank 0 一次 i8 GEMM 耗时 4.085 秒，其余调用中位数 1.563ms，rank 1 中位数 1.507ms，不能解释为持续 FFN 不均衡。P 的旧 gateway 仍保留约 150MB GPU 0 上下文，其他 CPU worker 和跨卡 scheduler 不再分配本卡外显存。AIME25/HumanEval 于 14:37 并行启动，使用 `/home/work/eval.sh` 副本，仅修改数据集、模型、API 端点和结果目录，生成配置保持；分数与 badcase 待完整结果。

## 2026-11-06 完整数据集结果与后续缓存故障

原 context-fix 组合完成 AIME25 **25/30（83.33%）**、HumanEval **163/164（99.39%）**，预测、review、配置和 report 位于 `perfopt-20261105/accuracy-contextfix/`，逐题证据为 `badcase-evidence.json`。AIME 零基索引 12/14/29 均生成 64000 tokens、`max_tokens` 结束且最终预测为空；20 将点 C=(-15,0) 到竖直边 GH 的水平距离误写成 9-x（应为 24-x），输出 1753 而目标为 293；23 将 9+120+10 误算成 149（应为 139），最终输出 159 而目标为 149。HumanEval/132 反复推敲/重建括号算法后达到 64000 tokens，最终 completion 为空，判题首个断言失败。以上区分截断和数学错误，未据此证明机械复读或 adapter 数值漂移，也不宣称 handoff 精度问题已全部解决。

原评测脚本的顶层 `enable_thinking=false` 被当前官方 ChatCompletionRequest 忽略；CPU 实测解析结果的 `chat_template_kwargs` 为 None，嵌套字段可进入该参数。实际本轮包含长 reasoning，成绩保留原脚本的真实语义，不称为关闭思考的分数；字段可进入解析也不单独证明特定模型模板遵循该开关。

全集结束后的 P 稳定基线在首个重复 4K 请求卡住。非阻塞栈及 locals 证明 CP rank 0/3 的逻辑新 token 数分别为 4096/256；watchdog 的缓存 census 是在途快照，不是独立的泄漏证据。自动 py-spy 又停住全部 scheduler，现场和终止记录已保存。新增前缀协调归 plugin，详见其开发记录：原 upstream match/COW 保留，共同可用 prefix 通过 attention CPU groups 协调，不一致时重新 lookup，未关闭 HiCache、CP 或 SBO。8-rank 真实缓存回归及 P/D 长前缀 2/2/7 集成通过，实际 cached_tokens=3840。

仅 CP 修复的三轮基线只有首轮有效，Mean TTFT 16480.66ms、时长 27.56 秒；第二轮预热在 Mamba checkpoint 报 `Can not alloc mamba cache`。原生 fork 的压力分支会释放已消费的 COW 源状态锁。plugin `a403228` 将此生命周期处理接在原结果 snapshot 之后，仅对已完成 `copy_done` fence 的 HCU GLM extend 批次释放旧 Mamba 锁，保留 FULL KV 锁并更新 receipt。未打开 lazy buffer，也未增加原来的 96 个槽位。真实 8-GPU 满池反例旧路径失败，候选全部通过，复用旧 slot 不影响两个请求的自有 COW 状态；仓内 8-rank 缓存/压力测试同样通过。

该锁修复与 CP 协调已部署，P 于 17:34 重启，manifest 为 `mamba-lock-fix-deployment.json`。新一轮长前缀 2/2/7 均正常结束，后两次 cached_tokens=3840；`p-f14-cowfix-baseline3/` 正在执行三轮同硬件稳定基线。normal INT8 异步计数仍是独立候选，尚未部署；最终组合的精度回归待执行，不能混用前后版本的精度与性能。

17:34 组合仍在第二轮预热报槽位不足，只有首轮有效（Mean TTFT 16595.32ms）。后续定向库存复现确认：失败的三个 rank 各有 47 个 pending HiCache 写回节点，7 个完成 ACK 覆盖 43 个节点，但全 rank 最小完成数被其他 rank 的空队列压为零，写回锁无法回收。plugin `e3d020f` 将本地写回回收限定到 GLM CP LayerSplit、attention TP=1/PP=1、write_through、cache 模式且无 L3；原 ACK synchronize/锁释放、load 和全局 ready-count/digest 同步保留。真实 8-GPU 资源反例与正式 Hook 回归通过，D2H 快照逐位一致；整服务原参数重启与 `p-f14-writeack-baseline3/` 三轮压力待完成。另一项释放新 checkpoint 锁的局部候选已回退且从未部署，不能把它作为本轮根因修复。临时异常诊断已清理，normal INT8 异步计数仍未部署。

## 2026-11-07 最终 P/D 回归

P 已迁回 a12r4n13，D 保持 f14r1n15/16；INT8、无 E，保留 CP、HiCache、DeepEP、EAGLE 与既有 Graph 配置。实验根目录为 `/home/work/agent/das-glm53-tx-pd`，P 与 router 启动入口分别为 `tx/run-p-a12r4n13-current.sh` 和 `tx/run-router-a12r4n13-current.sh`，A12 容器内 API 为 `127.0.0.1:30002`。长前缀三条请求完整返回 2/2/7，后两次命中 3840 tokens。F14 临时 P 和 router 已停止，容器保留。

写回回收修复后，同硬件两臂各 1536 个压力请求全部成功。P 异步计数使 rank 0 的 430 次同步复制降至 0，10 steps 减少约 406ms CPU 调用时间；三轮端到端 TTFT 中位数 16.26→16.30 秒，未测得提速。F14 A/B 不与 A12 混作性能对照。

A12 最终组合并行完成 AIME25 **25/30（83.33%）**、HumanEval **163/164（99.39%）**，194 个请求均完成、无请求错误。两份脚本与上一轮逐字节相同。AIME25 零基索引 9 同一乘法等式重复 2469 次后截断；12、14 达到 64000 tokens，正确答案只出现在推理中；21 为容斥计数错误，23 多计开区间的一个零点。HumanEval/116 的题面示例与文字规格及测试不一致，模型选择了只按数值排序，遗漏二进制 1 的数量。证据与原始报告见 `perfopt-20261105/accuracy-final-a12/`。总分与旧组合相同，错题不同；仍有复读，尚不能归因于算子或宣称 handoff 的精度问题已解决。

## 2026-11-05 grammar mask 启动线程兼容

用户在原 P 启动入口重启后，首次 grammar mask 调用报 Triton
`OutOfResources: threads, Required: 2048, Hardware limit: 1024`。
官方 `bitmask_ops.py` 对 FP32 logits 选择 32 warps；gfx936 的 warp size
为 64，导致单 block 超过 1024 线程。该失败已在同镜像独立探针复现。

修复位于 adapter 的 `hcu_adapter/bridges/_sglang/sampling.py`，属于既有
采样能力的硬件启动兼容，不新增计算档位。采样桥安装时只包装该 JIT
kernel 对象的 `run`，按实际设备 warp size 限制 `num_warps`；仍调用
当前原始 launcher，保留其 dtype、shape、索引和 bitmask 数学语义。
预导入的公共函数及 xgrammar 别名继续引用同一个 kernel 对象，因此无需
替换调用者或编辑官方源码。非 HIP 环境不安装；CPU 输入不查询 GPU；
重复安装不叠加包装。运行绑定以 `sampling` 登记到 adapter registry，
未新增环境变量、catalog 档位或配置 UI 项。

本次验证：5 项离线接缝回归通过；L0 GREEN，32 条历史 bare-environ
警告；a12r4n13 的 plugin-dev（Python 3.10、Torch 2.11、gfx936）通过
21 项 GPU 检查，包含 FP32/FP16/BF16、奇数及模型规模词表、指定行、
一维输入和变 logits/bitmask 的 Graph replay，均与参考逐位一致。
原始失败和 GPU 结果在实验目录 `grammar-launch-fix/gpu-probe.log`，
探针在同目录 `probe.py`。本次算子检查不等于重新完成 P/D 启动和数据集
精度验收；服务集成另行核验。原历史源码清单保留快照语义。

## 2026-11-04 交付版本与来源

- Plugin：HYGON-AI/sglang-plugin-das，`feat/glm5.3-flash`；本地基底 `405b8dc`。
- Adapter：maxiaobetter/adapter-das（HYGON-AI/adapter-das 的 fork），`feat/glm5.3-flash`；本地基底 `b8af034`。
- 两仓分别提交，必须配套使用；原开发工作树和远端实验目录未修改。
- Plugin 运行时完整同步远端 `sources/short-topk-alignment-20261104/decode/plugin`，包含其依赖，而非仅复制 GLM 文件。它与 prefill 快照仅 `models/glm5_next.py` 不同：decode 版同时允许 W8A8 和 SlimQuant 两个量化类，W8A8 prefill 条件仍保留。
- Adapter 运行时完整同步 `sources/kda-beta-fp32-candidate-20261103/adapter`。这把 FP32 beta 修正统一带入配套版本；原实验曾只给 P 使用该候选，统一版本的 D 路径仍须回归。
- Plugin 两个文件仅清除末尾多余空行以通过 diff 检查，原始/交付哈希均记录。
运行时代码及打包入口逐文件 SHA256 见 `docs/glm5.3-flash-source.json`。文档保留本地历史记录，不能把历史分数视为当前快照的成绩。
- 远端原包含通用框架兼容和其他模型依赖，因此本分支是可追溯候选快照，不是已经缩减到最小的 GLM 专属补丁。后续按依赖关系拆分评审。

## 2026-11-04 环境与启动配方

用户指定后续 plugin/adapter 暂用 Python 3.12 镜像：

```text
10.16.1.152:5000/jenkins/model_test_env/vllm:0.28.1rc1-ubuntu22.04-dtk2610-py3.12-20261103-1558
```

外网仓库地址为 `42.228.13.241:5000`。21～24 已导入，镜像 ID 为
`sha256:bba1aa236bcad9dbfcbb2438c4d1d53011f6671bda634720d8bf809b95adfd33`。
这是环境物料准备完成，尚未在该镜像上完成本分支的 GLM 模型启动与精度验证。
镜像名含 vllm 不表示本次切换推理框架；本次代码仍为 SGLang 适配。

Plugin 提供 `examples/glm5.3-flash/launch.sh`、`router.sh`、`adapter-config.json`、`deepep.json`，
以及源码校验器 `verify_upstream.py` 和无插桩源码指纹 `upstream-source.json`。
脚本从远端最后候选启动参数整理：P=TP8/CP8/EP8、INT8、normal DeepEP；
D=两节点 TP16/DP16/EP16、INT4、low_latency DeepEP；EAGLE 5 steps/6 draft tokens。
节点编号 22/23/24 是脚本角色选择值，IP 由变量提供；21 不在该 P/D 拓扑内。
已移除诊断 observer 环境，脚本不启动容器、不停止其他服务，也不安装依赖。

容器内使用对应 Python 3.12 解释器，按 cookbook 安装匹配的官方 SGLang 源码与依赖；
再用 `python3 -m pip install --no-deps -e "$PLUGIN_DIR"` 注册 plugin entry point。
显式设置 `ADAPTER_DIR`、`SGLANG_DIR`、`PLUGIN_DIR`，避免旧 `/work/site` 包混入。
历史记录固定的官方 SGLang commit 为 `baa9befa381511cd4d56f83980951e5a45fffc6d`，
来源为 cp-pd-source-manifest 与原安装文档。远端 `sources/official` 是无 `.git` 的导出树，
本次不能重新由它证明 commit 身份，故同时交付实际官方 Python 包的逐文件 SHA256，启动必须匹配。
`SGLANG_DIR` 可指向含 `python/sglang` 的源码树，或含 `sglang` 的包目录。
新启动脚本先校验源码，再设置精确 PYTHONPATH，不继承旧 observer/site 路径。

实验树 `sources/paired-layer-boundary-20261103-v2/migration/sglang` 对比保存的
`official/python/sglang`，只有 `srt/model_executor/model_runner.py` 不同：文件末尾新增
`_install_glm53_boundary_observer()`，无条件 import `glm53_paired_observer.install_runner` 并安装。
因此只 unset observer 环境不足以去除导入依赖。部署使用保存的无插桩官方树；校验器会拒绝该诊断树，
无需给产品提供 observer stub，也不修改在服官方文件。哈希检查覆盖包文件，不认证外部依赖、C++ 构建产物或镜像。

```bash
export PLUGIN_DIR=/path/to/sglang-plugin-das
export ADAPTER_DIR=/path/to/adapter-das
export SGLANG_DIR=/path/to/compatible-sglang
export MODEL_DIR=/path/to/models
export WORK_DIR=/path/to/writable-work
export GLM53_NODE=22
export HOST_IP=<prefill-ip>
export DECODE_MASTER_IP=<decode-rank0-ip>
bash "$PLUGIN_DIR/examples/glm5.3-flash/launch.sh"
```

D 两节点分别设 `GLM53_NODE=23/24` 与本机 `HOST_IP`，其余配对变量保持一致。
模型子目录分别为 `GLM-5.3-Flash-W8A8-INT8` 和 `GLM-5.3-Flash-W4A8-INT4`。
NIC 默认 bond1/shca_0..3，须按节点真实设备核对。模型服务就绪后单独启动 router：

```bash
export ROUTER_HOST=<router-ip>
export PREFILL_URL=http://<prefill-ip>:30822
export DECODE_URL=http://<decode-rank0-ip>:30822
bash "$PLUGIN_DIR/examples/glm5.3-flash/router.sh"
```

router 使用 round_robin、请求超时 7500 秒、禁用自动重试；默认服务/metrics/bootstrap 端口
为 30881/29081/30998。`sglang_router` 须在对应解释器环境可导入；未在本次启动或重装 router。
客户端按原评测 manifest 配置，不能将 router 的超时设置当作生成 token 上限。
脚本中的 1M context、32K chunk、HiCache/Graph 参数继承实验配置，不表示已在新镜像验收。
脚本通过 bash 语法检查；没有在本次整理期间启动模型。

必要数值开关：两角色 `SGLANG_USE_FUSED_SILU_MUL_QUANT=1`；D 的
`W8A8_SUPPORT_METHODS=3`；`SGLANG_HCU_NATIVE_MHC=1`；LightOp MLA gather 开启。
`HCU_CONFIG` 选择 DTK sparse MLA、LightOp INT8 MM、Triton grouped topk。
网页 `tools/hcu_config_ui.html` 已能生成旧 cookbook 的通用启动命令及 P/D/router 三段命令，
但尚未接入本次 GLM-5.3 plugin+adapter 配方，不能用它替代本次脚本；
plugin `architecture.html` 是文档导览。本次以仓内 `.sh` 为启动交接入口。

## 已完成的代码与证据边界

| 环节 | 当前实现/证据 | 边界 |
|---|---|---|
| KDA prefix state | adapter 贯穿 track_state/track_chunk_idx，保存 FP32 快照；历史 cold/hit/host reload 定向验证通过 | 不等于所有长度/批次/新镜像已验证 |
| KDA beta | raw beta 在 FP32 做 sigmoid，避免转回低精度 | 已同步；与前述状态快照是两项不同修正 |
| FFN 量化及截断 | plugin GLM MLP → row_fusion → LightOp，传 swiglu_limit；W8A8/SlimQuant 接收预量化输入 | 不只加启动开关；需要模型 forward 和量化接口共同接通 |
| 两次 DeepGEMM 中间激活 | INT8 runner 保留 SiLU、clamp、量化顺序及相应分支 | normal/masked、INT4/INT8、空批次须分别核实，不能跨分支外推 |
| 短序列 KPool | 原生有序 arange 替代零 logits 的 topk 顺序；保留 PAGED/RAGGED、逻辑索引及 tail padding | 远端同前缀重复性 24/24 通过；九题 6/9，复读未解决 |
| MLA / HiCache / LayerSplit / MTP | 含 native MLA、按页广播、非 owner scratch、draft host 层映射、speculative 兼容 | 历史定向探针已记录；本次整合快照未做新的 GPU 验收 |

短索引根因证据：三题、每题两次、八个 prefill rank，共 24 对；第 0～2 层完整边界及
第 3 层投影/归一化/Q 相同，首差异在相同零 logits 的 topk 索引顺序。
48 次捕获均实际命中 DTK sparse MLA；固定 Q/KV、交换索引顺序可逐位复现 24 对输出。
这证明这些输入上的 MLA 顺序因果，不能单独解释全部 GPQA 错题和复读。

## 未完成与已知问题

1. **当前精度与复读未闭环（最高优先级）**：修复后三题 2/85/131，每题两次、八个 P rank，24/24 配对通过。
   这是 P 重复性验证：45 层抽样六边界、0/1/2/3/44 层完整边界、第3层 MLA、末端 logits 全部通过；
   不是所有层所有元素的原生/迁移等价，也不覆盖 D verify 状态。
   最新九题 6/9（66.7%），错 7/90/127；102 由错转对，第7题由对转错，总分持平。
   全部 stop、无请求错误/重试/截断，max_tokens=65536。127 连续50词窗口最多出现3次，其他八题未检出该类长段完全重复，
   不等于没有语义重述。逐题答案、参考、响应哈希与计数见 `glm5.3-flash-results-20261104.json`。
   **上述结果仅归属远端 P/D 分别使用的快照**，不是统一提交组合或 Python3.12 新镜像验收。
   历史 P-only 融合候选九题 7/9，P+D 融合候选 6/9（错 90/102/127），均正常 stop，仍有重复片段。
   这是筛选坏例、非固定种子的单次结果，不是 GPQA 总体准确率或因果 A/B。
2. **历史全集不可继承**：较早冻结配置 HE 97.56%、MATH500 95.40%、GPQA 90.40%、HMMT 80.00%；
   对照 97.56/95.60/89.90/80.00。后续代码和环境已改变，最终候选四项全集必须重跑，保留截断与失败。
3. **镜像对齐未完成**：文档 Python 3.10 镜像尝试在 DeepEP Buffer.sync → ROCSHMEM GetClosestNicToGpu 崩溃；
   单卡可见性兼容是候选线索，未证实根因。cwd=/ 配合 PYTHONPYCACHEPREFIX 的启动失败已通过显式工作目录规避。
   Python 3.12/DTK26.10 需要重新核对 Torch、LightOp、FlashMLA、DeepGEMM、DeepEP/Mooncake 版本和共享库哈希。
4. **所有路径审计仍待完成**：长序列 pooled topk 的 AITER/LightOp dispatch、MQA 量化、KPool 写入/尾状态、
   head-gate 精度、MTP index sharing、prefill chunk/cache hit/host reload、decode eager/graph 与 verify/draft 分支。
   每项记录 caller、实际 kernel、shape/dtype、量化/截断/索引顺序及状态输入，不以启动成功代替对齐。
5. **性能未对齐**：历史热点为 ragged MQA、INT8 MM、CP 重排/广播及 expert token 不均衡；
   有独立优化收益但无最终同环境端到端验收。D2H 临时方案曾退化，不作为推荐配置。
6. **运行风险**：曾因独立 GPU 探针与 resident 模型争用触发 NUMA 内存策略 OOM；复测应使用明确独占窗口。
7. **工程收口**：远端运行时包含其他模型通用依赖、历史回退及诊断入口；需继续按归属拆分，补齐正式版本矩阵。
   本次不新增单测、不启动完整测试或 GPU 评测；保留当前源码而不把候选接口重写成另一套行为。

## 接手顺序与证据位置

先固定两仓提交、官方源码、镜像和核心库哈希，检查进程实际 import 来源；再按 prefill/decode/verify
做逐分支清单。已有远端 24/24 与九题 6/9 可作诊断依据；统一提交组合/新镜像须单独复测，继续定位余下错题与复读，之后跑全集与同配置 profile/性能。
任何失败保留原请求、响应、token 上限与 finish_reason，不将重复推理仅按最终答案正确忽略。

原始远端任务根目录由维护者提供为 `GLM53_TASK_ROOT`，不要依赖临时 `/tmp` 脚本。
关键证据相对路径：

- `logs/mla-internal-repeat-20261104-v2-complete.tar.gz`，SHA256 `6d37c690b24f71e4016b7caa2cafef278afbd2c2e128d119ea18e330257fbb0b`。
- `logs/ffn-lightop-pd-gpqa-badcase-20261103-v1`：P+D 九题原始输出与 review。
- `logs/short-topk-aligned-repeat-20261104-v2/alignment-verification.json`：最新 24/24 passed。先前 client.log 的 waiting/cases_sent=0 已被后续结果取代。
- `logs/short-topk-aligned-gpqa-badcase-20261104-v1`：九题全部结束，status 的 pending_review 字样不代表生成未完成；本次重新读取 response 并核对参考为 6/9。
- `scripts/short-topk-aligned-repeat-20261104-v2`：历史启动与诊断脚本。
- 两仓 `docs/glm5.3-flash-source.json`：已同步源码的来源与 SHA256；配套历史细节见 plugin 的 `docs/hcu-sglang-development.md`。

## 本次验证

两仓使用 Python 3.12 执行 `ci/l0_gate.py`；plugin 还按 runner 配方检查 guide/migration-plan 快照；
两仓执行 `git diff --check` 与暂存差异检查。Python AST 语法检查包含在 L0 中，启动脚本运行 `bash -n`。
这些是静态交付验证，不代表新镜像上的模型精度、性能或页面浏览器渲染验收。
仓库当前未绑定 GitHub Actions 的自动检查不能写成“GitHub CI 已通过”。

## 更新范围与仍待执行的验收

本次只更新交接、证据、启动保护及 router 配方，没有改变两仓运行时代码、重启远端服务或运行 GPU 评测。
**adapter 网页配置按用户要求暂不接入**；旧网页生成器不能代替这套脚本。
原代码提交 plugin `8dac29c` / adapter `90a2c91` 的来源哈希契约保持不变。
脚本完整性与模型验收分开：尚待统一提交组合的 import/启动、Python3.12 核心库一致性、全路径数值和最终精度/性能复测。
