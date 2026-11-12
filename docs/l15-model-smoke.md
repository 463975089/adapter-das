# L1.6 模型启动就绪：真实模型 TP 与显存配置验收

L1.6 的 manifest 和统一入口见：

- `tests/manifests/l16_model_readiness_v1.json`
- `ci/l16_model_readiness_gate.sh`
- `python -m tools.l16_model_readiness validate`

L1.6 位于 L1.5 组件门和 L2-F 完整模型功能验证之间。它只回答指定模型能否按
绑定的 SGLang 源码和 adapter 配置启动，并完成 eager/graph 基础生成；完整精度、
性能、并发和多机功能仍由 L2-F/L2-P/L3 负责。

`tools.l15_model_smoke` 使用社区 Engine 执行真实 checkpoint 或显式
dummy 模型；二者的权重证据分开记录。默认 `--tp-size 1`，
可选择 1/2/4/8。每个 TP 规模、模型和 dtype 都须分别验收。

`--mem-fraction-static` 转发社区 Engine 的静态显存预算，默认保持
0.25。较大真实 checkpoint 可显式指定其他预算，例如
`--mem-fraction-static 0.7`；取值必须是零和一之间的有限数。
控制器、worker、实际 server info 和参考报告须匹配同一预算，
参数丢失或被忽略不能通过。旧报告缺失字段只按历史 0.25 解释，
不证明其他预算。该选项不改变模型 dtype 或原精度门槛。

启动前确认所选设备均空闲，再通过 HIP/CUDA 可见性暴露恰好 TP 数量的
设备。预检会逐张分配、计算及同步，并记录 `runtime.devices`；
它验证设备可用性，不负责判断是否有其他用户作业。

`--host adapter` 默认禁止加载额外 plugin；显式使用
`--host adapter+plugin` 时，每个 TP 配置仍须检查公开 plugin loader
实际注册了 `hcu` hook。预检和 worker 同时绑定 host 与 TP 参数，
不能以多卡设备可用代替 plugin 加载成功。

```bash
HCU_ADAPTER_HOST=none \
HIP_VISIBLE_DEVICES="$DEVICE_IDS" CUDA_VISIBLE_DEVICES="$DEVICE_IDS" \
python -m tools.l15_model_smoke \
  --up-tree "$HCU_UP_TREE" \
  --families checkpoint --checkpoint "$HCU_CHECKPOINT" \
  --tp-size 2 --modes eager,graph --profile accuracy \
  --out results/model-tp2.json
```

不要继承与本轮选择冲突的 `ROCR_VISIBLE_DEVICES`。预检保留社区源码
绑定、导入来源及完整字节检查。控制器向 worker 和 Engine 传递 TP
数量，并核对实际 server info 和 scheduler 状态；请求 TP2 但运行
TP1、worker 缺少 TP 配置或设备数量不符均不能通过。

主报告和每个 worker 的 `tp_size` 必须一致。旧报告没有该字段时仅按
历史 TP1 解释，不能用于证明多卡。实际 graph 分配、三批固定输入、
缓存复用与变输入生成，以及 checkpoint 前后字节绑定继续按原合同验证。

精度参考从已绑定的报告读取 TP 配置，使用隐藏 GPU 的 CPU FP32 参考：

```bash
HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
python -m tools.l15_checkpoint_reference \
  --smoke-report results/model-tp2.json \
  --out results/model-tp2-accuracy.json
```

精度报告以 `execution_tp_size` 标记受测配置，继续使用原版本化策略。
模型 smoke 的 `passed` 只证明启动、生成及对应证据完整，
必须同时检查独立精度报告；TP2 成功也不外推 TP4/TP8、EP 或其他模型。
量化参考仍须使用已验证的显式量化参考入口，不能作为普通 FP32 模型处理。

## 上游同步后的接口回归

对接 adapter 上游 `b3f4bb0250dbf4b84bd71473fe03817204e18293`
时，同时保留上游 host/plugin 检查和新增 TP 配置：
`preflight` 的第四个位置参数仍为 host，设备数量通过
`device_count=` 显式传递。控制器向 worker 同时传递 host 与 TP，
并核对报告中的两项配置。

远端隔离快照完成 269 项 CPU 回归，GPU 全部隐藏，同一解释器结束时
CUDA 未初始化；L0 GREEN，31 条已有警告。组合回归实际执行生成的
设备探针，覆盖 adapter/plugin 两种模式、两设备逐个分配，以及 plugin
未注册 hook 时拒绝启动。另覆盖 TP2 worker 参数遗漏、忽略或布尔值伪装。
该结果只证明工具与接口合同；新上游基点上的 HCU 矩阵、真实模型和精度
仍须分别验收，不能继承此前代码组合的通过结论。

### 同步后 Qwen3-0.6B BF16 TP2 复验

与 `d237c09` 运行文件字节一致的远端隔离快照完成 accuracy profile：
eager/graph 均启动成功，六组中英文、512/2048 token、复用及变输入
请求全部生成完成。两张设备逐个预检通过，worker、server info 和
scheduler 均绑定 TP2。社区全部 10594 个跟踪文件及额外源码检查、
checkpoint 前后字节审计均通过。

12 组请求的输入、输出 token IDs、所选及完整报告 top20 概率，与同步前
TP2 对照逐位一致；六对 eager/graph 结果也逐位一致。重新运行的独立
CPU FP32 参考仍为 **原 V1 FAIL**：96/96 位置 top1 一致，所选 logprob
最大误差 0.027206，top20 logprob 最大误差 **0.604497**，超过原门槛
0.15。新旧参考的 policy、limits 和全部 maxima 相同；不放宽门槛、
不修改默认 dtype，也不以启动或模式一致替代精度验收。

完整日志、实际参数、模式/旧版对照及最终审计绑定保存在仓库外
`external-evidence:tp259`。此结果仅覆盖该模型的 BF16 TP2 配置。

### 同步后默认 TP1 兼容性复验

同一快照省略 `--tp-size`，完成真实 Qwen3-0.6B BF16 的完整
accuracy profile eager/graph 生成。实际预检一张设备，worker、
server info 和 scheduler 均为 TP1；社区源码及 checkpoint 前后
字节审计通过。12 组请求的输入、输出 token IDs、所选及完整报告
top20 概率，全部与引入 TP 参数前的原单卡对照逐位一致。

此项确认默认单卡行为保持一致，没有重新执行 CPU FP32 精度参考，
不新增精度通过结论。报告、原始日志、配方差异和原对照绑定保存在
仓库外 `external-evidence:tp262`。

### 显存预算参数回归

显存参数及参考绑定的远端 CPU 组合回归为 291 passed，无失败或跳过，
同一解释器结束时 CUDA 未初始化；L0 GREEN，31 条已有警告。
负控覆盖非有限数、边界值、布尔值，以及控制器、worker、实际
server info 的配置遗漏或不一致。新增 memory_fraction 子集在修改前
为 16 failed / 1 passed，修复后组合回归通过。

真实 Qwen3-0.6B BF16 TP1 分别省略该参数和显式指定 0.7，
两组均完成完整 accuracy profile 的 eager/graph 生成。控制器、
worker 和 server info 分别绑定 0.25、0.7；每组 12 个请求的输入、
输出 token IDs、所选及完整报告 top20 概率，均与此前默认 TP1
对照逐位一致。社区源码前后审计和全部 603 个候选文件字节检查通过，
本地另从归档原始报告独立核对配置、权重绑定及全部概率对照。

此项仅证明受测模型的预算配置传递与输出兼容性，没有重新执行
CPU FP32 精度参考，也不证明更大模型、其他预算或 TP 配置通过。
原始报告、日志和四个受测 Python 文件保存在仓库外
`external-evidence:memory274-275`，归档 SHA256：
`7726829e50491e37075109fee8a47818baeab73140cb0618ff3b5c039487bc27`。


## L1.6 控制器失败合同

L1.6 每次启动子任务前删除旧报告，逐项核对报告类型、家族、eager/graph
模式、dtype、TP、显存预算、host 及 pristine 社区 commit。每个模式必须有
成功的 worker、源码完整性和真实/dummy 权重身份；真实 checkpoint 还要求
权重绑定和运行后审计证据。源码发生变化时停止后续任务。

所有选中 case 的 checkpoint 参数在第一个模型任务前检查。子任务复用
`tools.l15_runtime.run_job` 的独立 Linux 进程组和超时清理；失败与超时保留
本次已完成的 case 报告及日志。Windows CPU 回归只验证控制器合同，Linux
进程组清理需在远端执行生命周期回归。


## 基础请求重放

`smoke` profile 按原请求、变输入、原请求重放执行三个批次，要求重放的 greedy
输出 token 相同，并报告所选 logprob 的最大差。概率差用于诊断，不代替独立精度
参考；原 accuracy profile 的请求和精度阈值保持不变。报告中的
`request_replay.cache_hit_proven=false` 明确表示请求重放本身不能证明实际缓存命中。
后续需要独立缓存计数或受控 cache-on/off 证据。

`tools.l15_upgrade_check` 在严格组件矩阵之后调用 L1.6 runner，要求汇总报告绑定
本次审查的 SGLang candidate commit；缺失或其他 commit 的报告均不能使升级就绪。


### 实际缓存命中探针

初次 Dense 实测中，三批带 `logprob_start_len=0` 的请求均报告 `cached_tokens=0`。
因此新增独立探针：保留原请求，改为仅取输出 logprob（`logprob_start_len=-1`），
连续预热和重放。Dense/MoE/MLA dummy，以及已登记的 Qwen2/Qwen3/Llama/
DeepSeek V2/V3 checkpoint smoke 要求两个重放请求各自的 `cached_tokens` 为正整数，
并且 greedy token 相同；结果写入 `cache_probe`。recurrent 模型及 accuracy profile
暂不启用该探针，单列适用范围，不能记作缓存命中通过。

新增探针尚需独立 HCU 实测，不能由此前 `d744a43` 的启动/重放结果转授。
