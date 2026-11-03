# HCU SGLang 启动就绪门禁执行计划

目标：社区 SGLang 源码不变，通过 adapter 的 Triton 功能基线与模型执行链回归，
为明确的模型及配置提供启动依据。不以性能作为本轮验收条件。

## 当前迭代目标（2026-11-03）

用户要求按以下顺序持续推进，每个独立功能点验证后及时提交：

1. **社区源码完整性**：直接使用社区源码，不改写其 Python、C++/CUDA/HIP
   文件，也不在启动时向社区源码目录打磁盘补丁。适配实现留在 adapter，
   优先使用正式扩展接口，其次运行时接口适配。记录当前仍会写回上游的路径，
   逐项迁移；不能仅因用户没有手工改文件就宣称“源码零修改”。
2. **多模型功能一致性**：社区 SGLang + adapter 在 HCU 上运行多个真实
   checkpoint，覆盖 Dense、MoE、MLA、状态空间/混合架构及适用量化配置。
   保留上游功能与数值语义；硬件专属例外必须有具体原因和证据，不因缺少
   NVIDIA 库就直接排除可用 Triton/Torch 实现恢复的功能。
3. **功能与精度验收**：性能不作为当前门槛。验收包含真实加载、启动、
   prefill/decode、生成、数值对拍或语义金标，以及受测功能对应的状态、
   变输入 graph 和请求隔离回归。dummy 模型和历史战报不能替代当前组合
   的真实权重验证；启动成功、有限 logprob 本身不能证明模型精度。
4. **每日追新最后启动**：保留社区升级对账工具；其余优化和验证完成前，
   不配置或启用每日巡检。最终再接入每日社区 main 变更跟踪、适配修复、
   最新源码 + adapter 的模型回归与结果记录。

模型配方优先参考用户指定的
[inference-cookbook-das](https://github.com/HYGON-AI/inference-cookbook-das)，
逐模型记录 cookbook commit/脚本、checkpoint、卡型、依赖/框架/adapter
版本、实际参数、生成与精度结果。GPU 测试只在用户授权的远端空闲卡运行；
禁止在本节点执行全量 adapter 测试。

2026-11-03 用户新增 nmz7 调试授权：nmz7 与 nmz28 都使用
`sglang_liph_adapter` 容器，代码目录相同。nmz7 经资源检查确认 GPU
4–7 空闲、0–3 在用；首次回归选择物理 GPU 4，只暴露这一张卡。
每次任务启动前重新检查空闲状态，结果单独记录节点和设备，不沿用
nmz28 的环境结论；不清理或占用其他用户任务。

2026-11-03 用户进一步授权 nmz21 的同名容器。经 nmz7 免密跳转确认
代码目录相同，8 张 BW1100 均约 97% 空闲显存，本轮仅暴露物理 GPU 0。
清理该新容器镜像残留的 SGLang 0.5.18 DTK 定制包与 kernel 包，
核对 SGLang 从社区源码、`sgl_kernel` 从 adapter 导入；不更改其他节点环境。

当前前置差距：`jit_*`、`dsv4_*`
等存在源码磁盘写回，须以实际调用点逐项审计，
上述目标尚未完成。对象广播 `dist_bc` 已迁移为运行时只读 numpy buffer
兼容，保留已导入函数别名，不再写回 `common.py`。nmz28 隐藏 GPU 的
回归为 **6 passed**：包括以 Git 原始社区广播函数执行两 rank spawn/Gloo，
验证双向广播、空/中文/bytes/大对象和测试源文件字节不变。
这是广播路径的迁移证据，尚不代表整个 adapter 启动不再修改源码。
回归日志为 `results/l15-readiness/pristine-dist-bc.log`；本地 CPU 契约
4 passed（需远端社区树的 Gloo 项在本地不运行），L0 GREEN（31 warn）。
真实 checkpoint eager/graph 复验在启动前因 GPU 0 仅约 4% 空闲而被阻止，
未运行，也未占用其他卡；该集成复验仍待空闲窗口补齐。

新增 nmz7 授权后，在空闲物理 GPU 4 补齐真实 Qwen2-0.5B checkpoint
eager/graph 回归，**2/2 PASS**；报告为
`results/l15-readiness/iteration1-nmz7-checkpoint.json`。此次使用旧完整性
判据，社区树仍包含其他历史磁盘补丁，所以只作为广播迁移后的启动回归，
不作为新“原样源码”目标通过的证据。

### 源码完整性门禁

`HCU_ADAPTER_HOST=none python3 -S -m tools.upstream_source --up-tree <社区树>`
逐文件将实际字节与 HEAD Git blob 比较，不依赖索引、mtime、
assume-unchanged 或 skip-worktree。被忽略的新增 Python/JIT 源文件也
计为失败；子模块未递归验证则明确拒绝。报告保存所有差异路径。
该工具证明检查时的源码状态，不能单靠前后快照证明从未发生又被撤回的写入。

严格组件和模型入口默认在 GPU 预检前拒绝已有改动；组件运行结束、每个
模型 worker 结束后再核对一次，要求 commit 未改变且源码仍完整。
模型即使生成成功，只要改写社区源码也判失败并停止后续模型；预检失败
覆盖旧绿色报告。历史 27 case / 12 model-mode 的通过记录继续保留，
但不转授为新完整性规则下的通过结果。

nmz7 审计当前共享社区树：**10,594 tracked files，30 changed files，
0 added sources**。报告为 `results/l15-readiness/iteration2-upstream-source.json`。
这说明原样源码目标尚未达成；不通过 reset/删除现有工作树改动制造绿色，
后续在保留现场的前提下迁移适配并从干净社区树验证。
门禁与广播、编排工具组合远端 CPU 回归 **68 passed**；补充旧报告覆盖、
已暂存新增源码等拒绝测试后，最终源码完整性专项远端 **12 passed**。
日志为 `iteration2-nmz7-source-tests.log`、`iteration2-source-extra.log`；
真实模型 CLI 对当前脏树在任何 GPU worker 启动前拒绝，并保存 30 文件差异
（`iteration2-source-rejection.json`）；L0 GREEN（31 warn）。

### DeepSeek 双流 graph 保护迁移

`moe_dualstream_gate` 不再写回 `deepseek_v2.py`。仅为原始 `forward`
提供私有的分支选择 predicate，保留原函数对象、签名和已有别名。其他函数
仍可见真实 capture 状态，不修改 `alt_stream` 等共享实例属性；原社区
`forward_normal` 自己接收 `skip_shared_experts`、返回类型和输入元数据。
DeepEP、mega-MoE、已有专门 graph 路径、空 batch、eager/breakable
分支不变。仅在真实 `torch.version.hip` 下安装，安装顺序移到算子桥之后，
避免为提前改文件而提前导入模型。对应磁盘补丁白名单已移除。

nmz7 隐藏 GPU 的控制流/广播/安装结构回归 **24 passed**，包含直接执行
社区 HEAD 原始 `DeepseekV2MoE.forward`（非手抄协议）以及早先绑定的方法
别名、公开 install、参数透传和全局 capture 状态不变。证据为
`results/l15-readiness/iteration3-moe-runtime.log`。这些是控制流证据，
不冒称完整 DeepSeek checkpoint、数值或 graph 端到端已重新通过。

独立干净工作树 `sglang-0.5.22-pristine-iter3` 在运行前完整性通过，
commit 为 `baa9befa381511cd4d56f83980951e5a45fffc6d`。
nmz7 物理 GPU 4 的 MLA dummy graph worker 启动/生成通过，
但总控正确判为失败：启动仍改写 **28 个文件**（此前为 30 个），
其中 `common.py` 和 `deepseek_v2.py` 均不再变化。
其余为 25 个 JIT C++/HIP 输入文件和 `fp8.py`、`deepseek_v4.py`、
`eagle_utils.py`。保留该工作树和失败证据，不回滚现场制造通过。
报告为 `iteration3-pristine-before.json`、`iteration3-pristine-mla.json`；
这不是源码零改写、多真实模型或精度验收通过的证据。

### Speculative KV-lens pinned 分配迁移

`spec_h2d_pin` 不再写回 `eagle_utils.py`，也不重编译原函数源码。
仅给原函数克隆提供私有 Torch 分配接口，将两个 CPU int32 元数据
tensor 直接分配为 pinned 内存；保留原函数对象、签名、已有别名和
社区调度/检查/H2D 函数体。Torch pinned 分配负责异步复制生命周期，
不再靠固定 8 深 ring 推测 DMA 已结束；pin 不可用时告警并使用社区
pageable 行为。原 helper 的兼容调用也移除不安全的 ring 复用。
安装移至算子桥之后，真正 CUDA 宿主与显式关闭时均不安装。

nmz7 隐藏 GPU 的最终组合回归 **45 passed，6 skipped，1 deselected**：
四个新增 GPU 参数项在隐藏 GPU 时跳过，另外两个 skip 分别为非 HIP
专用入口及子进程无 SGLang PYTHONPATH；未把它们算作通过。
CPU 算子契约另有 **38 passed**。此前空闲 GPU 4 上，原社区函数的真实
H2D、离线协议的真实 H2D，以及 48 波三流变长度异步拷贝均通过；
最终新增的旧 helper GPU 参数项尚待空闲卡复验。nmz7 后续检查
GPU 4–7 空闲显存均不足 3%，已停止进一步 GPU 分配。
日志为 `iteration4-spec-final-cpu.log`、`iteration4-ops-cpu.log`、
`iteration4-spec-real-h2d.log`、`iteration4-spec-h2d-gpu.log`。

获 nmz21 授权后，GPU 0 上完整 speculative 专项 **22 passed**，
补齐旧 helper 的多流异步拷贝参数项，包含直接执行 Git 原始函数、
真实 H2D、pin 回退、只读源文件、已有别名、spawn 子进程与漂移拒绝。
报告为 `iteration4-nmz21-spec-final.log`；此前“等待空闲卡”的缺项
现已补齐。此专项不等于完整 speculative 模型或全量 adapter 套件。

独立干净社区树 `sglang-0.5.22-pristine-iter4` 的 MLA dummy graph worker
通过，真实启动日志确认运行时桥已安装。总控因剩余 **27 个源码文件**
改写仍判失败（25 个 JIT 文件、`fp8.py`、`deepseek_v4.py`）；
`eagle_utils.py`、`common.py`、`deepseek_v2.py` 均保持原样。
证据为 `iteration4-pristine-before.json`、`iteration4-pristine-mla.json`。
启动日志另暴露 JIT warp needle/FP8 登记问题，后续迁移 JIT 时需一并处理。
这不代表完整 speculative 模型、所有 GPU 参数项或全量套件已通过。

nmz21 清理镜像残留后，另建干净社区树
`sglang-0.5.22-pristine-iter4-nmz21`，同一社区 commit、物理 GPU 0，
真实 `/model/qwen2/Qwen2-0.5B-Instruct` checkpoint 的 graph worker
启动及两批生成通过；实际捕获 prefill 与 decode graph。
总控因同样的 **27 个源码文件**改写仍判失败，Python 差异仅为
`fp8.py` 和 `deepseek_v4.py`。报告为
`iteration4-nmz21-pristine-before.json`、`iteration4-nmz21-checkpoint.json`。
该结果证明当前容器中的真实权重启动/生成链路，不代表语义金标、
完整 speculative 模型或源码零改写验收通过。

### FP4 专家加载与 block 配置迁移

`dsv4_ep_bf16`、`dsv4_fp4_dequant_blockshape` 不再写回 `fp8.py`，
兼容入口统一指向 `fp4_expert_load`。宿主桥在算子/vendor 安装之后
绑定原社区 `Fp8MoEMethod.process_weights_after_loading_block_quant`，
保留函数对象、签名、已有方法别名及原加载函数体。普通 FP4→FP8
转换仅使用该次调用的私有 cast 接口，不替换模块全局 cast。
真实 cast 产生 128×128 block 后，在原方法继续执行前对齐配置。

同时修复旧补丁的配置缺口：过去只改 `method.weight_block_size`，
后续 requant 与部分 runner 仍读取 `quant_config.weight_block_size=32×32`。
现在为当前方法复制 config 并设为 128×128，其他层共享的原 config
保持不变。非转换路径不改配置。`SGLANG_HCU_EP_BF16=1` 继续使用
社区 FP4 cast，再将两个专家矩阵按 block scale 转成 BF16；先算完两份
再替换参数、清理 scale，保留 DeepEP BF16 的显式选择语义。
该已有变量现纳入 env 注册表/手册；没有新增默认 BF16 转换。

nmz21 隐藏 GPU 的加载/广播/MoE/spec/结构组合 **72 passed，
10 skipped，1 deselected**，其中 8 个 skip 为 GPU 参数项，
另 2 个为非 HIP 专用与子进程未配置 SGLang 来源。原始社区加载函数、
FP4 cast 与专家 dequant helper 均从 Git HEAD 提取执行；
只读源文件、已有别名、关键字调用、spawn、配置隔离和漂移拒绝均有回归。
GPU 0 空闲检查后，转换专项 **4 passed**，包含社区原始函数和离线协议
的 FP8/BF16 两条路径；多专家、多 block、非单位 scale 与独立的
逐 32 元素 FP4 解码参照精确一致。日志为
`iteration5-fp4-final-cpu.log`、`iteration5-fp4-runtime-gpu.log`。
首轮夹具缺少上游 `convert_mxfp8_to_block` 初始化导致的失败保留在
`iteration5-fp4-runtime-cpu.log`，修正夹具后重新执行上述组合。

独立干净树 `sglang-0.5.22-pristine-iter5`，社区 commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`，nmz21 GPU 0：
真实 Qwen2-0.5B checkpoint graph worker 启动/两批生成通过，
但总控因 **26 个文件**改写仍判失败。`fp8.py` 现保持原样；
剩余是 25 个 JIT 文件及 `deepseek_v4.py`。
报告为 `iteration5-pristine-before.json`、`iteration5-checkpoint.json`。
此证据不等于真实 DSV4 checkpoint、DeepEP 多卡执行或全部精度验收。
L0 GREEN（31 项既有 warning），未执行全量 adapter/vendor 套件。

### DSV4 CT-int8 WQKV scale 加载迁移

`dsv4_wqkv_scale_patch` 不再读写或重编译 `deepseek_v4.py`。
仅将原 `load_weights` 两处 `name.endswith` 的 suffix 常量扩展为 tuple，
让 `.weight_scale` 走原 q/kv 融合分支；原函数对象、指令字节、签名、
已缓存别名、权重迭代及 PP/MTP、异步加载和错误检查保持原路径。
安装器验证签名与两处单参数调用位置，漂移拒绝；上游已有对应支持时不改 code。
移除该 Python 文件的 text-patch 白名单，安装放到算子/vendor 初始化之后。

nmz21 隐藏 GPU 的六模块组合回归 **97 passed、12 skipped、1 deselected**；
随后新增的 spawn 与上游原生支持加载回归 **3 passed**。测试包含从 Git
HEAD 提取的社区原始加载器与融合 helper、非单位 INT8 per-channel scales、
独立反量化投影、PP/MTP、真实线程池异步、未配对/重复分片及只读源码。
物理 GPU 1 的两臂加载与投影数值测试 **2 passed**；CPU 算子契约
**38 passed**。日志为 `iteration6-scale-runtime-cpu.log`、
`iteration6-scale-spawn.log`、`iteration6-scale-runtime-gpu.log`、
`iteration6-ops-cpu.log`，均位于远端 `results/l15-readiness/`。

GPU 0–3 后续出现占用，模型任务预检拒绝后重新检查并选择物理 GPU 4
（约 96.9% 空闲）。干净隔离树 `sglang-0.5.22-pristine-iter6` 启动前
完整性通过；真实 Qwen2-0.5B-Instruct graph worker 启动及两批生成通过、
退出码 0。总控仍正确判失败：运行后 **25 个 JIT C++/HIP 文件变化**，
没有 Python 文件变化，也没有新增源码。`deepseek_v4.py` 保持原样，
worker 日志确认运行时 scale 桥已安装。报告为
`iteration6-pristine-before.json`、`iteration6-checkpoint.json`。
L0 GREEN（31 项既有 warning），未运行全量 adapter/vendor 套件。
这是本次加载桥和真实 Qwen2 执行回归证据，不等于真实 DSV4 CT-int8
checkpoint 已验证，亦不等于整体源码零改写或多模型精度目标已完成。
下一步迁移剩余 JIT 编译输入，保留这些 portability 修复和数值语义。

## 原四步实施与验证记录

1. **严格执行与结果判定**：校验真实单卡 HIP、源码来源、版本绑定；
   每个必测 nodeid 必须实际通过，skip/xfail/缺失报告不能算通过。保留逐用例结果。
2. **完整计算链**：补 MoE 专家计算、量化 Linear、状态更新及真实变输入 Graph replay；
   证明调用 Triton 实现，并与独立数值参照对拍。
3. **模型最小闭环**：小模型真实执行链与本地真实权重启动/生成分账；
   单卡证据不为 TP/EP 背书，随机权重不为 checkpoint 加载背书。
4. **上游变更对账**：比较绑定符号、签名、补丁与 nodeid 漂移；
   固定基线与待适配版本分开记录，串接回归与模型冒烟。

每步验证后单独提交；测试失败保留原始证据，不放宽断言制造绿色。
GPU 测试在用户授权的 nmz7/nmz21/nmz28 的 `sglang_liph_adapter` 容器运行。
本轮代码目录为 `/public/home/lipenghui/lph-adapter/adapter-das`；
社区源码为同级 `sglang-0.5.22`。

## 严格组件门

```bash
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  bash ci/l15_readiness_gate.sh /home/lipenghui/lph-adapter/sglang-0.5.22
```

该入口强制实际执行五家族矩阵，不继承旧门禁的 observe/off 或失败总数预算。
它验证源码指纹、导入来源和真实 HIP 分配，输出 GPU/软件版本、两个仓库的
commit/dirty 状态和逐 nodeid 阶段结果。报告缺失、所有 skip、xfail、xpass、
setup/teardown 异常及超时均不通过；运行前删除旧逐 case 报告，预检失败覆盖旧汇总。

`coverage=cpu-contract` 的 sparse emu 项仅证明 CPU 数学契约，不计 GPU 计算覆盖。
组件门通过不等于真实模型已通过；后续模型闭环需另有启动/生成证据。

## 第一步验证（2026-11-03）

- nmz28 / `sglang_liph_adapter` / 单卡 GPU 0：Dense prefill/decode、
  fragmented layout、graph metadata 三个严格 case 全通过，保留逐 subtest
  结果，passing parent 不能掩盖子用例 skip。
- 同容器隐藏 GPU 后，矩阵与清单回归 **67 passed**；L0 **GREEN（31 warn）**。
- 原始报告：`results/l15-readiness/phase1-dense.json`（节点本地生成，不入版本库）。
- 这是严格执行机制和 Dense 的证据；其余家族与模型启动尚待后续步骤验证。

## 第二步验证（2026-11-03）

- 提交 `5cb6c11`：五家族严格矩阵 **27/27 PASS**（25 GPU case + 2 CPU 契约）。
- 社区 MoE routing 接入后，计算链与 GDN 桥另跑 **13 passed**。
- 实际发现并修复 GDN bridge 多余转置；非方形状态、envelope stride、
  禁写、ring 和三轮 graph replay 均通过独立 FP32 对拍。
- 报告：`results/l15-readiness/phase2-all.json`；
  最终链路日志：`results/l15-readiness/phase2-final-chains.log`。

## 模型最小闭环入口

```bash
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 HCU_ADAPTER_HOST=none \
  python3 -m tools.l15_model_smoke \
    --up-tree /path/to/sglang \
    --checkpoint /path/to/local-checkpoint
```

默认顺序执行 Dense、MoE、MLA、Mamba2、GDN hybrid 的两层 dummy 模型，
以及显式本地 safetensors checkpoint。每个模型分别启动 eager 和 graph，
经真实 Engine/scheduler 执行两批变输入，每批两个请求，每请求生成 8 token；
检查词表范围和有限 token logprob。graph 必须从 scheduler 读回真实 decode
graph 分配量，日志另保留 batch 1/2 的捕获过程；当前上游启动路径没有填写
role-switch 使用的 capture-bs 列表，不能以空列表误判未捕获。
结果保留 resolved server info、原始输出及每次启动日志。
该验证检查执行闭环，不把 dummy 的随机输出当语义金标，也不冒称 HTTP、
多卡 TP/EP、所有模型或所有量化格式已经可用。

每次启动使用独立 Linux 进程组。正常退出或超时后仅清理本次任务所属进程组，
避免失败的 scheduler 占住卡影响下一项。`--families` 可选择子集；
`--binding-manifest` 需与审查后的目标版本绑定一致。

## 第三步验证（2026-11-03）

- nmz28 单卡 GPU 0：Dense、MoE、MLA、Mamba2、GDN hybrid 五类两层
  dummy 模型，以及真实 `Qwen2-0.5B-Instruct` checkpoint，分别完成 eager
  与 graph 启动/生成，共 **12/12 PASS**。真实权重来自容器
  `/model/qwen2/Qwen2-0.5B-Instruct`，包含 safetensors 与 tokenizer。
- 每项两批、每批两请求、每请求 8 token；最终规则复核词表范围、有限
  logprob 与 token 对齐、eager 无 graph 分配及 graph 实际分配。六项 graph
  的 decode 分配均大于零；这不是只检查启动参数。
- 模型探测暴露旧 MLA 默认 `rope=0` 的覆盖缺口：新增受上游支持的
  `256+32`、`512+64` KV 布局，prefill/decode/真实 graph 三种模式，
  **6/6 参数化节点通过**。当前矩阵为 28 case；证据为第二步完整
  27 case 加本次新增 case 的增量验证，未声称重新完整跑过 28 case。
- MoE tiny 配方采用上游 fused RoPE 支持的 head64；MLA tiny 配方采用
  `512+64`，不再使用上游 grouped decode 不支持的任意小尺寸。
  这些是验证配方修正，未修改社区实现或放宽数值断言。
- 报告位于 `results/l15-readiness/`：`phase3-dense.json`、
  `phase3-dense-graph.json`、`phase3-models.json`、
  `phase3-models-retry.json`、`phase3-mla-final.json`，
  新 MLA 契约为 `phase3-mla-contracts.json`。旧失败保留在原报告，
  最终验证按 family/mode 选择成功执行，并用最终规则重新核验原始输出。
- 工具与失败判定回归 **91 passed**；包含真实 Linux 子进程及嵌套超时
  清理、空原因 XPASS 拒绝。未重新执行全量 adapter/vendor 套件。

## 第四步验证（2026-11-03）

- `ci/upstream_diff.py` 对不可变 Git 快照比较已声明的符号、签名/函数体、
  白名单补丁 needle 的位置/次数、上游测试 nodeid 与版本绑定文件。
  新路径复制同一个 needle 也要求审查。
- `tools.l15_upgrade_check` 串接静态审查、严格组件门、模型闭环；陈旧或
  其他 upstream commit 的报告不能使升级就绪，审查接受文件必须精确匹配
  baseline/candidate/changes。未完成全部阶段时保持 `ready=false`。
- nmz28 实际运行 `--baseline-ref HEAD --candidate-ref HEAD --audit-only`：
  **0 changes、15 baseline gaps、19 dynamic registration sites**；
  `same_revision=true`、`audit_only=true`、`ready=false`。这些缺口需运行时
  验证，不能将零差异解释为所有绑定完整或完成一次版本升级。
- 同容器工具组合回归 **91 passed**；补充 needle 新路径检查后，本地仅跑
  漂移/编排 CPU 测试 **12 passed**。L0 **GREEN（31 warn）**。
  故意修改签名、函数体、删除符号/nodeid/needle 的回归均能拒绝。
- 证据：`results/l15-readiness/phase4-workflow-audit.json`、
  同名 `-stages/upstream-diff.json`、`final-tool-tests.log`、`final-l0.log`。
  本次交付手动升级工作流，未配置定时巡检，也未验证新社区版本。
