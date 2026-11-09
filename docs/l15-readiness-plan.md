# HCU SGLang 启动就绪门禁执行计划

目标：社区 SGLang 源码不变，通过 adapter 的 Triton 功能基线与模型执行链回归，
为明确的模型及配置提供启动依据。不以性能作为本轮验收条件。

原始运行报告、日志和环境坐标保存在仓库外，禁止提交真实节点、
地址、账号、容器名及个人目录。下文 `external-evidence:` 表示外部
归档标识，`validation-host-*` 为匿名历史标签，不是可连接的节点。
这些历史证据只证明各自绑定的版本与配置；迁移到新上游基点后须重新验证，
不能将历史通过计数作为新组合的验收结果。

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

真实 checkpoint 的 TP 参数、逐设备预检和精度报告绑定方式见
[真实模型 TP 配置验收](l15-model-smoke.md)。

2026-11-03 用户新增 validation-host-a 调试授权：validation-host-a 与 validation-host-c 都使用
`adapter-validation` 容器，代码目录相同。validation-host-a 经资源检查确认 GPU
4–7 空闲、0–3 在用；首次回归选择物理 GPU 4，只暴露这一张卡。
每次任务启动前重新检查空闲状态，结果单独记录节点和设备，不沿用
validation-host-c 的环境结论；不清理或占用其他用户任务。

2026-11-03 用户进一步授权 validation-host-b 的同名容器。经 validation-host-a 免密跳转确认
代码目录相同，8 张 BW1100 均约 97% 空闲显存，本轮仅暴露物理 GPU 0。
清理该新容器镜像残留的 SGLang 0.5.18 DTK 定制包与 kernel 包，
核对 SGLang 从社区源码、`sgl_kernel` 从 adapter 导入；不更改其他节点环境。

2026-11-04 用户再次确认 validation-host-b 可用于调试。三个节点的授权持续有效，
仍在每次作业前检查资源；本次实查 validation-host-b 容器正常、计算利用率为 0，
但 8 张卡显存占用 88%–89%，因此没有在该节点启动新 GPU 作业。
不能把计算利用率为 0 当作显存空闲，也不能沿用之前的空闲检查。

历史前置差距包括 `jit_*`、`dsv4_*` 的源码磁盘写回，
现已逐项迁移；Qwen2 当前组合通过严格源码完整性检查，其他模型与配置
仍须逐项验证，上述完整目标尚未完成。对象广播 `dist_bc` 已迁移为运行时只读 numpy buffer
兼容，保留已导入函数别名，不再写回 `common.py`。validation-host-c 隐藏 GPU 的
回归为 **6 passed**：包括以 Git 原始社区广播函数执行两 rank spawn/Gloo，
验证双向广播、空/中文/bytes/大对象和测试源文件字节不变。
这是广播路径的迁移证据，尚不代表整个 adapter 启动不再修改源码。
回归日志为 `external-evidence:pristine-dist-bc.log`；本地 CPU 契约
4 passed（需远端社区树的 Gloo 项在本地不运行），L0 GREEN（31 warn）。
真实 checkpoint eager/graph 复验在启动前因 GPU 0 仅约 4% 空闲而被阻止，
未运行，也未占用其他卡；该集成复验仍待空闲窗口补齐。

新增 validation-host-a 授权后，在空闲物理 GPU 4 补齐真实 Qwen2-0.5B checkpoint
eager/graph 回归，**2/2 PASS**；报告为
`external-evidence:iteration1-validation-host-a-checkpoint.json`。此次使用旧完整性
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

validation-host-a 审计当前共享社区树：**10,594 tracked files，30 changed files，
0 added sources**。报告为 `external-evidence:iteration2-upstream-source.json`。
这说明原样源码目标尚未达成；不通过 reset/删除现有工作树改动制造绿色，
后续在保留现场的前提下迁移适配并从干净社区树验证。
门禁与广播、编排工具组合远端 CPU 回归 **68 passed**；补充旧报告覆盖、
已暂存新增源码等拒绝测试后，最终源码完整性专项远端 **12 passed**。
日志为 `iteration2-validation-host-a-source-tests.log`、`iteration2-source-extra.log`；
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

validation-host-a 隐藏 GPU 的控制流/广播/安装结构回归 **24 passed**，包含直接执行
社区 HEAD 原始 `DeepseekV2MoE.forward`（非手抄协议）以及早先绑定的方法
别名、公开 install、参数透传和全局 capture 状态不变。证据为
`external-evidence:iteration3-moe-runtime.log`。这些是控制流证据，
不冒称完整 DeepSeek checkpoint、数值或 graph 端到端已重新通过。

独立干净工作树 `sglang-0.5.22-pristine-iter3` 在运行前完整性通过，
commit 为 `baa9befa381511cd4d56f83980951e5a45fffc6d`。
validation-host-a 物理 GPU 4 的 MLA dummy graph worker 启动/生成通过，
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

validation-host-a 隐藏 GPU 的最终组合回归 **45 passed，6 skipped，1 deselected**：
四个新增 GPU 参数项在隐藏 GPU 时跳过，另外两个 skip 分别为非 HIP
专用入口及子进程无 SGLang PYTHONPATH；未把它们算作通过。
CPU 算子契约另有 **38 passed**。此前空闲 GPU 4 上，原社区函数的真实
H2D、离线协议的真实 H2D，以及 48 波三流变长度异步拷贝均通过；
最终新增的旧 helper GPU 参数项尚待空闲卡复验。validation-host-a 后续检查
GPU 4–7 空闲显存均不足 3%，已停止进一步 GPU 分配。
日志为 `iteration4-spec-final-cpu.log`、`iteration4-ops-cpu.log`、
`iteration4-spec-real-h2d.log`、`iteration4-spec-h2d-gpu.log`。

获 validation-host-b 授权后，GPU 0 上完整 speculative 专项 **22 passed**，
补齐旧 helper 的多流异步拷贝参数项，包含直接执行 Git 原始函数、
真实 H2D、pin 回退、只读源文件、已有别名、spawn 子进程与漂移拒绝。
报告为 `iteration4-validation-host-b-spec-final.log`；此前“等待空闲卡”的缺项
现已补齐。此专项不等于完整 speculative 模型或全量 adapter 套件。

独立干净社区树 `sglang-0.5.22-pristine-iter4` 的 MLA dummy graph worker
通过，真实启动日志确认运行时桥已安装。总控因剩余 **27 个源码文件**
改写仍判失败（25 个 JIT 文件、`fp8.py`、`deepseek_v4.py`）；
`eagle_utils.py`、`common.py`、`deepseek_v2.py` 均保持原样。
证据为 `iteration4-pristine-before.json`、`iteration4-pristine-mla.json`。
启动日志另暴露 JIT warp needle/FP8 登记问题，后续迁移 JIT 时需一并处理。
这不代表完整 speculative 模型、所有 GPU 参数项或全量套件已通过。

validation-host-b 清理镜像残留后，另建干净社区树
`sglang-0.5.22-pristine-iter4-validation-host-b`，同一社区 commit、物理 GPU 0，
真实 `/model/qwen2/Qwen2-0.5B-Instruct` checkpoint 的 graph worker
启动及两批生成通过；实际捕获 prefill 与 decode graph。
总控因同样的 **27 个源码文件**改写仍判失败，Python 差异仅为
`fp8.py` 和 `deepseek_v4.py`。报告为
`iteration4-validation-host-b-pristine-before.json`、`iteration4-validation-host-b-checkpoint.json`。
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

validation-host-b 隐藏 GPU 的加载/广播/MoE/spec/结构组合 **72 passed，
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
`baa9befa381511cd4d56f83980951e5a45fffc6d`，validation-host-b GPU 0：
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

validation-host-b 隐藏 GPU 的六模块组合回归 **97 passed、12 skipped、1 deselected**；
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

### JIT 编译输入迁出社区源码（GPU 集成待复验）

`jit_sources` 将现有八组 portability 变换集中在 adapter 私有 staging
目录执行，仅复制 JIT `csrc/include` 编译资产，排除独立 Python 生成脚本。
输入文件字节、变换实现及 forwarder 定义共同生成缓存 key；全部变换、
白名单和输出哈希通过后原子发布到
`~/.cache/hcu_adapter/jit-inputs/<sha256>`。同 key 并发发布只能读到完整
bundle；已有缓存损坏拒绝，不覆盖“自愈”。manifest 记录输入、输出和变换。
CUDA→HIP 头文件也移入 bundle，不再写 DTK include。

社区 `resolve_sources` 与原 `load_jit` 的函数对象保留，已缓存别名和
Torch 包装仍调用原对象。以私有调用命名空间映射相对/绝对 source，
以及默认、显式和实时 dependency provider 的 include；原
BuildSpec/ninja/cache、编译参数及依赖追踪仍在使用，不转移社区 Python
模块或改写 `__file__`。量化变换先生成 packed BF16 shuffle 分支，再由
warp 变换把接收类型修正为 `T`，避免旧顺序造成首轮安装缺针。
补齐已有 v2 FP8 conversion needle 的治理登记，不删除 portability 修复。

validation-host-b 隐藏全部 GPU 的最终缓存/环境/启动结构及 CPU 算子组合回归
**75 passed、5 skipped、1 deselected**（`iteration7-final-cpu.log`）。
其中 source 两臂分别
使用离线接口及 Git HEAD 原始社区资产/函数；spawn 子进程实际安装文件
写入 audit hook，先验证写入社区目录会被拒绝，再并发生成同一 bundle。
包含真实官方 BuildSpec→ninja→cache→tvm-ffi **CPU 编译 2 passed**
（另有独立 `iteration7-host-compiler.log`）：从缓存编译 source 和传递 include，
两轮改变头文件值后返回值随之改变，证明缓存失效与真实编译路径。
另隐藏 GPU，以明确 `gfx938` 编译目标实例化真实 HIP wrapper：
RoPE NeoX/interleaved 及 FP8 group32/group128 **4 passed**
（离线合成资产四臂按设计 skip），日志为
`iteration7-hip-template-compile.log`。未运行这些 device 函数，
这不是 GPU 数值验证。L0 GREEN（31 项既有 warning）。

实际社区编译器绑定与重复安装通过（`iteration7-actual-compiler-bind.log`）。
干净 `sglang-0.5.22-pristine-iter7` 隐藏 GPU 后执行宿主导入，
前后源码完整性为 **0 changed、0 added sources**，报告为
`iteration7-pristine-before.json`、`iteration7-bootstrap-source-after.json`。
隐藏 GPU 导致部分其他可选桥无法取得设备 capability，导入日志保留相关
warning；该结果只证明本次导入没有源码写回，不作为功能就绪证据。

此前 validation-host-b 多次显存预检不足，确认有其他用户的 8 卡
`all_reduce_perf` 压测，未停止其任务。节点释放后重新预检，
物理 GPU 0 空闲显存比例 0.969343，再补齐设备验证。

### JIT FP8 输出存储接口与源码不变的真实 Qwen2 回归

首轮 GPU 测试中 RoPE 两种布局通过，FP8 分组量化在 FFI 边界失败：
DTK DLPack 暴露名义 `float8_e4m3fn`，社区 HIP kernel 要求 `uint8`
存储。原始失败日志保留为 `iteration7-jit-inputs-gpu.log`。
`jit_fp8_ffi` 现在覆盖 generic/v2 与已有 DSV4 两个 factory，仅在
调用编译导出时对输出做零拷贝 uint8 view；输入、scale、其他 dtype
及 factory 的名义 dtype 选择不变。保留原 `cache_once` 函数对象与
已有 factory 别名，适配 warm cache 与后续 cold key；布局漂移拒绝。
原有 `HCU_JIT_FP8_FFI` 补入环境注册表，显式 0 在启动时跳过安装。

validation-host-b 隐藏 GPU 的最终接口/环境/启动顺序回归 **25 passed、3 skipped**
（`iteration8-fp8-ffi-cpu-final.log`，排除另行探测的 4 个 DSV4 GPU 项）。
包括离线及 Git 原始社区 factory/cache、指针和 stride、warm/cold
别名、spawn、非目标 dtype、快关与失败原子性。
RoPE 两布局及 generic/v2 FP8 group32/128 的真实编译、独立参考与
三轮变输入 graph 共 **6 passed**（`iteration8-jit-inputs-gpu.log`）。

真实 `/model/qwen2/Qwen2-0.5B-Instruct` 在
`sglang-0.5.22-pristine-iter7`、社区 commit
`baa9befa381511cd4d56f83980951e5a45fffc6d` 上 eager/graph **2/2 passed**，
报告 `iteration8-qwen2-checkpoint.json`。每模式两批两请求各生成 8 token，
检查词表、有限 logprob 及实际 graph 分配；控制器核对 10,594 个跟踪文件，
最终 **0 changed、0 added sources**。这证明该组合的加载和生成以及源码
不变，不代表模型精度已验收。此前 25 个 JIT 文件写回失败报告继续保留。

额外 DSV4 连续/变长布局 × group32/128 数值探测发现 **4 failed**：
编译及 FFI 已通过，scale 对拍通过，FP8 输出值失败。原始日志
`iteration8-dsv4-ffi-gpu.log` 保留；初步定位共享软件转换的最高指数段
过早饱和，待独立修复，不能以 generic/v2 的通过代替 DSV4 数值验收。

### DSV4 gfx938 FP8 最高指数段转换修复

已确认社区 `deepseek_v4/fp8_utils.cuh` 的软件转换在 `exp8 >= 15`
时直接饱和，使 OCP E4M3 的合法最高段（256～448）过早变为 448。
`jit_quant_fp8` 在 adapter 编译输入缓存内仅把 gfx938 纳入现有
DTK OCP 转换分支，原始头文件字节不变；其他架构继续原有分支，
尤其不修改 gfx942 的 FNUZ 语义。新增 needle 仍经白名单登记和缓存哈希。

validation-host-b GPU 0 对原失败用例原样复测，连续/变长布局 × group32/128
**4 passed**（`iteration9-dsv4-ffi-gpu.log`）。每项由原社区 public API
实际编译，独立 FP32 SiLU×mul/分组 scale/FP8 参考对拍，三轮改变输入
幅度和专家有效 token 数后 replay；有效行检查 scale、逐元素误差和
cosine，未放宽首轮失败断言。隐藏 GPU 的 source/FFI 两臂回归
**25 passed**（`iteration9-jit-cpu.log`）；L0 GREEN（31 既有 warning）。
这是该设备上的算子证据，尚不代表真实 DSV4 checkpoint 或完整模型精度。

### 真实 Qwen3 与可重复 CPU FP32 参考探针

validation-host-b 物理 GPU 1 重新预检空闲显存比例 0.969343，
真实 `/model/qwen3/Qwen3-0.6B` eager/graph **2/2 passed**，
报告 `iteration9-qwen3-checkpoint.json`。同一 pristine-iter7 社区树，
两批两请求每请求 8 token，控制器源码完整性 **0 changed、0 added sources**。
硬件 BW1100/gfx938，Torch 2.11.0、HIP 6.3.26113。

配方参考固定为 `HYGON-AI/inference-cookbook-das` commit
`40adc4d30ec2640e3391cfd78759bd3598e95f71`，
`docs/model-deployment/sglang/qwen3.md` 的
“Qwen3-0.6B IFB BW1100 1x SGLang 0.5.12” 小节。保留单卡、BF16 和
同一 checkpoint 选择；本轮为了验证社区 0.5.22 的 adapter 计算链，
使用 Engine API、Triton attention、mem_fraction_static=0.25、
context_length=128、max_total_tokens=512，并分别启用 eager/graph。
配方原为 0.5.12、fa3、mem_fraction_static=0.9、page_size=64、关闭
graph 和 overlap、跳过 warmup，且附带 HTTP 工具/推理解析器及定制环境。
实际 page_size=1、trust_remote_code=False、overlap 与 warmup 启用，
tool_call_parser/reasoning_parser 均为 None。
这些部署项未按原配方逐一验收，本报告只覆盖实际 Engine 参数；
未执行配方中的 sysctl 或套用其定制环境变量。

新增可重复参考入口（远端容器内、隐藏 GPU）：

```bash
HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
  python3 -m tools.l15_checkpoint_reference \
  --smoke-report external-evidence:iteration9-qwen3-checkpoint.json \
  --out external-evidence:iteration10-qwen3-reference.json
```

工具要求严格源码检查成功、真实 checkpoint、eager/graph 两模式完整
worker、两批两请求和完整 uncached prompt IDs，拒绝 dummy/失败/缺失/重复
证据。独立 Transformers 5.12.1 的 CPU FP32 eager 模型对同一
prompt+生成 token 序列 teacher-force，记录逐位置参考 logprob、绝对误差
及相对参考 top1 的 greedy regret，保留 BF16 平局附近的差异。运行前后
校验本轮参考权重/JSON 文件 SHA256 与输入 smoke 报告字节，禁止覆盖原报告。

Qwen2/Qwen3 各 **64 个生成位置**：Qwen2 最大 logprob 绝对差
**0.053874**，最大 regret **0**，top1 匹配 64/64；
Qwen3 最大绝对差 **0.055951**，最大 regret **0.024235**，
top1 匹配 62/64。报告为 `iteration10-qwen2-reference.json`、
`iteration10-qwen3-reference.json`。这是对所选 token 概率的独立短输入
探针，报告标记 `exploratory-short-prompts`，不输出精度验收 `passed`。
历史 smoke 没有绑定 HCU 执行时的权重字节哈希，因此当前参考时间点的
fingerprint 不能倒推证明此前 HCU 使用的权重完全一致。
最终参考工具与原 smoke 失败契约回归 **23 passed**
（`iteration10-reference-final-tests.log`），含 NaN、dummy、源码失败、
重复 mode、缺波次/缓存 prompt、错误概率/非 greedy token、权重/配置变更及
原报告防覆盖；L0 GREEN（31 既有 warning）。
后续须补同轮 checkpoint 字节绑定、更全面的输入/长度/分布和模型族覆盖，
才能提升为完整精度门禁。

### 将 checkpoint 文件指纹绑定到实际模型执行

`tools.checkpoint_files` 以流式 SHA256 读取 safetensors 与配置/tokenizer
资产，不依赖 mtime 缓存；校验索引中的每个分片存在且路径合法。
哈希时同时检查文件身份和最终清单，已经读完的文件在后续哈希中被改动也拒绝。
控制器在任何 worker 启动前固定原始指纹，并把清单交给真实 checkpoint
worker；worker 在 Engine 加载前、关闭后分别重新读取字节，控制器结束每个
worker 后再次核对。缺少 worker 指纹、内容不一致、分片缺失均不能通过，
检查失败保留该 job 和已完成结果，文件变化后停止继续启动模型。

CPU 参考运行前要求新报告的 controller/worker 清单一致，
再把当前权重字节与 HCU 执行时清单核对，结束后再次检查。
一致的报告标记 `checkpoint_binding=matched-hcu-execution`；
历史未绑定报告继续标记 `legacy-unbound`，保留其证据局限。
字节绑定本身不把短输入探针提升为完整精度验收。

validation-host-b 隐藏 GPU 的绑定/参考/执行失败契约 **32 passed**
（`iteration11-checkpoint-binding-cpu.log`），覆盖缺失/越界分片、哈希期间
已读文件被修改、加载前与执行期间权重变化、旧 worker 缺少指纹、
控制器停止后续启动及参考端 controller/worker 不一致。
L0 GREEN（31 既有 warning）。
GPU 1/2 分别重新运行真实 Qwen2/Qwen3，每模型 eager/graph **2/2 passed**，
每次 worker 结束后的源码完整性均为 **0 changed、0 added sources**。
对应 `iteration11-{qwen2,qwen3}-bound.json` 和
`iteration11-{qwen2,qwen3}-reference.json`；每模型 8 个资产文件匹配，
两份参考报告均为 **matched-hcu-execution**。
每模型 64 个生成位置的最大 logprob 差分别为 **0.053874 / 0.096956**，
最大 greedy regret 为 **0 / 0.024235**；这次证据补齐同轮文件绑定，
仍保留短输入/所选 token 概率的覆盖边界。

### 固定多长度输入与 top-20 + tail 精度策略

真实 checkpoint 的 `--profile accuracy` 使用三批两请求：
中英文短问答、精确 512/2048 token 文档、复用同一 512 token 输入并改变
2048 token 文档中的访问码。worker 记录实际输入 ID 和 case 标签，
请求原社区 API 返回每个生成位置的 top-20 logprob；
context_length=4096、max_total_tokens=16384，仍为单卡 BF16/Triton、
mem_fraction_static=0.25，eager/graph 各执行全部六个请求。

CPU FP32 eager 参考对相同 prompt+输出序列 teacher-force，比较所选 token
logprob、greedy regret、同一组 20 个 token 的 logprob，以及这 20 类加
其余词表概率质量构成的分布。额外检查参考 top-20 被遗漏的概率质量，
防止任取 20 个低概率 token 却得到零差异。指标允许近似平局的名次重排。
这是一种明确的分布合并，不称作完整词表 KL/JSD。
固定策略 `bf16-cpu-fp32-top20-v1` 的门槛为：
所选 token 与 top-20 的最大 logprob 差各 0.15、最大 greedy regret 0.15、
合并分布 JSD 0.002、tail 质量差 0.03、参考头部遗漏质量 0.02；
逐位置比较后取最大值，NaN/Inf 或缺失项直接拒绝。
只有 case、源码检查、eager/graph、同轮权重绑定和所有数值条件均完整，
才输出 `l15-checkpoint-accuracy` 的 `passed`；普通 smoke 仍为探索报告。

```bash
HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 \
  python3 -m tools.l15_model_smoke \
  --up-tree /path/to/sglang-0.5.22-pristine-iter7 \
  --families checkpoint --checkpoint /model/qwen3/Qwen3-0.6B \
  --profile accuracy --modes eager,graph \
  --out external-evidence:accuracy-generation.json
HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
  python3 -m tools.l15_checkpoint_reference \
  --smoke-report external-evidence:accuracy-generation.json \
  --out external-evidence:accuracy.json
```

首轮 `iteration12-{qwen2,qwen3}-accuracy-generation*` 失败保留：
Transformers 5.12 的聊天模板默认返回字典，导致请求传入字符串 key。
现在显式 `return_dict=False`，并在创建 Engine 前验证 token 列表和 case。
模拟该真实返回协议的回归已补入；最终 CPU 组合 **43 passed**
（`iteration13-accuracy-cpu.log`），两个真实 tokenizer 的独立预检也通过
（`iteration13-real-tokenizer-cases.log`）：短输入长度 Qwen2=31/30、
Qwen3=24/23，其他两批均为 512/2048。

validation-host-b GPU 1/2 的 `iteration13-{qwen2,qwen3}-accuracy-generation.json`
均完成 eager/graph **2/2 生成通过**，每模式三批两请求各 8 token；
社区源码和 checkpoint 字节检查均通过。
独立 CPU FP32 参考各比较 **96 个生成位置**，两份
`iteration13-{qwen2,qwen3}-accuracy.json` 的最终 verdict 均为 **FAIL**，
命令退出 1，未放宽 V1 阈值：

- Qwen2：所选 logprob 差 0.070135、greedy regret 0.008249；
  top-20 logprob 差 **0.546768** 超限；合并 JSD 0.001373、
  tail 质量差 0.009165、参考头部遗漏质量 0.000553。
- Qwen3：所选 logprob 差 0.041764、greedy regret 0；
  top-20 logprob 差 **0.732415** 超限；合并 JSD 0.001123、
  tail 质量差 0.009017、参考头部遗漏质量 0.000101。

V1 是保留失败的初始诊断策略，BF16 舍入基线尚待校准。
进一步在相同短问答和 token 序列上，将独立 CPU eager 模型从 FP32
转换为 BF16 再计算，原始诊断为
`iteration14-dtype-reference-probe.{json,log}`。
CPU BF16→FP32 在 HCU 所报头部 token 上的最大差：
Qwen2 英文/中文 **0.643486 / 0.815510**，
Qwen3 **0.791748 / 0.530690**；HCU→CPU BF16 的最大差仍为
约 0.50～0.75。后续 iteration18 检查发现，整体转换还会舍入
官方 FP32 RoPE buffer，因此该历史实验不能作为纯 dtype 舍入基线，
其原始记录保留但不用于归因结论；以独立官方加载的 iteration18 为准。
当前证据不足以把 V1 的失败直接归为 HCU 实现错误。
下一步需扩展独立舍入基线、逐层/逐位置归因，再确定适用 BF16 的精度策略；
当前报告继续保持失败，不能称作多模型精度验收通过。

### 真实 Qwen3.5 混合架构 checkpoint

验证工具新增复合配置支持：从 `text_config` 取得实际文本词表，校验
正整数且不把外层/视觉元数据当成文本词表；包含视觉配置的条件生成
checkpoint 使用 Transformers 官方 `AutoModelForImageTextToText`。
原 CausalLM 仍通过原公开 AutoModel 入口加载，参考端同时拒绝外层或
文本配置中的未支持量化。报告记录实际模型类与文本 model_type。
CPU 配置/词表/模型执行/精度诊断组合 **51 passed**
（`iteration15-composite-cpu.log`），L0 GREEN（31 既有 warning）。

validation-host-b 物理 GPU 3（预检空闲显存比例 0.969343），
`adapter-validation`，真实 `/model/Qwen/Qwen3.5-2B-Base`，
社区 pristine-iter7 commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`，
eager/graph **2/2 passed**。模型包含 24 层（18 线性注意力、6 完整注意力），
本轮验证该条件生成 checkpoint 的纯文本路径。
每模式两批两请求各 8 token，报告 `iteration15-qwen35-bound.json`；
两次结束后的源码完整性均 **0 changed、0 added sources**，
10 个 checkpoint 资产指纹一致。
`iteration15-qwen35-reference.json` 的 CPU FP32 eager
`Qwen3_5ForConditionalGeneration` 采用独立 Torch fallback，
权重为 **matched-hcu-execution**。64 个生成位置的最大 logprob 差
**0.063152**，最大 greedy regret **0**，参考 top1 匹配 **64/64**。
这补齐真实混合架构的加载/生成和短输入对拍，不为图像、NEXTN、量化、
多卡或该模型的完整多长度精度背书。

Cookbook 来源固定为 commit
`40adc4d30ec2640e3391cfd78759bd3598e95f71`，
`docs/model-deployment/sglang/qwen3.5.md` 的
“Qwen3.5-27B IFB BW1100 2x SGLang 0.5.18” 家族配方；
该配方没有本轮 2B-Base 的同名条目，明确采用较小 checkpoint 验证家族链路。
实际为社区 0.5.22、Engine、TP1、BF16、Triton 普通/线性注意力、
page_size=1、KV dtype=auto、mamba cache=extra_buffer、
context_length=128、max_total_tokens=512、mem_fraction_static=0.25，
分别 eager/graph；与原配方的 TP2/fa3/page64/FP8 KV/NEXTN 存在差异，
不套用其定制环境或声明原部署配方逐项通过。

### checkpoint 哈希结束时的字节复核

后续 CPU 组合复现哈希负控偶发漏检：config 在读取后由 256 改为 512，
等长写入若处于同一时间戳粒度，stat 身份可能没有可见差异。
失败保留在 `iteration16-lm-head-option-cpu.log`。
`fingerprint` 现在结束时再次流式读取全部资产并比较 SHA256，
同时重新核对清单，不仅依靠 inode/mtime/ctime。
回归包含真实元数据与强制完全相同元数据两臂，均在
`iteration16-final-cpu.log` 中通过。额外读取不改变模型文件、内存占用仍按块限制；
本阶段优先保证权重绑定检查，不以 I/O 性能作为验收条件。

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
GPU 测试在用户授权的 validation-host-a/validation-host-b/validation-host-c 的 `adapter-validation` 容器运行。
本轮代码目录为 `/public/path/to/adapter-das`；
社区源码为同级 `sglang-0.5.22`。

## 严格组件门

```bash
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  bash ci/l15_readiness_gate.sh /path/to/sglang-0.5.22
```

该入口强制实际执行五家族矩阵，不继承旧门禁的 observe/off 或失败总数预算。
它验证源码指纹、导入来源和真实 HIP 分配，输出 GPU/软件版本、两个仓库的
commit/dirty 状态和逐 nodeid 阶段结果。报告缺失、所有 skip、xfail、xpass、
setup/teardown 异常及超时均不通过；运行前删除旧逐 case 报告，预检失败覆盖旧汇总。

`coverage=cpu-contract` 的 sparse emu 项仅证明 CPU 数学契约，不计 GPU 计算覆盖。
组件门通过不等于真实模型已通过；后续模型闭环需另有启动/生成证据。

## 第一步验证（2026-11-03）

- validation-host-c / `adapter-validation` / 单卡 GPU 0：Dense prefill/decode、
  fragmented layout、graph metadata 三个严格 case 全通过，保留逐 subtest
  结果，passing parent 不能掩盖子用例 skip。
- 同容器隐藏 GPU 后，矩阵与清单回归 **67 passed**；L0 **GREEN（31 warn）**。
- 原始报告：`external-evidence:phase1-dense.json`（节点本地生成，不入版本库）。
- 这是严格执行机制和 Dense 的证据；其余家族与模型启动尚待后续步骤验证。

## 第二步验证（2026-11-03）

- 提交 `5cb6c11`：五家族严格矩阵 **27/27 PASS**（25 GPU case + 2 CPU 契约）。
- 社区 MoE routing 接入后，计算链与 GDN 桥另跑 **13 passed**。
- 实际发现并修复 GDN bridge 多余转置；非方形状态、envelope stride、
  禁写、ring 和三轮 graph replay 均通过独立 FP32 对拍。
- 报告：`external-evidence:phase2-all.json`；
  最终链路日志：`external-evidence:phase2-final-chains.log`。

### 官方 FP32 输出层单变量诊断（2026-11-03）

`tools.l15_model_smoke --enable-fp32-lm-head` 显式转发社区公开 Engine
选项，默认不传入该选项。controller、worker、实际 server info 与 CPU
参考报告绑定同一配置，拒绝忽略选项或混用旧 worker 证据。此功能用于归因，
不改变 adapter 默认行为或 `bf16-cpu-fp32-top20-v1` 的任一阈值。

validation-host-b 同一容器、同一社区 commit，Qwen2/Qwen3 分别沿用物理 GPU 1/2；
启动前显存空闲比例均为 0.969343。两套模型 eager/graph 共 **4/4**
启动及生成通过，controller/worker/server 均确认选项开启；各 **96 个位置**
完成独立 CPU FP32 参考。相较 iteration13，权重字节、每行输入及生成
token 序列均相同。源码完整性检查通过，未改写社区源码。

- Qwen2：最大所选 logprob 误差 0.066419，greedy regret 0.008249，
  top20 最大误差由 0.546768 降至 **0.512245**；
  top20+tail JSD 0.001217，tail mass 误差 0.009126，
  missed mass 0.000553。
- Qwen3：最大所选 logprob 误差 0.033474，greedy regret 0，
  top20 最大误差由 0.732415 降至 **0.642015**；
  top20+tail JSD 0.000850，tail mass 误差 0.007696，
  missed mass 0.000139。

两者仍仅因 top20 最大 logprob 误差超过 **0.15** 而 **FAIL**，对应
参考进程正常以状态 1 退出；不能解释为模型启动失败，也不能声称精度通过。
该实验表明只改变输出层选项不足以消除偏差，尚不能断言是 HCU 特有问题。
后续仍需逐层或同精度配置定位。

证据位于 `external-evidence:iteration16-{qwen2,qwen3}-fp32-head-`
前缀的 `generation.json/log` 和 `accuracy.json/log`；既有失败保留。
隐藏 GPU 的四模块 CPU 回归 **59 passed**，日志
`iteration16-final-cpu.log`；L0 GREEN（31 既有 warning）。
本次未重新执行全量 adapter/vendor 套件。

### 固定序列 CPU dtype 归因入口

```bash
HCU_ADAPTER_HOST=none VLLM_PLUGINS=none HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
  python3 -m tools.l15_checkpoint_dtype_probe \
    --smoke-report external-evidence:iteration13-qwen2-accuracy-generation.json \
    --out external-evidence:iteration17-qwen2-dtype.json --threads 4
```

仅接受已通过 eager/graph 执行且绑定实际 checkpoint 字节的 accuracy
请求；加载前后复核资产与输入报告，拒绝量化 checkpoint。两种 CPU dtype
都 teacher-force 相同 HCU 输出序列，完全相同的输入/输出才去重。
通过官方 `from_pretrained` 分别加载 FP32、BF16（包括输出层），
以相同的 Transformers eager、禁用 KV cache 计算；不能整体 `.to(BF16)`
舍入原本 FP32 的 RoPE buffer。报告保留两次加载的浮点 buffer dtype、
shape 和字节摘要。保留三组
top20+tail 对比：HCU→CPU FP32、HCU→CPU BF16、CPU BF16→FP32；
后者使用 CPU BF16 自己的 top20，另在 HCU 的同组 token 标签上定位
最大 dtype 差异。具体位置从 0 开始，报告的 HCU rank 从 1 开始。

输出 `l15-checkpoint-dtype-diagnostic`、`diagnostic_only=true`，
不输出精度 passed 或替换 V1 策略。CPU BF16 并不是相同实现的
SGLang/HCU 基线，既不能直接证明 HCU 存在错误，也不能豁免原失败。

2026-11-03，validation-host-b/validation-host-a/validation-host-c 均无满足预检条件的空闲 GPU，
本轮仅使用隐藏 GPU 的 CPU 进程。初次整体转换模型的试跑因 RoPE buffer
舍入风险而作废，保留在 `iteration17-*-converted-rope-invalid.*`
及 `iteration17-converted-rope-invalid.txt`，未用于精度判断。
修正后分别加载 dtype，回归覆盖官方 loader 参数、独立加载、FP32 buffer
保持、序列与预测位置对齐、三方误差定位、权重绑定拒绝和运行中证据变化：
**34 passed**（`iteration18-dtype-cpu.log`），L0 GREEN（31 既有 warning）。
真实 Qwen2/Qwen3 的多长度 CPU 对照使用 `iteration18-*-dtype.json/log`，
具体实验结论需在执行结束后读取报告，不以测试通过替代。

两模型均已结束，各对拍 **96 个生成位置**，两次官方加载的 RoPE
`inv_freq` / `original_inv_freq` 均保持 FP32，且字节摘要完全一致。
三组 top20 最大绝对 logprob 差分别为：

- Qwen2：HCU→CPU FP32 **0.546768**，HCU→CPU BF16 **0.743473**，
  CPU BF16→FP32 **0.753010**。
- Qwen3：HCU→CPU FP32 **0.732415**，HCU→CPU BF16 **0.786287**，
  CPU BF16→FP32 **0.710793**。

Qwen2 的 HCU→FP32 最坏点位于 `short-zh` 的第 4 个输出位置，
token108840、HCU rank18：HCU 与 CPU BF16 均为 -19.0，CPU FP32
为 -19.546768；该点符合 dtype 差异的表现。Qwen3 的最坏点则位于
`short-en` 第 6 个输出位置，token16209、HCU rank11：HCU
-7.171320，CPU FP32 -6.438905，CPU BF16 -6.447533。此处两种 CPU
精度相近，而 HCU 与二者仍相差约 0.72，不能以其他位置存在 BF16
舍入差异解释该点；后续须继续定位 SGLang/HCU 的具体计算链。
原 V1 FAIL 维持不变，诊断进程的成功退出仅表示对照完成。

### 真实 Channel-FP8 checkpoint 启动（2026-11-03）

资源复查发现 validation-host-a GPU 已释放，选择物理 GPU4 并再次进行容器内预检：
空闲显存比例 **0.969710**，BW1100/gfx938，DTK HIP 6.3.26113、
Torch 2.11.0；adapter `0129b81`，同一 pristine 社区 commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`。
真实 `/model/qwen3/Qwen3-0.6B-Channel-FP8` 的 config 声明
compressed-tensors、channel FP8 权重、dynamic token FP8 激活、
忽略 lm_head；9 项 checkpoint 资产绑定前后相同。

`iteration19-qwen3-fp8.json` 记录 eager/graph **2/2** 启动生成通过，
各两批两请求各 8 token；两模式输出相同，首个回答分别含 Paris、
4、Beijing、6。eager decode graph 分配为 0，graph 为 0.078 GB；
每次执行后源码 **0 changed / 0 added sources**。
实际参数为 TP1、BF16 compute、attention triton、page1、KV auto、
context128、max_total_tokens512、mem_fraction_static0.25。

配方来源沿用 cookbook commit
`40adc4d30ec2640e3391cfd78759bd3598e95f71` 的
`docs/model-deployment/sglang/qwen3.md`、
`Qwen3-0.6B IFB BW1100 1x SGLang 0.5.12` 条目。原配方是 BF16
checkpoint、fa3/page64、关闭 graph、mem0.9，未提供此 0.6B
Channel-FP8 精确条目；本实验明确替换为本地 FP8 checkpoint，
并采用上述社区 0.5.22/Triton 参数。

此项只证明该真实 checkpoint 的执行闭环，尚无独立量化精度结论；
server info 的 `quantization=None` 表示未显式设置该启动参数，
不能据它证明实际量化层路由。社区 ModelConfig 会从 HF config 自动
解析，但仍需补实际加载层/内核的运行时证据及独立量化参考，不能
把 config 声明、正常文本或启动成功当作量化数值验收。

### 同卡整模型 FP32 对照（2026-11-03）

smoke 新增显式 `--dtype float32`，通过社区 Engine 的公开 dtype 参数
执行，默认仍为 `bfloat16`。controller、worker、server info 与参考报告
必须一致；历史缺失 dtype 字段的报告仅按原 BF16 默认解释。未改社区
源码、adapter 默认计算精度或任何 V1 阈值。

validation-host-a 物理 GPU4，逐项预检空闲显存比例 0.969710，同一 checkpoint
`/model/qwen3/Qwen3-0.6B`、容器、社区 commit 和固定 accuracy 用例，
顺序执行 BF16、FP32，各完成 eager/graph **2/2** 生成及
独立 CPU FP32 参考 **96 个位置**。两者实际 checkpoint 字节、
每行输入与生成 token 序列完全一致，每次源码完整性检查通过。

- BF16 仍 **FAIL**：所选 logprob 最大差 0.041772，
  top20 最大差 **0.732379**，合并 JSD 0.001124，
  tail mass 差 0.009020，missed mass 0.000101，greedy regret 0。
- FP32 配置在**原 V1 策略**内 **PASS**：所选 logprob 最大差
  **0.003732**，top20 最大差 **0.034702**，合并 JSD 0.00000419，
  tail mass 差 0.000433，missed mass 0.000000223，greedy regret 0。
  这不是更严格的 FP32 数值策略验收，也不能外推为 BF16 已通过。
- 原 `short-en` 第 6 个输出位置 token16209 的 HCU FP32 logprob
  为 -6.416562，CPU FP32 为 -6.438905，原 BF16 HCU 约 -7.1713；
  执行精度改变明显缩小该偏差，BF16 的具体层/算子仍待定位。

报告为 `iteration20-qwen3-{bfloat16,float32}-{generation,accuracy}.json`
及相应日志。FP32 qknorm JIT 的编译尝试失败后社区走其原有 fallback，
真实生成继续成功；本次未因这个有 fallback 的尝试改写社区实现。
接口/配置/字节绑定/概率诊断组合 **76 passed**
（`iteration20-dtype-option-cpu.log`），L0 GREEN（31 既有 warning）。
本次未重新执行全量 adapter/vendor 套件。

### FP8 实际路由与 CT 组件数值闭环

validation-host-a 物理 GPU5 再次预检空闲显存比例 0.969710，使用社区公开
`Engine.start_profile(activities=["CPU","GPU"], with_stack=True,
record_shapes=True)` / `stop_profile()`，在 eager 模型预热后记录两个请求
各 8 token 的实际执行。未改社区源码或用调试 wrapper 替换模型计算。
加载日志明确 `Qwen3ForCausalLM, quant=compressed-tensors`，
trace 记录以下实际调用链：

- `CompressedTensorsW8A8Fp8.apply_weights`、`apply_fp8_linear`、
  `scaled_fp8_quant`、`_apply_fallback_scaled_mm` 与
  `aten::_scaled_mm` 各 **1008 次**；
- 设备端 dynamic-per-token quant 内核合计 **1008 次**；
- gfx938 `F8SS` FP8 GEMM 内核三种变体合计 **1008 次**。

原始 trace 598120 个事件，压缩约 13 MB，SHA256
`755d0eddde15e37c0d2f880562555b7018226ff6ef91fe26f79491cadab5f25d`。
`iteration21-fp8-runtime.json` 绑定 9 项 checkpoint 资产、原始 trace
及执行后社区源码检查；`iteration21-fp8-route-evidence.json` 从已校验
hash 的原始 trace 抽取调用计数。运行时 adapter stamp 为
`306c11c, dirty=true`（当时存在后续 `2406183` 的 dtype 诊断改动），
不能将其写成干净提交。profiler 的 ROCTracer duplicate-flow 警告不用于
性能分析，本阶段也不设置性能门槛。

发现旧组件测试的 FP8 分支未进入 CT scheme，且显式关闭 padding，
不能替代这条真实路径。已在 `test_quantized_linear_chain` 增加
`ct-fp8`：真实 scheme、channel scale、加载后处理、dynamic token
激活量化、默认 padding，覆盖 1/17/33 token、非零 bias、
独立 FP32 oracle 和三轮变输入 graph replay。新增 **3 passed**，
源码完整性通过；随后正式量化家族严格门 **8/8 case PASS**，
其中该链的 INT8/FP8/CT-FP8 合计 **9 个参数化节点**均执行，
无 skip/xfail，源码检查通过，L0 GREEN（31 既有 warning）。
报告为 `iteration21-ct-fp8-chain.json/log`、
`iteration21-quantization-family.json` 及其 cases。
实际路由与组件数值闭环均已补齐；真实量化模型的完整概率对拍仍待补。

### 同机 Transformers CPU/HCU 定点归因

`iteration22-qwen3-device-probe.json` 在 validation-host-a GPU4（预检空闲比例
0.969710），对原 `short-en` 的同一冻结序列，分别官方加载 FP32/BF16，
先 CPU、再仅将设备迁到 HCU；保留 FP32 RoPE buffer，eager attention、
禁用 KV cache，不安装 SGLang 宿主桥。checkpoint 与输入报告前后字节
检查通过。该探针只覆盖一个短序列，不是完整模型验收。

- Transformers FP32 的同机 CPU/HCU top20 最大差 **0.0000601**，
  合并 JSD 0.00000000836；目标位置 token16209 的 logprob 分别
  **-6.438941 / -6.438981**。说明该探针的普通 HCU Torch FP32
  路径与 CPU 很接近，不能把 SGLang BF16 的偏差笼统归为设备不支持。
- Transformers BF16 同机 CPU/HCU top20 最大差 **0.5625**，
  合并 JSD 0.002814；目标位置分别 **-6.256028 / -6.635359**。
  SGLang BF16 原目标值约 -7.1713，与 HCU Transformers BF16
  也有差异；仍需在融合 GEMM/归一化/attention/增量执行之间继续归因。
- validation-host-a CPU 是 Intel Xeon Platinum 8558P，而之前 validation-host-b 是
  Hygon C86-4G；BF16 CPU 结果不能跨节点当成同一实现。
  环境、Torch build/git 与诊断字节绑定保存在
  `iteration22-qwen3-device-environment.json`。该探针保持诊断性质，
  不改变原 BF16 V1 FAIL。

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

- validation-host-c 单卡 GPU 0：Dense、MoE、MLA、Mamba2、GDN hybrid 五类两层
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

## 显式量化选项与参考边界（2026-11-03）

`tools.l15_model_smoke --quantization w8a8_int8` 转发社区 Engine 公开选项，
默认仍使用 checkpoint 自动检测。controller、worker、实际 server info 和
参考输入必须一致，忽略选项或陈旧 worker 报告会失败。CPU FP32 参考和
dtype 诊断同时拒绝 checkpoint 量化声明与显式在线量化，不能把未声明量化
的原始 config 当作在线量化执行的数值金标。

真实 `/model/hygon/Qwen3-30B-A3B-Channel-INT8-w8a8` 自动检测的失败保留于
`external-evidence:iteration23-qwen3-moe-int8.json/log`：社区 CT MoE
构造拒绝并提示仅支持 NPU。社区另有 GPU `W8A8Int8MoEMethod`，且 cookbook
固定版本 `40adc4d30ec2640e3391cfd78759bd3598e95f71` 的同模型 BW1100
配方显式指定 `--quantization w8a8_int8`，因此先验证现有官方入口。
该入口是否完成真实加载、生成及数值验证须分别保留后续实测证据。

validation-host-b 隐藏 GPU 的工具回归 **114 passed**，报告
`external-evidence:iteration25-quantization-option-cpu.log`。
此次新增节点授权包括 validation-host-a、validation-host-b、validation-host-c 的 `adapter-validation`，
代码共享目录相同；GPU 作业仍须逐次检查空闲资源。

## INT8 Linear 与量化模型验证（2026-11-03）

### INT8 Linear 公开路径的缺失符号修复

显式 `w8a8_int8` 的首次真实 30B 执行继续暴露
`W8A8Int8LinearMethod.apply -> NameError: int8_scaled_mm`，原始失败保留在
`external-evidence:iteration26-qwen3-moe-int8.json` 及 jobs 日志。
此前仅给 CT scheme 补 GEMM global，未覆盖显式量化的另一个消费者。
`hcu_adapter.int8_linear` 复用现有 `sgl_kernel.gemm_int8.int8_scaled_mm`，
先验证两个消费者的方法和 globals 契约，再绑定各自模块符号。
原方法及已缓存别名保持原样；没有新增数学实现或源码写回。
catalog/UI/runtime registry 同步登记为 candidate，不据此作性能晋级。

validation-host-b 隐藏 GPU 的桥接/矩阵工具回归 **35 passed**，L0 **GREEN（31 warn）**，
见 `iteration31-int8-linear-cpu.log` 和 `iteration31-int8-linear-l0.log`。
设备回归与真实模型复测需以各自结果判断，不能由 CPU 绑定回归替代。

validation-host-b GPU4 的既有 INT8 GEMM 与新增真实 W8A8 Linear 调用链共 **20 passed**，
后者包含 6 个参数化节点及三轮变输入 graph；见
`iteration31-int8-linear-gpu.log`。严格 MoE 家族另获 **7/7 case PASS**，
包括新 INT8 MoE，报告 `iteration30-moe-family.json` 的执行后社区源码
完整性通过。以上仍不代表真实 30B 模型或完整精度验收通过。

### 真实 30B Channel-INT8 模型启动与生成

`iteration31-qwen3-moe-int8.json` 在 validation-host-a GPU4/BW1100 gfx938 上记录
真实 `Qwen3-30B-A3B-Channel-INT8-w8a8` 的 **eager/graph 2/2 PASS**。
16 项 checkpoint 资产包含 7 个 safetensors shard，controller/worker
前后字节绑定一致，两项执行后社区源码完整性通过。基线仍为
`baa9befa381511cd4d56f83980951e5a45fffc6d`，实际执行时 adapter 戳为
`2832e48 dirty=true`，包含随后提交的 `5efb916/11a282f/80432d2` 工作内容；
不能把报告原始戳改写成后来的干净 commit。

官方选项为 `quantization=w8a8_int8`、BF16、TP1；scheduler 回读 decode
graph 分配 **0.021 GB**，eager 为 0。每模式两批、每批两请求，每请求
8 token，保留输出 IDs、有限 logprob 与原始文本。部分输出以 Paris/4/6
开头，中文请求是尚未完成的选择题续写；这些短输出不作为语义金标，
也不证明 eager/graph 概率一致或全模型量化精度合格。

参考 cookbook 固定 commit
`40adc4d30ec2640e3391cfd78759bd3598e95f71` 的
“Qwen3-30B-A3B-Channel-INT8-w8a8 IFB BW1100 1x SGLang 0.5.12”：
相同模型/BF16/显式 W8A8/TP1，但本次社区 0.5.22 使用 MoE auto
及 Triton attention、KV auto/page1、mem_fraction_static=0.25、
context128/max_total_tokens512，区别于配方的 lightop/FA3/FP8 KV/page64/
mem0.9。因此是社区基线启动探针，不是配方全部选项验收。

### 全量回归中的 Python 兼容修复（2026-11-04）

validation-host-b 的原始 `iteration31-regression-full.log` 在收集期失败：
Python 3.10 缺少 `tomllib`，而包声明支持 Python ≥3.9。
打包测试增加标准库/`tomli` 兼容导入，`test` extra 声明
Python <3.11 的 `tomli` 依赖；未跳过测试。该容器真实 wheel 构建/
内容检查与 entry-point 检查 **2 passed**，见 `iteration32-packaging.log`。
完整 suites 复跑另存 `iteration32-regression-full.log`，须按最终结果判断。
原纯 CPU 契约 **38 passed**。旧 triage 的 rope 探针默认引用不存在的
`/public/share/GLM-5-W4A8_V2_6`，此错误暂作为环境/诊断问题保留，
不计为算子通过。

全量复跑 `iteration32-regression-full.log` 的真实结果为
**971 passed、56 failed、29 errors、132 skipped、2 xfailed、1 deselected**；
执行后社区源码完整性通过。47 项失败和 29 项 setup error 属于 vLLM 路径
（包括 `test_model_mutation.py` 的实际 vLLM Worker），按用户要求暂缓。
其余 9 项分别为 kpool pool fixture 两项、影子导入隔离一项、CPU
参考/dtype 测试的宿主隔离五项及 all-reduce 快关一项；后续修复须以独立
增量报告验收，不能修改这份原始全量结果。

旧 triage `_pytest` 摘要只截取 passed 后面的内容，会漏掉前面的失败数。
现保留完整 pytest 统计，并在失败时输出完整原始 stdout/stderr；
统计/异常/非零返回码回归 **3 passed**，报告
`iteration33-triage-reporting.log`。RoPE 使用显式
`HCU_CKPT=/model/glm5/GLM-5` 的同节点独立复测得到 q/k cos=1、
max_abs=0；见 `iteration32-rope-probe.log`。原路径错误没有被覆盖。

上述 9 项非 vLLM 失败已完成增量修复：删除 all-reduce 拒绝门对退役
`_tar_installed` 的引用，恢复显式快关；kpool pool fixture 增补社区
`index_page_size` 字段；影子导入与 CPU 参考测试使用干净 HOSTnone
子进程，生产工具的“已加载宿主即拒绝”守卫不变。
validation-host-b 后续资源检查显示显存已占用，GPU 作业在启动前拒绝；未终止其他进程。
改在已确认空闲的 validation-host-a GPU5 验证全部相关文件，得到
**51 passed、1 skipped、1 deselected**，见
`iteration34-regression-fixes.log`。skip 是旧 aiter 布局才适用的 opt-in
场景，deselected 为本轮不运行的双卡 stress 项。
该证据是全量失败后的增量验证，不把原 56 failed/29 errors 改为整轮全绿。

最终增量检查：validation-host-a GPU4 的量化家族严格门 **9/9 case PASS**，零
skip/xfail，执行后社区源码完整性通过，见
`iteration34-quantization-family.json`。最终 L0 **GREEN（31 warn）**，
见 `iteration34-final-l0.log`。当前保留的 vLLM 47 failed/29 errors
尚未修复；全模型 BF16/量化精度及其他既定覆盖缺口仍需继续迭代。

### BF16 逐层归因与增量解码复现（2026-11-04）

使用同一 validation-host-a GPU4、Qwen3-0.6B checkpoint 字节和社区 commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`，在 adapter `d813991`
执行只读诊断，未修改模型 forward 或社区源码。HF 独立按 BF16 加载，
仅把设备迁移到 HCU，保留 RoPE 的 FP32 buffer；参考为 HF eager。
SGLang 通过公开 custom weight loader 接口安装 PyTorch hooks，
显式检查 begin/update/end 均成功，并检查张量文件确实生成。
这不是模型精度验收，也没有改变原 V1 阈值或 BF16 失败状态。

`iteration36` 对 short-en prompt 加前 5 个固定输出 token 做 29-token
全量 prefill。与 `iteration35` 未成功安装 hooks 的对照执行相比，
输出及 logprob 数据相同，响应差异仅为请求 ID 和时间字段。
首层 norm 输入逐元素相同，输出最大差为 0.03125（最后位置
0.00390625）。使用相同输入和真实 norm 权重做 CPU FP32 公式对照：
HF 输出精确匹配“归一化后先转 BF16、再乘权重”，SGLang 输出精确
匹配“FP32 归一化并乘权重、最后转 BF16”。社区 RMSNorm 已区分
默认语义和 `cast_x_before_out_mul` 的 HF 语义，不能据此强改默认实现。
证据：`iteration36-layer-comparison.json`、`iteration36-*-layers.pt`
和 `iteration36-layer-probe-runtime.json`。

为避免以全量 prefill 代替原增量解码，`iteration37` 重放原 accuracy
wave0 的两个请求，context4096/total16384、eager、单卡 BF16/Triton
attention，生成 6 个 token。两请求前 6 个输出 ID 均与原记录一致，
所选 token logprob 的最大差分别为 0、0；最后一次采样确认为第 6 次
forward、batch2。额外保存 fused norm 的原始输入和残差，避免只保存
已转 BF16 的相加结果而丢失方差计算信息。

在 short-en 最后解码位置，首层输入仍完全一致，norm 输出最大差
0.00390625；最终 norm 输出最大差 1.25。原最差 top20 token 16209
的 logprob 为 SGLang BF16 −7.17131996、同卡 HF BF16 −6.63534498，
与此前 CPU FP32 −6.438905 的差异不能用改参考 dtype 豁免。
56 个实际 norm 调用在各自 SGLang 输入上与社区 JIT 默认公式对照：
55 个逐元素一致，另一个为 layer03/post_norm，最大差 0.0078125；
尚未据此证明 adapter 错误，亦未以公式对照代替真实 NVIDIA 内核运行。
社区 JIT 默认 fused-add 路径以 FP32 相加值计算方差、以转回激活 dtype
的相加值作归一化分子，不能误用全 FP32 或全 BF16 相加公式判断。

两轮 worker 均正常退出、执行后社区源码完整性通过。证据为
`external-evidence:iteration37-{hf,sglang}-layers.log`、
`iteration37-*-layers.pt`、`iteration37-sglang-generation.json`、
`iteration37-layer-comparison.json` 和 `iteration37-layer-probe-runtime.json`。
hooks 仅用于 eager 归因，含主机同步，不进入生产或 graph 路径；
本轮没有证明完整 BF16 精度通过，也没有修改算子实现。
后续仍需在固定输入上逐段分离 GEMM、attention 和激活舍入的贡献，
并继续既定模型族与配置覆盖。

### Mamba2 真实变输入 graph 计算链（2026-11-04）

新增 `test_mamba2_decode_fresh_input_graph` 并接入严格家族矩阵，
补齐原先只有 metadata 生命周期和模型 graph 启动的组件数值缺口。
官方 `MambaMixer2` 执行 in-projection、causal Conv1D、SSM、gated
RMSNorm 与 out-projection；独立 CPU FP32 参考实现整个递推，保留
BF16 激活/卷积状态存储边界。batch1/page1/group1 和
batch3/page16/group2 各做 eager/真实 graph，三轮改变输入与映射槽位，
同时检查输出、连续递推状态、图内外状态一致性及其他槽位禁止改写。
本功能没有替换生产计算或修改社区源码。

validation-host-a GPU4：整个计算链文件 **25 passed**，严格 Mamba 家族
**7/7 case PASS**、零 skip/xfail，矩阵工具回归 **32 passed**；
执行后社区源码完整性通过。证据为 `iteration39-chains.log`、
`iteration39-mamba-family.json`、`iteration39-matrix-cpu.log` 和
`iteration39-mamba-runtime.json`。清单目前 31 case，本次没有重跑其他
四个家族；纯 Mamba2 真实 checkpoint 和 speculative/padded graph
等范围仍不因该组件通过而视为验收完成。

### FP8 grouped expert GEMM 数值与图合同（2026-11-04）

新增 `test_fp8_grouped_expert_contract`，直接检查既有 adapter Triton
masked/contiguous 实现；不改变 DeepEP 路由或实现默认值。4 个参数节点
覆盖 K/N=128/256、有符号 FP8、非单位 128×128 权重 block scale、
per-token-group 激活 scale 和非连续 scale view。masked 覆盖空专家、
1/17/32 token 及 padding 禁写；contiguous 覆盖 64-row aligned
专家分组和 129 行尾 tile。FP32 独立反量化对拍保持 cosine ≥0.9999、
maxrel ≤0.05；真实 graph 三轮改变输入、权重、尺度与 counts/expert IDs。

validation-host-a GPU5：完整 FP8 GEMM 文件 **17 passed**，接入后的严格量化家族
**10/10 case PASS**、零 skip/xfail，矩阵工具 **32 passed**，社区源码
执行后完整性通过。见 `iteration40-fp8-grouped.log`、
`iteration40-fp8-grouped-runtime.json`、`iteration41-quantization-family.json`、
`iteration41-matrix-cpu.log` 和 `iteration41-quantization-runtime.json`。
L0 **GREEN（31 warn）**、纯 CPU 算子 **38 passed**；见
`iteration41-l0.log`、`iteration41-ops-cpu.log`。
当前清单 32 case，Mamba 与量化分别验证，未合成一次全家族整轮证据；
真实 DeepEP 多卡 dispatch/combine、模型精度和其他原定缺口仍需继续。

### 修复既有 DeepGEMM 开关漏登（2026-11-04）

检查 FP8 合同时发现 `HCU_TRITON_MASKED` 和
`HCU_TRITON_MASKED_FP8` 已被 installer 使用，但不在 `hcu_env.REGISTRY`。
实际复现：两键显式 0/1 都解析为 `(None, "unknown")`，
而 `enabled` 将该值当作开启，导致显式关闭被忽略。
补齐两个既有键的登记和手册，默认 1 保持原有实际选择；
未修改通用 `enabled` 的未知键行为，也不新增数学实现或更换默认内核。
关闭 FP8 路由时保留 vendor 入口及未验证 scale-layout 的响亮拒绝，
不为未验证 vendor 数值背书；开关只在安装阶段生效。

干净 HOSTnone 子进程以 provider 替身运行真实 installer，覆盖默认、
显式 0、显式 1 优先于配置 0、配置 0；检查 BF16 ABI wrapper、
FP8 两入口和 layout 拒绝。修复前新增 **8 failed**，修复后连同
环境守卫、平台默认及启动结构回归 **24 passed**，L0
**GREEN（31 warn）**；见 `iteration42-deepgemm-switch-before.log`、
`iteration42-switch-after.log`、`iteration42-l0.log`。

随后在 validation-host-a GPU4 暴露单卡、真实社区与 DTK 依赖下分别启动干净
SGLang 宿主解释器：默认、两键显式 0、两键显式 1 **3/3 成功**，
真实 `deep_gemm` 函数绑定及解析值/来源均符合预期，
执行后社区源码完整性通过。证据为
`iteration42-real-installer-{default,0,1}.log` 和
`iteration42-real-installer-runtime.json`。这只证明真实安装/导入和选路，
没有在关闭态执行已知未验证的 vendor GEMM，也不替代模型或多卡精度。

### 真实 CT-FP8 的独立 CPU QDQ 参考（2026-11-04）

`tools.l15_checkpoint_reference` 新增显式
`--quantized-reference ct-fp8-token-channel`。默认仍拒绝量化执行；
此入口仅接受已验证的 Dense Qwen3 compressed-tensors FP8 channel 权重、
动态 token FP8 输入、忽略 lm_head 的格式，要求原 HCU 报告绑定 checkpoint
字节，拒绝 INT8、MoE、静态/非对称/分组量化、输出/KV 量化和执行选项冲突。
在隐藏 GPU、HOSTnone 的独立进程运行，例如：

```bash
HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
python -m tools.l15_checkpoint_reference \
  --smoke-report external-evidence:iteration45-ctfp8-generation.json \
  --quantized-reference ct-fp8-token-channel \
  --out external-evidence:iteration47-ctfp8-bfloat16-accuracy.json
```

参考通过 Transformers 公开加载接口和 compressed-tensors 的
`run_compressed=False` 进行 CPU FP32 eager 执行。完整加载信息必须无缺失、
多余、形状不符或错误；逐个核对全部原始 tensor，量化权重必须精确等于
原始 FP8 值乘 channel scale，其余 tensor 也须精确匹配。实际 Linear 名称、
数量和量化参数必须完整匹配；Python profiler 只观察官方执行，不替换计算，
每次完整 forward 必须逐层观察到真实输入量化与 fake-quantize 调用。
仅反量化权重而移除激活 QDQ 不是此 checkpoint 的参考。

validation-host-a GPU4、BW1100/gfx938、Torch 2.11.0/HIP 6.3.26113，
Transformers 5.12.1、compressed-tensors 0.17.0；
社区固定 `baa9befa381511cd4d56f83980951e5a45fffc6d`。
真实 `/model/qwen3/Qwen3-0.6B-Channel-FP8` 的 9 项资产字节绑定通过，
507 个 tensor 精确核对通过，196 个 Linear 在 5 次不同输入参考 forward
各执行一次激活 QDQ。原固定双语、512/2048 token、复用/变输入请求，
eager/graph 生成 **2/2 PASS**；96 个输出位置的原 V1 精度门为 **FAIL**：
最大所选 logprob 误差 0.235903、greedy regret 0、
top20 logprob 误差 2.673436、top20+tail JSD 0.016584、
tail mass 误差 0.056121、missed mass 0.003061。
两模式各请求指标相同，greedy 一致不能代替分布验收。
新增工具及相关回归 **47 passed**，L0 **GREEN（31 warn）**。
证据为 `iteration45-ctfp8-generation.json`、
`iteration47-ctfp8-bfloat16-accuracy.json`、`iteration47-cpu.log`、
`iteration47-l0.log`、`iteration47-reference-validation.json`。

另以显式整模型 FP32 作归因诊断，社区加载器在 eager/graph 均明确拒绝：
`torch.float32 is not supported for quantization method compressed-tensors`，
仅允许 FP16/BF16。没有执行 CPU 对拍，也没有修改社区 dtype 准入；
此结果不能为 BF16 差异归因或豁免。见
`iteration46-ctfp8-float32-generation-jobs/checkpoint-{eager,graph}.log`、
`iteration46-ctfp8-float32-runtime.json`。所有执行后社区源码完整性通过。

cookbook 固定 `40adc4d30ec2640e3391cfd78759bd3598e95f71`，
`docs/model-deployment/sglang/qwen3.md` 中 Qwen3-0.6B IFB BW1100 单卡
配方为 SGLang 0.5.12、BF16 checkpoint、fa3、page64、关闭 graph、mem0.9；
没有对应 0.6B Channel-FP8 配方。本次差异为社区 0.5.22、FP8 checkpoint、
triton、page1、mem0.25、eager+graph、context4096、total tokens16384，
不能将其记作 cookbook 原配方已验收。

### 提交前全量回归（2026-11-04）

validation-host-a GPU4 执行 `pytest tests/kernels tests/adapter tests/vendor -k "not stress"`：
**1017 passed、47 failed、29 errors、131 skipped、2 xfailed、1 deselected**，
用时 682.58 秒。47 个失败与 29 个错误均属于 vLLM 路径，按当前范围暂缓；
其中 13 个失败位于未以 vllm 命名的 `test_model_mutation.py`，
实际依赖 `VLLM_TARGET_ROOT` 或 vLLM runner 的 `reload_weights`。
不能仅按文件名判断宿主，也不能将整套报告记为 GREEN。
其余 SGLang/adapter 用例未发现失败；跳过与 xfail 不算通过。
CPU 算子契约另有 **38 passed**。原始日志为
`iteration48-suite.log`、`iteration48-cpu.log`，运行戳及源码检查见
`iteration48-validation.json`；测试时 adapter 为 e5d2943 加本功能未提交字节，
随后相同实现提交为 b36ad0b，保留原始 dirty 戳。

补跑 ops 分诊时，旧的默认 `/public/share/GLM-5-W4A8_V2_6` 不存在，
因此该次 rope 记录为环境失败；将 `HCU_CKPT` 指向现有
`/model/GLM-5-Channel-INT4-w4a8` 后重新通过，rope 最小 cosine
**1.000000**，rmsnorm **0.999997**，BF16 GEMM excess **1.02x**，
graph safety 通过。该路径问题属于探针默认配置，不是算子数值失败。

validation-host-a GPU5 空闲显存比例 **0.9697097** 时，完成一次完整的 32-case
L1.5 family matrix：Dense 3、MLA 5、MoE 7、量化 12、Mamba 5，
**32/32 PASS、零 skip/xfail**，结果为 `iteration49-all-families.json`，
运行日志为 `iteration49-all-families.log`，执行后社区源完整性复核通过。

## 第四步验证（2026-11-03）

- `ci/upstream_diff.py` 对不可变 Git 快照比较已声明的符号、签名/函数体、
  白名单补丁 needle 的位置/次数、上游测试 nodeid 与版本绑定文件。
  新路径复制同一个 needle 也要求审查。
- `tools.l15_upgrade_check` 串接静态审查、严格组件门、模型闭环；陈旧或
  其他 upstream commit 的报告不能使升级就绪，审查接受文件必须精确匹配
  baseline/candidate/changes。未完成全部阶段时保持 `ready=false`。
- validation-host-c 实际运行 `--baseline-ref HEAD --candidate-ref HEAD --audit-only`：
  **0 changes、15 baseline gaps、19 dynamic registration sites**；
  `same_revision=true`、`audit_only=true`、`ready=false`。这些缺口需运行时
  验证，不能将零差异解释为所有绑定完整或完成一次版本升级。
- 同容器工具组合回归 **91 passed**；补充 needle 新路径检查后，本地仅跑
  漂移/编排 CPU 测试 **12 passed**。L0 **GREEN（31 warn）**。
  故意修改签名、函数体、删除符号/nodeid/needle 的回归均能拒绝。
- 证据：`external-evidence:phase4-workflow-audit.json`、
  同名 `-stages/upstream-diff.json`、`final-tool-tests.log`、`final-l0.log`。
  本次交付手动升级工作流，未配置定时巡检，也未验证新社区版本。

### DeepSeek-V4.1-Flash-Channel-INT4 真实启动闭环（2026-11-04）

在 validation-host-a 的 `adapter-validation` 容器中，以 adapter
`d1df244867a5738c4c4a15f35e4eae5c5e7fec09`、社区 SGLang
`baa9befa381511cd4d56f83980951e5a45fffc6d` 和 cookbook
`40adc4d30ec2640e3391cfd78759bd3598e95f71` 运行真实
`/model/DeepSeek-V4.1-Flash-Channel-INT4-w4a8` checkpoint。checkpoint
执行前后保持相同的顶层文件清单指纹
`b2f7e8532d6cfd5ec459f51f18e30ecac58292dc2279957b3ebf519b7d3b373f`；
`config.json`、`model.safetensors.index.json` 和
`tokenizer_config.json` 的 SHA256 也保持不变。社区源码目录执行前后仍为
固定 commit 且工作区干净。

当前可复现的启动配方是 TP4/EP1 correctness 路径：

```text
--tp-size 4 --ep-size 1
--moe-a2a-backend none --moe-runner-backend triton
--mem-fraction-static 0.90 --max-running-requests 16
--chunked-prefill-size 8192 --cuda-graph-max-bs-decode 16
--language-model-only
```

模型后端自动选择 page size 256，prefill graph 关闭，decode graph 捕获
`[1, 2, 4, 8, 12, 16]` 六个 batch 桶。运行时设置
`SGLANG_DSV4_FP4_DEQUANT=1`、`SGLANG_HCU_EP_BF16=1`，并关闭
`SGLANG_OPT_FUSE_MHC_POST_PRE`、两个 TileLang mHC pre/post 路径以及
`SGLANG_ROCM_USE_AITER_TILELANG_MHC`。460GB 真实权重加载完成，每卡约
116.4GiB；decode graph 捕获成功，用时约 67 秒；两次 `/generate` 均返回
HTTP 200 且请求返回码为 0。巴黎/埃菲尔铁塔请求生成连贯文本；算术请求
立即输出 stop token，仍需补充更强的语义断言后再记作完整语义验收。

这次验证确认了“L1.5 用例通过后可启动真实模型”的 TP4/EP1 闭环，不能
外推为 DeepEP EP4 已通过。TileLang 0.1.15 的 mHC pre 融合仍因
`ThreadSync: Cannot lower a required shared-memory sync inside the else branch`
而关闭；`mem-fraction-static=0.78` 不足以同时容纳权重和 KV cache，当前
实测最低可行值约为 0.8683。TP4/EP4 在社区 dispatcher 中会把 adapter
的 CT W4A8→W4A16 correctness 路径送入已退役的
`forward_deepgemm_masked`，所以 EP4 作为后续独立功能点处理，不能纳入
当前启动通过结论。

完整配方、环境、指纹和请求证据见
`external-evidence:iteration51-dsv41-int4.json`，原始服务日志和请求
结果见同名目录。下一步先在 validation-host-b 做离线 EP4 dispatcher 复现，再修改
adapter 运行时路由；修改前必须增加离线回归，且不写回社区源码。

### CT W4A8 DeepEP low-latency W4A16 路由（2026-11-04）

为处理上一个闭环暴露的 EP4 边界，adapter 新增了
`DeepEPMoE.run_moe_core` 的窄路由：当 dispatch 输出同时包含
`masked_m`/`expected_m`，且层权重确认为 CT W4A8→W4A16 的 uint8
expert 权重时，adapter 在 expert-major
`[local_expert, capacity, hidden]` 布局上执行两次 masked W4A16 grouped
GEMM 和 SiLU gate，再把相同布局交回 DeepEP combine。普通 TP/EP1
WNA16、BF16、FP8/W4AFP8、normal dispatch 和其他量化路径都保留上游
实现；异常不会静默改走另一数学路径。

本功能提交为 `a7db6fc`，CPU 路由/契约回归为 **9 passed**。validation-host-c
容器中的社区源码目录仍为
`baa9befa381511cd4d56f83980951e5a45fffc6d`，未发生文件改写；显式以
社区 0.5.22 pristine Python 路径加载后，调用安装函数可观察到
`DeepEPMoE.run_moe_core._hcu_w4a16_deepep=True`。这只证明 seam 安装和
布局契约，尚未替代真实 EP4 权重加载、DeepEP 通信、masked GEMM 数值和
生成验证。

validation-host-c 当前 8 张卡仍由既有服务占用，显存约 94%，因此本轮没有启动新的
GPU 作业。下一步是在 validation-host-c 释放后，用真实
`DeepSeek-V4.1-Flash-Channel-INT4-w4a8` 做 TP4/EP4 启动、graph/eager
生成和与 EP1 的数值/语义对拍；若该 masked W4A16 ABI 与实际 DeepEP
buffer 形状不一致，先补离线形状/数值回归再修 adapter。

### validation-host-b EP4 契约补齐（2026-11-05）

用户再次授权使用 validation-host-b。本次启动前 8 卡显存及利用率均为 0%，
仅暴露物理卡 0–3，在 `adapter-validation` 内使用同一代码目录、
社区 `baa9befa381511cd4d56f83980951e5a45fffc6d` 与真实
DeepSeek-V4.1 INT4 checkpoint。启动前源码字节审计通过，48 个分片全部存在。

离线回归复现了原 EP4 分支的两个缺口：社区 CT Triton scheme 转换后仍保留
`w13_weight_packed`/`w2_weight_packed` 属性，而 adapter 只识别不带
`_packed` 的名称；channel-wise 权重的单尺度不能通过固定 group 白名单。
现兼容这两种名称，并将合法的 channel 尺度等价展开到 vendor group-32 网格。
保留 checkpoint 的 `swiglu_limit`（gate 仅上界截断、up 双向截断），
对非 SiLU、非 gated、输入侧 routing 和其他 GEMM1 变换显式拒绝；
带 zero-point 的权重不进入该对称分支。设备上的行数范围通过异步断言
验证，避免 Python 读取 GPU 标量，并在 graph replay 时检查当前行数。

validation-host-b GPU0 定向执行 `test_deepep_w4a16.py`、`test_ct_w4a8_moe.py`、
`test_w4a16_moe_deepgemm.py`：**26 passed，24.41 秒**。新增真实 GPU
对拍包含 group128/channel-wise 两种尺度、clamp、空 expert、
变输入和变行数 graph replay；原始日志为
`external-evidence:iteration54-validation-host-b-ep4-regression.log`。
这是组件证据，不代表真实 EP4 模型已通过；本次真实启动仍在独立验证中。

启动脚本 `_launch_dsv41_int4_ep4_pristine.sh` 同步修正为此次实际参数：
显式四卡可见性、BF16 DeepEP dispatch、纯语言模型、mem0.90、
关闭 prefill graph，由模型后端选择 page size，保留服务 warmup。
源码预检在 `HCU_ADAPTER_HOST=none` 下执行，默认监听本机地址。
validation-host-b `bash -n` 通过；原始 EP4 启动已在 decode graph warmup 中复现
`forward_deepgemm_masked is deprecated`，修复后的重跑记为 iteration55，
不能将脚本语法通过当作模型启动通过。

### EP4 权重副本显存修复（2026-11-05）

iteration55 进入新的 EP4 分支后，在 decode graph warmup 的
`_grouped_gemm` 输出分配处 OOM（再申请 480 MiB；单卡仅余 688 MiB）。
此前已成功加载全部真实权重。vendor 路径在各层原权重之外持有完整
marlin repack 与 group32 尺度缓存，不能用小形状组件通过外推整模型可用，
也不能仅下调 KV 预算掩盖累积的权重副本。

DeepEP LL serving 改用 `grouped_gemm_triton.w4a16_masked` 直接读取
原始 uint8 packed 权重和原始分组/channel 尺度，按设备行数跳过空 expert
与无效 tile；FP32 累加、BF16/FP16 输出，不缓存第二份权重，也不展开尺度。
vendor 路径保留为显式数值比较入口，EP1 与非目标量化的路由不变。
两次 GEMM 后仍使用原有 clamp/SiLU 语义及 DeepEP combine ABI。

validation-host-b GPU0 同组定向回归 **30 passed，27.06 秒**，新增 raw-weight tier
与 vendor tier 的 group/channel clamp graph 对拍、无效行归零、变行数
重放、非整 tile K/N/M，以及 raw tier 不产生 repack cache 的断言。
原始日志为 `external-evidence:iteration56-regression.log`。
iteration55 执行后社区源码 **10594 个 tracked file 字节审计通过**，
commit 不变且无新增源文件。真实 EP4 端到端验收仍待修复后重跑。

### validation-host-b EP4 整模型重跑边界（2026-11-05）

iteration56 在 `b917a47` 上完成真实权重加载及四卡 decode graph
捕获，batch 桶为 `[4, 8, 12, 16]`，耗时约 30 秒，graph 占用约
4.52 GB；捕获后 rank0 仍有 8.53 GB、其余 rank 约 9.39 GB 可用。
因此原 repack 缓存导致的 OOM 未再次出现。HTTP 应用启动且
`/model_info` 返回 200，但默认服务 warmup 的首次 prefill 触发
GPU VMFault，尚无成功的 EP4 生成结果，不能记为模型通过。

iteration57 的诊断参数 `AMD_SERIALIZE_KERNEL=3` 被当前运行库警告为
无效值（仅接受 0/1）。它再次完成 graph 捕获后，在
`dsv41_sparse.token_req_indices` 的 `repeat_interleave` 调用附近崩溃；
由于异步执行和无效的序列化设置，该栈仅作为定位线索，不作根因结论。
两轮结束后的社区源码审计均通过：10594 个 tracked file 字节不变，
commit 仍为 `baa9befa381511cd4d56f83980951e5a45fffc6d`，无新增源文件。

独立 GPU `repeat_interleave` 探针覆盖 1/2/8/16/128/8192 token、
多请求及零长度段，18 组均与 CPU 参考一致。该结果不能排除真实请求
元数据、跨流生命周期或更早算子的问题。下一轮仅在临时诊断入口记录
eager prefill 的请求索引、设备/CPU 长度与输出 token 数，不写社区源码。
iteration58 的诊断钩子误在 decode graph 捕获期间同步，触发捕获错误；
该轮属于诊断脚本失败，不归因于模型，后续诊断必须避开 capture。

### V4.1 EP 补齐请求索引（2026-11-05）

iteration59 在四个 rank 均记录到同一输入：真实请求 `req=[16]`、
设备及 CPU 长度均为 `[6]`，但 `positions.shape[0]=8`。EP 路径按
并行度补齐 token 后，社区 `token_req_indices` 将 8 直接作为
`repeat_interleave` 的 `output_size`，违反其必须等于 repeats 总和的
契约。诊断在展开前同步成功并用断言阻止了非法调用，区别于前面仅有
异步故障栈的线索。

`dsv41_request_indices` 仅为该函数安装私有 Torch facade，保留原函数
对象、既有 backend 别名、decode 和 scalar-repeat target-verify 分支；
不改全局 Torch 或社区文件。向量展开通过设备端累计长度和 searchsorted
映射有效 token，多出的行映射到社区预留的 request 0，避免借用有效请求
的位置 0。负长度及容量不足显式拒绝；不读取 GPU 标量，支持 graph。

validation-host-b GPU0 定向回归 **19 passed，19.10 秒**，覆盖 6→8、空段/空批次、
无补齐、非法长度、decode/verify 保持原行为、私有绑定/漂移拒绝，以及
同地址变请求、变长度、全 padding 的 graph replay。
真实社区 backend 的已导入别名和 spawn 子进程均通过正式 bootstrap
探针。L0 为 GREEN（31 条既有 warning）。证据在
`external-evidence:iteration60-validation-host-b-padding/`；复用入口为
`tests/tools/_probe_dsv41_request_indices.py`。这是元数据组件和接线
证据；修复后的真实 EP4 服务 warmup/生成仍须独立重跑。

iteration61 使用普通异步执行、无诊断钩子，已通过原来失败的 6-token
warmup。三个显式 DeepSeek 对话 token 请求分别正确输出 `42`、`Paris`、
`63`（包含 21-token 补齐输入和 20-token 无补齐输入），另一个 28-token
请求正确连续输出 `1 2 3 4 5 6 7 8 9 10`，覆盖多步 decode。
但随后 529-token 请求在 DeepEP dispatch 报
`x.size(0) <= num_max_dispatch_tokens_per_rank` 容量断言；
2048-token 与中文探针尚未执行，不能记作长输入通过。

这是配方中 `--deepep-mode low_latency` 与 `--chunked-prefill-size 8192`
不匹配暴露的独立边界。iteration62 把两者对齐为 128 后，在模型加载前
被社区 `chunked_prefill_size % page_size == 0` 校验拒绝：
该模型自动选择 page size 256。这一轮没有执行模型或请求。

配方改为显式固定社区
`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256`，并把 prefill
chunk 对齐到 256；长 prompt 仍由社区调度器分块处理，不截断用户输入。
为控制更大 dispatch capacity 的峰值显存，W4A16 在第二次 GEMM 前释放
已消费的第一投影及 gate/up 临时张量，交给 Torch stream/graph allocator
复用，不改数学运算。组件 graph 回归覆盖每 expert 128/256 行；
真实模型的通信 buffer 及其显存峰值另由端到端重跑验证。
短语义通过仍不代表完整模型精度，修正配方后的长输入与分块 KV/cache
验证须继续实测。

iteration63 在 validation-host-b GPU0 定向运行 EP4 seam、CT W4A8 与两种 GEMM
tier 的数值/变输入 graph 回归：**34 passed，30.47 秒**。
新增 `tests/tools/_check_dsv41_ep4_recipe.py` 直接解析实际启动脚本，
按社区 Engine 的 `resolve_once → check_server_args` 顺序执行预检，
再校验 chunk/最大请求数不超过 LL capacity；不加载模型权重。
256 配方通过社区参数校验，旧 128 配方按预期因 page 对齐拒绝。
预检报告读取社区 `resolving_view`，保留原始声明与解析结果的区别。

iteration64 已将 validation-host-b 代码同步到 `ae24890`，但启动前资源检查发现
8 张卡显存均占用 64%（计算利用率为 0）。控制器在加载模型前拒绝继续，
未产生服务日志或新模型进程，也未清理其他任务。现场记录为
`external-evidence:iteration64-validation-host-b-ep4/attempt.json` 与
`gpu-before.txt`。GPU 隐藏的独立源码审计仍通过，10594 个 tracked file
字节不变、无新增源文件；报告为同目录 `source-current.json`。

截至 iteration64，可确认的是修复前后定向组件、真实短请求和多步 decode 的上述证据；
256 配方的真实重启、529/2064-token 分块请求、中文与批量请求仍未执行，
EP1/EP4 的完整精度对拍也未完成。后续有空闲卡时，先在同一提交和配方
补齐这些验证，不能把参数预检通过外推为长输入或完整模型精度通过。

### iteration65：validation-host-a 四卡 EP4 分块验证（2026-11-05）

使用 adapter `ae24890`、社区 `baa9befa`，物理卡 0–3，沿用 dispatch
capacity/prefill chunk 均为 256 的已提交配方。真实权重加载、四卡 decode
graph 捕获及服务预热通过。三个短请求正确返回 `42`、`Paris`、`63`；
与 iteration61 的相同输入比较，输出 token 和所记录输出 token logprob
完全相同。这只是三个短请求的历史回归，不是独立参考或完整精度验收。

529-token 请求已完成 `256 + 256 + 17` 分块并正确返回 `42`，
耗时 204.58 秒，未复现此前的 DeepEP 容量断言或 OOM。2064-token 请求
执行中因用户需要让出环境而停止，中文、多步序列和批量探针本轮未执行，
不记作失败或通过。停止后 validation-host-a 全部 8 张卡显存和计算利用率均为 0；
运行前后社区全部 10594 个 tracked file 字节不变，无新增源文件。
证据位于 `external-evidence:iteration65-validation-host-a-ep4/`。

组件回归补齐 EP4 的实际容量边界：每 rank 最多 dispatch 256 个 token 时，
单 expert 接收缓冲区可容纳 `4 × 256 = 1024` 行，而原测试仅覆盖
128/256 行。新增 1024 行参数在空闲 GPU4 上定向运行，Triton/vendor
两条路径、channel/group scale、变 counts/空 expert 及变输入 graph replay
共 **4 passed，19.02 秒**。它验证组件数值和 graph 契约，不代表真实模型
所有 expert 同时填满或高并发显存已经通过。

本轮最初的控制器曾轮询社区 `/health`，该版本健康检查会发送一次生成请求，
默认 20 秒超时。发现额外流量后停止该控制器，改用已完成服务预热的日志标志，
再单独执行语义断言；健康检查超时不作为本轮语义测试失败或通过的依据。
保留此前日志，不修改社区健康检查行为或降低模型精度断言。

### iteration66：validation-host-b 四卡预热超时（2026-11-05）

在 validation-host-b 物理卡 4–7 运行 adapter `ae24890` 和同一份社区源码，
其余卡已有任务。真实权重加载和四个 rank 的 decode graph 捕获完成，
6-token 正常启动预热的 prefill 完成，但整个预热请求超过社区默认
600 秒超时，启动器随后终止自身进程树，退出码 137。语义控制器尚未
启动短输入、长输入或批量请求；本轮不能记录为模型启动或语义通过。
日志未观察到 OOM 或此前的 DeepEP dispatch capacity 断言，
但单凭超时也不能排除设备或通信停滞。

运行前后社区 10594 个 tracked file 字节不变、无新增源文件。
摘要及源码审计位于 `external-evidence:iteration66-validation-host-b-ep4/`。
下一轮诊断保留正常预热，通过社区公开的 `SGLANG_WARMUP_TIMEOUT=1800`
延长等待，并用 `--decode-log-interval 1` 记录每步是否推进；这两个
诊断参数不构成修复已验证或模型精度通过的证据。

### iteration68：将已知 EP 边界接入常规矩阵（2026-11-05）

validation-host-b iteration67 继续预热期间，将已有的三项 GPU 回归入口纳入
`tests/manifests/l15_family_matrix_v1.json`：请求补齐后的 graph 变输入及
有效请求隔离，以及 raw Triton INT4 在 expert capacity 1024 下的 group/
channel scale、clamp、变 counts 与变输入 graph replay。容量对应 EP4
每 rank dispatch 256 的接收上界；这些仍是单卡组件测试，不证明四卡
通信、真实模型所有 expert 同时填满或高并发显存通过。矩阵选择实际
服务使用的 Triton 路径，vendor 比较路径继续留在完整 adapter 测试中。

清单现在为 **35 个入口**：Dense 3、MLA 5、MoE 10、量化 10、Mamba 7。
validation-host-b 在独立临时目录、HOSTnone/隐藏 GPU 下确认新增三个精确 pytest
nodeid 均可收集，未更改运行中服务的 checkout；此收集结果不是 GPU
执行通过。矩阵工具 CPU 回归 **32 passed**，清单校验通过，L0 GREEN
（31 条既有 warning）。此前 iteration60/65 的组件执行证据继续保留，
35 项完整矩阵尚未重跑，不能将历史 32/32 结果转授为新清单通过。

### iteration67：保留正常预热并完成四卡启动（2026-11-05）

validation-host-b 物理 GPU 4–7、adapter `ae24890`，诊断启动仅增加社区公开的
`SGLANG_WARMUP_TIMEOUT=1800` 和逐步 decode 日志，其余沿用 256 配方。
四个 rank graph 捕获通过，正常预热完成全部 7 个 decode batch，
12:37:07 出现服务就绪标志；从 HTTP 开始监听到就绪共 **1555 秒**。
因此 iteration66 的 600 秒等待不足，本轮已确认设备持续推进并完成
正常预热，不能将此前超时直接归为 OOM 或容量断言复发。

四卡启动配方在未设置时导出社区变量 `SGLANG_WARMUP_TIMEOUT=1800`，
保留用户显式设置；不改 adapter 的全局默认或社区文件，不跳过预热。
逐步 decode 日志仅为本轮诊断参数，没有加入默认配方。
更新脚本通过远端 `bash -n`，L0 GREEN（31 条既有 warning）。

`external-evidence:iteration67-validation-host-b-ep4/startup-summary.json` 保存
实际服务进程参数、所选环境变量、诊断脚本 SHA256、四卡 graph 和就绪
时间。启动前完整性通过；2064-token 长输入探针已开始，后续语义结果、
运行后源码审计和完整精度验收仍待完成。该启动快照不代表整轮通过。

本轮结束后的完整结果：三个短请求均通过，分别返回 `42`、`Paris`、
`63`，与 iteration65 相同输入的输出 token 和所记录输出 token logprob
一致（最大差值 0）。这只是历史 W4A16 功能回归，历史运行没有前后
checkpoint 字节绑定，不能作为独立精度或 W4A8 等价验收。
2064-token 请求在客户端 900 秒预算内未返回；529-token、中文和多步
序列未执行。批量请求在服务 80 分钟总预算结束时断开，未取得结果，
服务退出码为 124。长输入日志可见分块推进，未观察到 OOM 或 dispatch
capacity 断言，但未完成请求不能记为通过，也不能据超时判为数值失败。

严格验证器将整轮 `semantic_regression_passed` 记为 `false`，保留传输
错误及缺失用例。社区源码运行前后全部 10594 个 tracked file 字节不变，
无新增源文件。本轮服务及 worker 已退出，清理后物理 GPU4–7 显存和
计算利用率均为 0；未停止 GPU0–3 的其他任务。完整摘要、原始语义响应、
历史比较及清理证据位于同目录。下一轮需增加请求与总运行预算，补齐
长输入、中文、多步和批量，不能用脚本退出码 0 代替探针结果。

### W4A8 激活量化契约差距（2026-11-05）

当前 checkpoint 配置的 routed/MTP 专家声明 channel INT4 权重
（`group_size=-1`）和 dynamic、symmetric、per-token INT8 输入激活。
实际 `HCU_W4A8_MOE=w4a16` 路径将这些组的 `input_activations` 置为
`None`，保留 INT4 权重但使用 BF16 激活。此前“数值更保真”的表述缺乏
整模型独立参考证据，已移除；W4A16 的组件数值、启动和语义回归不能转授
为原 W4A8 数值语义一致或精度更高。

社区 `baa9befa` 的 `_is_wint4afp8` 同时接受 INT/FLOAT 的 8-bit 动态
激活，后续选择 `CompressedTensorsW4AFP8MoE`。该 scheme 的
`create_weights` 直接以 `hidden_size // group_size` 分配尺度；对本配置
的 `-1` 会得到负维度，因此 `heal` 只补 `Linear` 键尚不足以执行该模型，
不能直接将后续差距全部归为缺少 CUTLASS。以上为配置和原始源码审计，
没有执行新的 GPU 数值测试，也没有建立有效的社区模型精度基线。

继续跟踪 scheme 的标准 `apply_weights → cutlass_w4a8_moe` 调用后，
确认其输入和 SiLU×up 中间张量均分配为 `float8_e4m3fn`；
`a13_scale/a2_scale=None` 时分别计算 tensor 级动态尺度，第二次 GEMM
前的尺度为 `max(abs(SiLU(gate) * up), 1e-10) / 448`。输入重排使用
同一标量尺度缩放所有 token 后转换 FP8，两个 GEMM 的 group_size
均硬编码为 128。因此配置分类接受 INT8，不表示执行的是 per-token
INT8；只增加 INT8 QDQ 也不能宣称还原了这条社区 FP8 路径。
这些结论来自未修改源码的静态调用链，不证明当前 channel checkpoint
可执行，也没有将标准路径的结论外推为 DeepEP LL 已通过。
具体源码 SHA256 和函数位置见同目录 `kernel-contract.json`。

后续必须核对配置声明与社区 scheme/实际 kernel 的激活 dtype、动态尺度、
舍入和 clamp 契约，恢复可验证的 W4A8 路径，并补原始权重的组件/层级
独立参考、变输入 graph 和真实模型分布验证。当前 W4A16 结果继续作为
该路径的功能基线；每日追新仍等待这些功能与精度缺口补齐。

### iteration71：checkpoint 声明的 W4A8 层级 CPU 参考（2026-11-05）

新增 `tools.ct_w4a8_reference`，为 channel INT4 权重、动态 per-token
INT8 激活提供独立 CPU FP32 Linear/单专家参考。它不选择服务 scheme，
也未加入整模型 `--quantized-reference` 的支持列表。INT32 打包按低位
nibble 在先、unsigned offset 8 解码；两个 Linear 边界都执行激活 QDQ，
保留可选 SwiGLU clamp。仅接受已核对的量化参数和有限正 channel scale。

validation-host-b 隐藏 GPU 的 **26 项 CPU 测试全部通过**，与官方
`compressed-tensors 0.17.0` 的打包解码、动态尺度、整数码与 QDQ 对拍，
包含零行、半整数舍入、不同 token 幅度、非整 pack 长度、符号位、
双 GEMM 及参数拒绝。该库 FP32 对称 INT8 使用 `[-128, 127]`，
尺度为 `absmax / 127.5`、ties-to-even；不能直接复用现有 SGLang
INT8 路径的 `/127` 数值契约。参考显式要求 `[batch, token, hidden]`，
拒绝 2D，因为此库的 token reducer 对 2D 会选择 tensor 级尺度。

另用真实 checkpoint 的 `layers.0.ffn.experts.0` 三份权重，输入形状
`[1, 3, 5120]`（合成幅度 1、1/32、零），完成无 clamp / clamp=0.75
两组 CPU 专家对拍，与官方库 QDQ + 解包权重 + FP32 Linear 的输出
精确相同。配置、索引及所选 shard 在探针前后实际读字节计算 SHA256，
均匹配 iteration70 的预先清单；未将此抽样核对外推到其他 shard。
证据位于 `external-evidence:iteration71-validation-host-b-w4a8-reference/`。

这是为恢复 checkpoint 声明的 W4A8 数学建立的层级参考，尚未执行
HCU W4A8 内核、routing/combine、真实推理隐藏态或整模型分布验收，
也不代表社区 tensor-FP8 路径已还原。W4A16 服务及其既有结果保持原有
验收范围，完整目标继续推进。

### iteration69：35 项家族矩阵完整复跑（2026-11-05）

validation-host-b 物理 GPU4、adapter `262f256`、社区 `baa9befa` 的严格矩阵
**35/35 入口通过**，退出码 0：Dense 3、MLA 5、MoE 10、量化 10、
Mamba 7。其中 33 项为 GPU 契约、2 项为 CPU 契约；pytest 明细包含
62 个 test item 和 50 个 subtest，不将它们相加冒称独立入口数。
逐项重查实际 nodeid、call phase、所有 phase/subtest 均通过，
无 skip/xfail、超时或收集错误。三个新增 EP 边界入口已实际运行。

运行前后社区全部 10594 个 tracked file 字节不变，无新增源文件。
结果位于 `external-evidence:iteration69-validation-host-b-matrix/`：
`outcomes.json` 保留全部 pytest 明细、环境与源码审计，并绑定原始完整
报告和 stdout/stderr 的 SHA256；原始 `matrix.json` 与日志本地保留。
这补齐新 35 项清单的执行证据，不等于全量 adapter 套件、四卡通信、
真实模型长输入/批量或原始 W4A8 精度验收。此前模型测试的未完成项
仍须单独补齐。

### iteration72：声明式 W4A8 的独立设备组件（2026-11-05）

新增候选 `hcu_adapter.kernels.w4a8_int8_triton`，暂不接入服务 scheme。
第一步为 expert-major 有效行执行 FP32 `absmax / 127.5`、ties-to-even
INT8 量化；第二步直接读取低位 nibble 在先的 INT4 原始字节，以
INT8×INT4 的 INT32 累加完成全 K 归约，再乘 token/channel FP32 尺度。
不缓存完整反量化权重，也不将 INT8 激活转换为 BF16 后再执行 dot。
仅支持有限有效输入及有限正尺度，padding 的整数码/输出为零、尺度为 1；
全零的有效行使用 FP32 epsilon。有效行数留在设备上，graph 重放重新读取。

候选在远端 `/tmp/adapter-w4a8-kernel72` 隔离执行，没有改模型 checkout。
先完成 7 个 gfx938 AOT 编译变体和 2 个 CPU Triton interpreter 对拍，
包含真实 K=5120 形状的编译和半整数舍入；离线证据不作为设备通过依据。
iteration70 清理后重新检查 GPU4–7 空闲，使用物理 GPU4，设备专项
**13 passed，约 22 秒**，无 skip/xfail 或收集错误。覆盖 FP32/BF16/FP16
输入、FP32/BF16 权重尺度、部分 tile、capacity=1024、空专家、NaN padding、
全零有效行，以及捕获后改变输入、尺度和 counts 的重放。整数码和有效尺度
精确相同、relative L2 < 2e-5、cosine > 0.99999，未放宽断言。
首次尝试因诊断记录器的 pytest hook 参数名错误而在执行前失败，实际执行
0 项；修正记录器后完成上述 13 项，保留原失败记录，不记为内核数值失败。

`external-evidence:iteration72-validation-host-b-w4a8-kernel/` 保存逐项/逐 phase
结果、脚本及编译/interpreter 报告。模型 checkout 基于 `51ad82e`，
候选不是该 commit 的已提交代码；设备报告绑定实际候选内核和测试文件的
前后 SHA256，二者均不变。catalog 仍标为 candidate，尚未接入服务。

此组件遵循 checkpoint 声明的 token-INT8 数学，不能称为社区
tensor-FP8 路径的复现。服务端专家链接线、routing/combine、真实隐藏态及
整模型分布验收仍须继续；当前 W4A16 模型的语义回归不转授为此组件通过。

### iteration73：真实权重单专家的 W4A8 设备链（2026-11-05）

沿用 iteration72 实际测试过的内核字节，在 validation-host-b 物理 GPU4 执行
`layers.0.ffn.experts.0` 的 gate/up/down 三份真实权重：
5120→2304→5120，每个 Linear 边界均执行动态 token-INT8 量化，
中间为 FP32 SwiGLU。独立 CPU FP32 oracle 的打包/量化契约已在
iteration71 对拍官方 compressed-tensors；本轮直接比较完整专家输出，
没有以 GPU 自身输出构造期望值，也未放宽 iteration72 的数值阈值。

无 clamp / clamp=0.75 两组各覆盖 eager、初次 graph 重放、全空有效行、
改变输入及 counts、重新激活，共 **10 组检查通过**。两个输入阶段均为
`[1, 4, 5120]`，有效行数分别 3 和 2；输入为可由 BF16 表示的合成值、
不同幅度及全零行，padding 填 NaN，不是实际推理隐藏态。有效输入的
整数码和尺度精确一致；完整专家最大 relative L2 为
**8.445362e-7**，非零输出最小 cosine 为 **0.99999988**，约 16.6 秒。

配置、索引及所用 `model-00003-of-00048.safetensors` 在 CPU 准备数据前后
实际读取字节，均匹配 iteration70 预先清单。生成的 CPU 参考/权重资产
另绑定 SHA256，GPU 执行前核对该资产及候选内核字节，避免换权重后沿用
期望值。证据为 `external-evidence:iteration73-validation-host-b-real-w4a8-expert/`。
未将所选 shard 的校验外推为全 checkpoint 本轮审计，也不将单专家
FP32 中间计算通过外推为服务的 BF16 存储、routing/combine、其他专家、
真实隐藏态、社区 tensor-FP8 或整模型分布精度通过。

### iteration70：四卡完整有限语义回归与权重绑定（2026-11-05）

validation-host-b 物理 GPU4–7，实际 adapter `51ad82e`、社区 `baa9befa`，
真实 DeepSeek-V4.1-Flash-Channel-INT4-w4a8 checkpoint，
执行路径仍为 **W4A16**。TP4/EP4、DeepEP LL、dispatch/chunk/page=256、
mem_fraction_static=0.90、max_running_requests=16，保留正常预热；
本轮诊断显式设置 `SGLANG_WARMUP_TIMEOUT=3600`，不改配方默认 1800。
四个 rank 的 `[4, 8, 12, 16]` decode graph 捕获完成，预热 7 个 decode
batch，HTTP 监听到 ready 为 325 秒；不对较此前加速的原因作无证据归因。

三个短请求分别返回 `42`、`Paris`、`63`，均通过，耗时约 157/163/166 秒。
四项扩展全部通过：中文返回 `Paris`（82 秒）、529-token 返回 `42`
（288 秒）、2064-token 返回 `63`（568 秒）、连续输出 1 至 10
（1168 秒）。两条长请求的实际输入共享前 512 token，page_size=256，
第二条请求实际命中 512 token，仍返回不同且正确的答案，覆盖缓存复用
及输出隔离。批量 `[21, 21, 20]` 输入的三条答案全部通过（267 秒），
日志实际出现 `#running-req: 3` 且 `cuda graph: True`，不只验证 HTTP
数组接收而未验证三请求 decode。

严格验证器独立重查答案、长度、批量输入、共享前缀/缓存命中、四 rank
捕获、实际配置、controller、清理、源码和权重绑定，整轮
`semantic_regression_passed=true`。10 组纯合成验证器 fixture 另确认：
错误答案即使自报 true、传输错误、缺项、缓存未复用/超出前缀、权重变化、
批量输入错配和清理失败都不能放行；fixture 通过不作为设备证据。

运行前后社区全部 10594 个 tracked file 字节不变、无新增源文件；
checkpoint 的 **54 份资产**在本轮执行前建立清单，执行后重新读字节
验证一致，不能把这份绑定倒授给 iteration65/67。全部探针完成后，仅对
本轮记录的服务/worker 身份发 SIGTERM；社区对子进程退出 -15 触发
SIGQUIT 和自身 kill_process_tree，最终 server.rc=137。该退出发生在
最后一次成功响应之后，原始尾日志与清理证据保留；不是把非零退出
伪装成自然退出 0。最终身份审计无残留服务或 worker、无 zombie。

完整结果和可复查脚本位于
`external-evidence:iteration70-validation-host-b-ep4/`。本轮仍有 14 条运行后
device-load/低剩余显存提示，未观察到实际 OOM 异常；不将这些提示抹去，
也不据此声称已覆盖所有内存边界。这补齐 iteration67 未完成的长输入、
中文、多步与批量功能回归，仍不等于原始 W4A8、独立精度参考、多模型
或全量 adapter 套件验收通过。

### iteration74：专家接口原型的量化边界差异（2026-11-05，未通过）

新增未提交原型将 gate/up 合并投影、FP32 SwiGLU、二次 token-INT8 量化
和 down 投影收口为 `w4a8_int8_masked_experts`。validation-host-b GPU4 上重跑原 13 项，
另加 6 项专家链（3 种输入 dtype × 无 clamp / clamp=0.75），结果为
**16 passed、3 failed**：原组件和有 clamp 的链通过，无 clamp 的三种
输入 dtype 均在改变输入/counts/尺度后的检查失败。原型源代码和测试的
实际字节另以 `.txt` 快照绑定；服务与已提交组件仍保持原有范围。

对 FP32 输入失败相位另做 eager 复现：第一段设备 GEMM 与独立 CPU
INT64 整数点积、FP32 token/channel 缩放精确一致；与 CPU 反量化 FP32
GEMM 最大差值为 3.337860e-6。第二次量化的一个值分别落在
`-127.5000076` 与 `-127.4999924`，整数码为 -128 与 -127，使最终专家
输出 relative L2 为 9.765185e-4，超过原 2e-5 门槛。对相同设备中间值
在 CPU 重新量化，整数码精确一致，因此该复现未指向量化内核对同输入
的舍入错误；eager 也有输出差异，不能直接归为 graph 重放独有故障。

完整失败、候选快照和逐段诊断见
`external-evidence:iteration74-validation-host-b-w4a8-experts/`。这里只定位了
FP32 样本的累加精度与后续量化非连续边界，尚未证明所有失败的完整归因，
也未证明哪个整模型数值路径满足用户目标。接下来需核对整数累加、FP32
fake-quant Linear 和实际服务存储的参考契约；保留原 CPU oracle 和失败
记录，继续建立独立验证。当前不把 16/19 或 iteration73 的单专家通过
外推为专家接口或完整 W4A8 链已通过。

### iteration75：独立整数执行 CPU 参考（2026-11-05）

保留 `linear` / `expert` 原 FP32 fake-quant 行为，另增显式
`int32_linear` / `int32_expert`。后者由 INT64 点积独立模拟无溢出的
INT32 完整累加，随后转 FP32，依次乘 token/channel 尺度；中间激活
仍为 FP32，不声称覆盖实际 BF16 存储。K > 2097151 显式拒绝。
validation-host-b 的 HOSTnone、隐藏 GPU 进程共 **44 passed**，包含原 26 项
和新 18 项，132 个 setup/call/teardown 相位均通过，GPU 未初始化。
新回归以 CT 0.17.0 官方整数码/打包、FP64 精确整数点积及 Python
整数和交叉检查，包含超过 2^24 的完整累加、非整字边界和两种 clamp。

证据位于 `external-evidence:iteration75-validation-host-b-integer-reference/`。
附带固定社区 `baa9befa` 的 generic INT8 GEMM 源码审计：文件与 git blob
逐字节一致，声明 INT32 accumulator、FP32 compute。这仅支持通用整数
执行口径，不证明 W4AFP8 scheme、原 W4A8 模型语义或模型精度等价。
iteration74 原 FP32 参考的 3 项失败未豁免；两种参考分别报告。

### iteration77：专家链的整数执行补充合同（2026-11-05）

validation-host-b GPU4 空闲检查后执行，与 iteration74 相同原型字节和输入生成器，
另设显式调用 `int32_expert` 的 6 项补充测试；没有替换原 `expert`。
FP32/BF16/FP16 输入 × 无 clamp / clamp=0.75 全部通过，18 个测试相位
均通过，耗时 15.36 秒。覆盖 eager、graph、变输入/counts/尺度和空专家，
保持 relative L2 < 2e-5、cosine > 0.99999 原阈值。候选与参考运行前后
哈希一致，实际 Torch 2.11.0、Triton 3.6.0、BW1100。

证据与独立测试快照位于
`external-evidence:iteration77-validation-host-b-integer-expert-contract/`。
这是合成专家链符合整数执行合同的证据，不能追认 iteration74 的
FP32 fake-quant 对照通过，也未覆盖真实路由、combine、BF16 中间存储、
服务接线或模型精度；原型仍未晋级为已验证服务实现。

### iteration79：真实 checkpoint 专家的整数执行对照（2026-11-05）

在 validation-host-b GPU4 复用已绑定的 `layers.0.ffn.experts.0` 权重，
5120 → 2304 → 5120，验证同一未晋级专家接口原型。运行前独立 CPU
隐藏 GPU 进程重新生成整数累加期望，保留 iteration73 的原 FP32
fake-quant 期望。原资产 SHA256 校验一致；config、index 与所选 shard
本轮前后重读字节校验，与 iteration70 前瞻绑定的对应资产一致。
绑定范围仅为所选资产，不能外推至整个 checkpoint 的新一轮验证。

无 clamp / clamp=0.75 各覆盖 eager、graph 初始、空专家、变输入及
有效行数、重新激活，共 **10/10 通过**；设备输出与独立整数参考
逐元素一致（max abs delta=0）。CPU 参考构建 17.75 秒，设备检查
29.08 秒。两相位有效行数 3 → 2，输入为合成 BF16 可表示值，
中间激活及最终输出仍为 FP32。整数参考与原 FP32 参考在这些样本的
relative L2 为 4.03e-7～8.45e-7，不代表 iteration74 的其他样本失败消失。

证据及脚本位于
`external-evidence:iteration79-validation-host-b-real-integer-expert/`。
本轮继续保留原 FP32 门禁；尚需验证实际 BF16 存储、路由权重、
dispatch/combine、服务 scheme/loader 接线及整模型独立精度。

### iteration82/83：CT 原 dtype 的 token-INT8 组件（2026-11-05）

核对官方 compressed-tensors 0.17.0 发现：BF16/FP16 的原输入执行会
保留尺度、除法和 QDQ 的存储 dtype，零行分别使用 BF16 epsilon
0.0078125、FP16 epsilon 0.0009765625。原组件显式采用 FP32 量化，
零行使用 FP32 epsilon，不能自动作为前述两种输入 dtype 的 CT
原生执行替代品。这是执行精度契约的差距，不把原 FP32 组件判为错误。

保留原 API，增加 CPU `token_int8_native` 与独立设备组件
`token_int8_native_masked`。validation-host-b 隐藏 GPU、HOSTnone 进程共 **58 passed**
（原 44 + 新 14），官方 CT 的整数码、尺度及 QDQ 逐元素对齐。
GPU4 **12 passed**：6 组 dtype/尺寸的非连续输入与变输入/counts
graph、3 组空容量、3 组原 dtype 量化接原整数 GEMM。后者先显式
将已舍入尺度提升 FP32，再与独立 FP64 精确整数点积、FP32 缩放对齐，
输出逐元素一致；不把反量化浮点 GEMM 作为整数点积参考。测试覆盖
NaN padding、零行、空专家、恢复有效行以及变尺度。CPU 5.15 秒、
设备 16.50 秒，全部 setup/call/teardown 通过且候选字节前后不变。
另在同节点隐藏 GPU 运行 `tests/test_ops_cpu.py`，38 passed。

证据位于 `external-evidence:iteration83-validation-host-b-native-int8/`。
原 FP32 专家链失败未豁免；原型未接入该新组件。下一步仍需确定并
验证专家 GEMM/SiLU 的中间存储、DeepEP combine dtype、正式 scheme
加载和整模型精度，不能将本轮组件通过视为完整 W4A8 验收。

### iteration84：分别存储激活的专家原型（2026-11-05，仅诊断）

候选显式整数参考与 `w4a8_int8_native_masked_experts` 原型，在完整 INT32 点积转 FP32、
依次乘 token/channel 尺度后，将投影存为输入 dtype；SiLU 与乘积也
各自保留该 dtype，二次量化使用 iteration83 的原 dtype 合同，最终
输出恢复输入 dtype。该合同不等于 CT fake-quant 浮点 Linear，也不等于
原 FP32 中间态专家；保留所有旧参考及 iteration74 失败。

validation-host-b HOSTnone/隐藏 GPU 的 CPU 参考 **69 passed**（原 58 + 新 11），
新专家参考用官方 CT 整数码和独立 FP64 整数点积交叉核对存储边界；
GPU4 **18 passed**（原 12 + 新 6），覆盖 FP32/BF16/FP16 × 无 clamp /
clamp=0.75 的 eager、graph、空专家、变输入/counts/尺度及重新激活。
原 relative L2 < 2e-5、cosine > 0.99999 保持不变。CPU 5.49 秒、
设备 19.28 秒，所有测试相位通过，候选字节前后不变。

证据位于 `external-evidence:iteration84-validation-host-b-native-experts/`。
该候选仅验证 SiLU 和乘积各自舍入的合同。后续核对发现社区存在融合
SiLU×up 路径，不能用本轮候选代替融合存储语义；候选保留为字节绑定的
诊断快照，代码未晋级服务实现。尚未接入正式 scheme/loader、
dispatch/combine 或整模型；不能据此宣称完整 W4A8 或模型精度通过。

### iteration86：融合激活存储的独立 CPU 参考（2026-11-05）

公开 CPU `int32_native_linear` / `int32_native_expert` 明确整数累加后
FP32 缩放、按输入 dtype 存储投影，以及 FP32 SiLU×up 后一次转回输入
dtype 的融合乘积边界。它不同于 iteration84 的两次激活存储，也不同于
CT fake-quant 浮点 GEMM、原 FP32 中间态参考或社区 tensor-FP8 scheme。
保留这些参考及原失败记录，不用改参考追认模型通过。

validation-host-b HOSTnone/隐藏 GPU 进程 **69 passed**，207 个测试相位通过，
CPU 5.74 秒，候选文件前后字节不变，GPU 未初始化。与官方 CT 0.17.0
整数码和独立 FP64 精确整数点积交叉核对，覆盖三种 dtype、两种 clamp
及非法 clamp 拒绝。证据及设备候选快照位于
`external-evidence:iteration86-validation-host-b-fused-native-experts/`。

固定社区源码 `cutlass_w4a8_moe.py` 与 git blob 一致，记录了 normal
路径的 BF16 intermediate + silu_and_mul 调用和 masked 的融合 FP8
量化调用；两条路径不同，不能将该源码观察当作原生 INT8 服务等价证据。
设备候选在 `/tmp` 等待约 301 秒，GPU4 显存与利用率均空闲后启动。
**18/18 通过**，54 个测试相位全部通过，无 skip/xfail；耗时 15.22 秒，
候选字节前后不变。包含新增 6 项原 dtype 融合存储专家的 eager/graph、
空专家、变输入/counts/尺度与重新激活，保持原 relative L2 < 2e-5、
cosine > 0.99999。正式服务接线、DeepEP combine 与整模型精度仍待验收。

### iteration87：真实权重的融合 BF16 专家合同（2026-11-05）

同一 `layers.0.ffn.experts.0`、5120 → 2304 → 5120，使用
iteration86 的显式整数执行与融合 BF16 存储。CPU 参考在隐藏 GPU
进程重新建立，所选 config/index/shard 前后重读字节匹配原前瞻绑定，
原资产 SHA256 校验一致。GPU4 的 eager、graph 初始、空专家、
变输入/counts、重新激活 × 两种 clamp 共 **10/10 通过**；
输出为 BF16，与独立整数参考逐元素一致，max abs / relative L2 均为 0。
CPU 13.95 秒、设备 29.82 秒，候选字节前后不变。

保留与原 CPU FP32 fake-quant 期望的独立比较：relative L2 为
**0.0181～0.0259**，说明 dtype/量化执行合同产生了可观差异，不能
将本轮逐元素通过追认为原 FP32 参考通过或模型精度通过。当前输入仍为
合成 BF16 值，绑定仅覆盖所选专家资产，不是完整 checkpoint 新验收。
证据位于 `external-evidence:iteration87-validation-host-b-real-fused-native-expert/`。
正式 loader、dispatch/combine、真实隐藏态和模型分布验证仍待补齐。

### iteration93～97：validation-host-a 真实加载及显式 INT8 候选接线（2026-11-05）

按用户最新指定切至 validation-host-a `adapter-validation`，使用物理 GPU4；
分配前 VRAM% 与 HCU% 均为 0。镜像残留的旧 SGLang/sgl_kernel
包与相应 metadata 已移到容器 `/tmp/adapter-validation-host-a-image-packages95`
备份，保留恢复清单。测试使用完整 adapter 快照加候选改动，
不是人工重设 bootstrap 状态或只注入 helper 模块。

新增显式 `HCU_W4A8_MOE=int8` 候选，保留原 regex 匹配及激活量化元数据，
复用官方 `CompressedTensorsWNA16TritonMoE` 的分配和 raw INT4 转换，
以独立 marker 接入 DeepEP LL core。默认 W4A16 不变；标准 dispatcher、
非 BF16、非对称量化、实际零点权重及未验证 runner 参数明确拒绝。
官方 loader 的单 packed-column 维度存在 stride/view 限制，当前 K/I
要求至少 16 且为 8 的倍数。iteration93 实际转换逐字节匹配且幂等。

iteration94 用真实 checkpoint config 解析发现：CT 0.17.0 将未指定的
对称 `zp_dtype` 规范化为 `torch.int8`，原 namespace doubles 未覆盖，
候选误拒绝。已修复并增加真实 `QuantizationArgs` 回归；
多符号注册也在全部成功后才发布，第二项注册冲突时回滚记录并可重试。
iteration96 的 L0 因 catalog 改动后的 tier-matrix 未同步失败，已重新生成，
保留原失败日志。

iteration97：**66 项 CPU 接缝回归、38 项 CPU 算子、18 项 HCU 原 dtype
组件均通过**，L0 GREEN（34 条既有警告）。全新进程使用真实 config
选中候选 scheme，保留无 `"Linear"` 的 regex 桶与激活配置，确认实际
SGLang 和影子包导入来源。官方 loader 加真实 DeepEP LL dispatch/combine
数据类型的 **10 项 eager/graph 检查全部通过**，涵盖两种 clamp、
空专家、变输入/counts/尺度及重新激活；合成 BF16 输出与独立 CPU
整数参考逐元素一致，topk IDs/weights 对象保留。这里手工构造了
dispatch output，没有执行真实通信或整模型。

社区固定 commit `baa9befa381511cd4d56f83980951e5a45fffc6d` 的
10594 个跟踪文件在专项前后逐字节核验，变化与额外源码均为 0。
真实模型 `text_config.routed_scaling_factor=1.5`，超出目前 None/1
范围；现代 quant-method 的 apply 入口、inplace 提示、真实 dispatch/combine、
真实隐藏态与模型分布仍需逐项补齐。原 FP32 专家失败及 iteration87
相对原 fake-quant 的 1.8%～2.6% 差异保持未通过，不做精度豁免。

### iteration99：诊断汇报不再将跳过计为通过（2026-11-05）

validation-host-a 扩大 ops 诊断实际结果：内核 254 passed / 2 skipped / 2 xfailed，
FA 53 skipped，AR gate 2 passed / 1 skipped / 1 deselected；RMS、
GEMM 与 bridge graph 探针通过，RoPE 因默认旧 checkpoint 路径不存在失败，
原退出码为 1。保留原始日志 gzip；不是全量 adapter 或模型验收。

发现 `_pytest` 仅根据退出码标 PASS，导致全部跳过的 FA 组也显示 ok，
总 verdict 还会对跳过的组输出 ALL GREEN。现改为：实际通过且没有
skip/xfail/xpass 才标 PASS；全跳过标 SKIP，混合或缺少通过计数标
INCOMPLETE。总退出码 0=通过、1=失败、2=验证不完整；无实际通过或
只有 INFO 同样不能绿。主动 deselected 不影响所选测试的通过判定。

独立 HOSTnone、三种 GPU 可见性均隐藏的进程中，15 项汇报回归及
38 项 CPU 算子共 **53 passed**，GPU 未初始化；L0 GREEN（34 条既有
警告），候选字节前后匹配。没有借此重跑或追认原设备套件。
证据位于 `external-evidence:iteration99-validation-host-a-triage-completeness/`。

### 上游同步后的完整矩阵及全量回归（2026-11-06）

固定同步候选 `d237c09` 对应的不可变运行快照，完整执行 51 项
L1.5 矩阵：Dense/MLA/MoE/Quantization/Mamba 为 7/17/10/10/7，
全部通过。实际 104 个参数节点、58 个 subtest，所有
setup/call/teardown 通过，无 skip/xfail、收集错误或超时。
前后社区 10594 个跟踪文件及额外源码审计通过；终结后核对候选
602 个快照文件摘要，全部匹配。这是一次完整 51 项运行，
不是先前 50 项与独立新增用例的拼接。

同一旧快照的全量 kernels/adapter/vendor（排除 stress）结果为
1541 passed、51 failed、137 errors、79 skipped、2 xfailed、
1 deselected。按实际堆栈分类，47 个失败及 137 个 setup 错误
属于暂缓处理的 vLLM；4 个非 vLLM 失败均位于 FA3 paged 路由，
其 packed 输出布局及 decode out 修复已另经完整 FA3 25 项验证，
不能倒推本轮全量变绿。Conv1D pytest 收集修复也不在该旧快照中。

本轮 ops triage 同样未通过：kernel 子集为 4 failed / 300 passed /
2 skipped / 2 xfailed，RoPE 探针缺少默认 checkpoint；FA 场景组
53 skipped，AR gate 2 passed / 1 skipped / 1 deselected。
这些缺少验证的项不能记为通过。全量前后社区源码审计通过，
候选字节保持不变；后续须在修复版本验证实际受影响路径。

完整原始报告、逐项相位、日志、源码审计和后置摘要绑定保存在
`external-evidence:matrix257-full261-fa264-complete-raw.json.gz`，
SHA256 `47707911280677faf2fae4ce97f8ef64d39038f1078e4c8530a0eedd2f87622a`。
上述矩阵和组件证据不代表多模型启动或原 BF16 FP32 V1 精度通过。

### 同步后混合架构及 CT-FP8 完整 profile 复验（2026-11-06）

在合并 Conv1D、FA3 和 prefill 登记修复的不可变快照上，分别运行
真实 Qwen3.5-2B-Base 与 Qwen3-0.6B-Channel-FP8。两者均保持
TP1、默认 BF16，完整 accuracy profile 的 eager/graph 生成通过：
短中英文、512/2048 token、缓存复用及变输入，各三批双请求。
每模型六对请求的完整输入/输出 ID、所选及全部报告的 top-20
概率逐位一致。社区 0.5.22 commit
`baa9befa381511cd4d56f83980951e5a45fffc6d` 的 10594 个跟踪文件
及额外源码前后审计通过，候选 603 文件摘要不变，权重资产在
controller、worker 和参考执行间绑定一致。

独立隐藏 GPU 的 CPU FP32 eager 使用 Torch 2.11.0、
Transformers 5.12.1。Qwen3.5 使用官方
`Qwen3_5ForConditionalGeneration` 的 Torch fallback，绑定 10 项
资产；96 个位置 top1 96/96、greedy regret 0，所选 logprob
最大差 0.069363、top20 最大差 **0.362125**、top20+tail JSD
0.001246、tail mass 误差 0.013135、missed mass 0.001279。
原 V1 **FAIL**，只有 top20 logprob 指标越界；历史短输入探针的
通过不能替代本轮完整 profile。

CT-FP8 绑定 9 项资产，官方 compressed-tensors 0.17.0
`ct-fp8-token-channel` 参考逐个核对 507 个 tensor、196 个
Linear，并在五次不同输入 forward 观察全部逐层激活 QDQ。
96 个位置 top1 96/96、regret 0，所选最大差 **0.235903**、
top20 最大差 **2.673436**、top20+tail JSD **0.048732**、
tail mass 误差 **0.056121**、missed mass 0.003061。
原 V1 仍为 **FAIL**；本段指标取本轮原始报告，不沿用历史数值。

Cookbook 仍固定 `40adc4d30ec2640e3391cfd78759bd3598e95f71`，
分别参考 `docs/model-deployment/sglang/qwen3.5.md` 的 27B TP2
家族条目与 `docs/model-deployment/sglang/qwen3.md` 的 0.6B BF16
单卡条目；没有本轮 2B-Base 或 Channel-FP8 的同名精确配方。
实际 Engine API、Triton attention、page1、KV auto、mem0.25、
context4096/max total tokens16384、eager/graph；混合模型使用
mamba extra_buffer。没有执行图像、NEXTN、HTTP 或解析器验证。

原始报告、配方差异、各模式完整概率、终结后比较绑定及源码/权重
审计归档为 `external-evidence:models269-270-complete-raw.json.gz`，
SHA256 `81875e6c89696b1f97d6527f302660bcfb3a549eb22f0159131b841def5a97c4`。
这补齐同步后两条真实运行链的证据，不构成多模型精度验收通过。

### 合并修复后的全量回归（2026-11-06）

固定含 FA3、Conv1D 和 prefill 登记修复的 603 文件快照，执行
完整 kernels/adapter/vendor（排除 stress）：**1565 passed**、
47 failed、137 errors、79 skipped、2 xfailed、1 deselected。
逐条核对 JUnit 和堆栈，47 个失败、137 个 setup 错误均属于
用户暂缓的 vLLM，非 vLLM 失败与错误为 **0**。这不把暂缓项
改记通过，也不把整个套件记作全绿。

后续 ops 的 kernel 子组为 316 passed / 2 skipped / 2 xfailed，
FA 场景组 53 skipped，AR gate 2 passed / 1 skipped /
1 deselected；旧快照的 RoPE 默认 checkpoint 路径仍失败。
RoPE 入口后续修复的独立 55 项 CPU 和 8 项 HCU 验证见精度
定位手册，未改写正在运行的旧快照或追认其 ops 全绿。

控制器正常终结，前后社区 10594 个跟踪文件及额外源码审计
通过，终结后再次核对全部 603 个候选文件摘要均一致。
原始全量/ops 日志、JUnit、184 项分类及源码审计归档为
`external-evidence:full267-complete-raw.json.gz`，
SHA256 `4fe1957e0a00d75d6be7943ee5983199d16b67e9842f30574a2cd016b75aa46a`。
随后独立完成当前 54 项 L1.5 五家族整轮验证：54/54 PASS，
115 个参数节点及 58 个 subtests 全部通过，无跳过或预期失败。
社区源码前后及全部 603 个候选文件摘要审计通过；该结果来自
独立矩阵报告，不由上述全量计数推导。完整证据与覆盖限制见
[当前完整矩阵结果](l15-family-matrix.md)。

### CT-FP8 官方 CPU dtype 归因（2026-11-06）

在同一 CPU 环境、隐藏全部 GPU 的独立进程，对此前绑定的真实
Qwen3-0.6B-Channel-FP8 完整 accuracy 请求，分别以官方
Transformers/compressed-tensors 加载 FP32 和 BF16 参考。
两者均独立 `from_pretrained`、`run_compressed=False`，不是将已加载
FP32 模型整体转 dtype。逐个核对 507 个原始 tensor，BF16 权重按
原始 FP8 channel 反量化后的加载 dtype 舍入；五次唯一前缀 forward
分别观察全部 196 个 Linear 的真实动态输入 QDQ。

FP32 的全部 12 行所选概率、greedy 指标及 top20 分布指标，与此前
原 V1 报告逐项完全相同。HCU 对 CPU FP32 的 top20 logprob 最大差
仍为 **2.673436**；对官方 CPU BF16 为 **2.5625**。
CPU BF16 对 CPU FP32，在 CPU BF16 自身的 top20 token 上最大差
为 **3.309192**。后一个指标使用不同的 top20 锚点，不能与前两者
直接相减来分摊误差。

该结果没有定位具体 HCU 算子错误，也没有为原 FP32 精度门提供豁免。
CPU BF16 包含 BF16 输出层、官方 eager attention/归一化/RoPE 存储边界，
并非 SGLang/HCU 的等价执行；所有参考均对固定 HCU 序列做完整前缀
teacher forcing，不证明独立 greedy 生成或增量 KV 行为。
生产默认 dtype、量化路径和原 V1 门槛保持不变。

首个诊断脚本因误将 CT 的可选 `quantization_enabled` 属性视为必有
而退出，保留该失败；按现有参考工具采用的缺省开启语义修正后，
在新目录完整重跑并实际观察 QDQ，未以缺省值代替执行证据。
原始报告、日志、实际执行脚本及原 V1 对照归档为
`external-evidence:ctdtype278-279-complete-raw.json.gz`，
SHA256 `ca96b73af4070b174bb4309dc04e4c459ec25beb36f3717861b06c303990a832`。
概率 tensor 留在仓库外的远端证据目录，归档保存其大小和 SHA256，
不宣称压缩归档内包含这些 tensor。

### 真实 BF16 Qwen3-30B-A3B MoE 验收（2026-11-06）

使用完整 48 层、128 专家的真实非量化 checkpoint，TP1、BF16、
静态显存预算 0.7，完成原完整 accuracy profile 的 eager/graph
加载、启动及生成。25 项 checkpoint 资产在控制器、两个 worker 和
独立参考间绑定一致；社区 10594 个跟踪文件及额外源码前后审计、
603 个候选文件摘要检查通过。没有截层、修改模型配置或改写社区文件。

独立 CPU FP32 Transformers `Qwen3MoeForCausalLM` eager 参考
完成全部 96 个位置，top1 为 **94/96**，greedy regret 最大
0.136753（仍在原 0.15 门槛内）。原 V1 **FAIL**：所选 logprob
最大误差 **0.386688**、top20 logprob 最大误差 **2.897985**、
top20+tail JSD **0.036908**，分别超过原 0.15、0.15、0.002。
tail mass 误差 0.018060、missed mass 0.005652 均在原门槛内。
模型执行成功与精度通过分别记录，不将控制器正常终结记作模型验收通过。

六对 eager/graph 请求的输出 token IDs 全部相同，但概率不一致。
按完整相同历史对齐 48 个预测位置，所选 logprob 最大差
**0.360927**，共同报告 top20 token 的最大差 **2.375**；
每个位置共同 top20 数量至少 17。六组请求均从首个生成位置就可见
概率差异，不能只归因于后续 decode，也不把未报告 token 当作零误差。

Cookbook 固定 `40adc4d30ec2640e3391cfd78759bd3598e95f71`，
参考 `docs/model-deployment/sglang/qwen3.md` 的 30B-A3B 单卡条目。
该条目表格标 BF16，示例命令实际指定 `float16`，保留这一差异。
本轮使用社区 0.5.22 Engine、BF16、Triton attention、page1、
mem0.7、context4096/max tokens16384 和 eager/graph；
示例为 SGLang 0.5.12、FA3、page64、mem0.9、关闭 Graph。
没有执行 HTTP、解析器、系统设置或多卡/EP 验证，不宣称原配方等价通过。

完整报告、原始日志、实际配方、六个受测工具文件和最终审计绑定归档为
`external-evidence:moe276-complete-raw.json.gz`，
SHA256 `91501fcd3130802066ca888bd52f8924222d21fee3ef181c86e38416c6b2d8cb`。
归档另提供真实文件摘要清单的 SHA256；控制器内
`memory274-validated-snapshot` 只是描述性标签，不是摘要。

### MoE prefill Graph 单变量对照（2026-11-06）

保持上述 BF16 TP1 模型与完整请求，仅通过社区公开参数
`disable_prefill_cuda_graph=True` 关闭 prefill Graph，保留 full
decode Graph。实际 server info 验证两阶段配置；启动与生成通过，
checkpoint 前后绑定、社区源码前后及 603 个候选文件审计通过。

与原 eager、原 graph 各对齐 48 个相同历史位置，输出 token 仍全部
相同，但概率没有对齐。对原 eager，所选最大差 **0.212463**、
共同 top20 最大差 **2.787537**；对原 graph，分别为 **0.440398**、
**3.875**。首个生成位置的差异也仍存在。该对照尚不能区分跨进程
变化和执行模式的贡献，不能据此确定根因或将关闭 prefill Graph
设为默认修复，也没有为该配置新增独立 FP32 精度通过结论。

原始 worker、完整模式对照、实际执行脚本及审计归档为
`external-evidence:moe280-complete-raw.json.gz`，
SHA256 `2ef5861277d6b0c2f7824f513f906370cd7d24d4bb21db09ad6ede7016fa92dd`。
本地从归档原始生成报告独立重算全部 12 项对照，与远端结果完全一致。

### MoE 相同 eager 配置的重复对照（2026-11-06）

回到原受测设备，使用相同 BF16 TP1、mem0.7、Graph 全关闭配置，
重新执行完整三批请求；再在同一 Engine 内通过公开 `flush_cache`
确认清空成功，重复相同三批请求。社区源码、checkpoint 前后及
603 个候选文件审计通过。

新进程首次 eager 与原 eager 的 48 个相同历史位置，输出 token
全部相同，但所选 logprob 最大差 **0.444063**、共同 top20 最大差
**2.640973**。同一 Engine 清空缓存后的重复执行也未逐位一致：
2048-token 用例在首个输出 token 分叉，其后七个不同历史位置全部
排除；其余同历史位置合计 **41** 个，所选最大差 **0.513023**、
共同 top20 最大差 **2.027433**。

因此，原 eager/graph 的差异不能单独作为 Graph 特有问题的证据。
此实验尚未记录实际调度批次、计算形状或逐层中间值，不能据此断言
内核非确定性、缓存错误或具体算子有误；也没有重新执行该次生成的
独立 FP32 精度参考，不继承原模型报告的精度指标。

原始首次/重复生成、缓存清空记录、完整历史对照、实际脚本及审计归档为
`external-evidence:moe282-complete-raw.json.gz`，
SHA256 `8322d61985d3d80f4320bb0d9c718760212b0abbbc16bb3530354727e0dcaf60`。
本地从原始报告重算全部 18 项对照，与远端结果完全一致。

### MoE 单请求串行逐层观察（2026-11-06）

在同一 Engine、BF16 TP1、mem0.7、Graph 全关闭条件下，对固定的
2048-token 请求串行执行未采样对照及三轮采样，每轮通过公开
`flush_cache` 确认清空成功。使用社区公开 forward hooks，
48 层各记录 8 次 MLP 调用，三轮共 **1152** 份实际样本，包含
计算前输入、gate、top-k 专家及权重、输出的最后一行和实际 batch metadata。
社区源码前后、checkpoint 前后及 603 个候选文件摘要检查通过。

两组相邻采样轮各有 6 个完整历史相同的预测位置；其余位置因输出
分叉排除。各自 288 个层/位置对照的实际 token、位置、序列长度、
形状及已采集的请求池索引、写缓存槽位完全一致。共同 top20 的
最大 logprob 差分别为 **1.874998**、**1.124989**。
首个生成位置最早观察到的 MLP 边界差异，均为第 1 层输入，
最后一行最大绝对差为 **0.0078125**；这不是首个错误算子的证明。
第 0 层其他 token 行尚未采集，不能由最后一行相同推断整层相同。

第二组对照另有一个 decode 位置，第 0 层 top-k 槽位顺序不同，
但按专家 ID 对齐后专家集合和对应权重完全一致，输入、gate 和
MLP 输出也逐位一致，不能将其判为路由语义变化。
未采样与首轮采样的 8 个预测历史相同，但概率不同，共同 top20
最大差 **2.396482**；CPU clone 带同步，不能声称观察器不影响执行。
本轮是定位实验，没有新增独立 CPU FP32 模型验收通过结论。

原始生成、全部 1152 份实际 tensor 样本、观察器及 CPU 预检、
分析与最终审计归档为
`external-evidence:moe285-complete-raw.json.gz`，
SHA256 `4cc832b77fec2f473893bada3eb959d306cc88ffcd1936f75498065d239101f9`。
本地核对全部归档文件摘要并从原始生成重算三项概率对照，与远端
结果完全一致；tensor 比较与 metadata/专家配对复核在隐藏 GPU
的远端 CPU 进程执行。早于终结审计的 metadata 报告保留其原始
pending 标记，归档同时包含实际终结状态及最终审计，不改写旧报告。

### MoE softmax 并列专家选择修复（2026-11-06）

继续使用相同真实 30B BF16 模型和 2048-token 请求，保存前两层
attention、MLP 及其嵌套投影、gate/top-k 的全部有效行和残差快照，
三轮共 96 份实际样本。两组相邻轮各有 8 个完整历史相同的位置，
已采集的 batch metadata 全部一致。第 0 层 attention 全部采样值、
MLP 输入和 gate logits 逐位相同；差异首先出现在 top-k，再进入
MLP 输出及下一层 attention。两组 prefill 的专家集合分别有
8 行和 6 行不同，全部是相同 BF16 logits 下的精确并列。
各轮选择均是数学上合法的 top8，所选权重相对独立 FP64 参考的
最大误差为 `4.09e-8`，但合法 top8 不代表满足社区的并列选择契约。

该完整行实验的源码、checkpoint 前后及 603 个候选文件审计通过。
原始输出、全部 96 份 tensor 样本、观察器 CPU 预检及逐行参考归档为
`external-evidence:moe288-complete-raw.json.gz`，
SHA256 `3786d4c3c3892540e72f3605ae93472172879d2ad76396bf942334633d0f546c`。
本地检查全部归档文件摘要并从原始生成重算三项概率对照，与远端一致。
CPU clone 的同步影响仍不能排除，不将该观察器实验代替模型精度验收。

对原形状真实 logits 的独立 HCU 重放，通过 Python 调用观察实查到
社区 `fused_topk` → 社区 `topk_softmax` → adapter 影子算子。
该进程的 `_is_cuda=False`、`_is_hip=True`、`_use_aiter=False`；
不能因模块源码存在 CUDA 分支就断言原请求使用社区 Triton router。
旧影子算子与实际 consumer 连续 12 次重放，专家集合均不稳定，
直接调用社区 Triton router 的 12 次则完全一致。

社区 power-of-two AOT CUDA softmax kernel 和 Triton router 都明确
要求分数降序、并列时选择较小专家 ID。影子 softmax 改用稳定降序排序
恢复这一规则，覆盖 destination-passing `topk_softmax` 和
`moe_fused_gate(scoring_func="softmax")`，不改变 sigmoid 路径、
gate dtype、生产 BF16 默认或原模型精度门槛。

新增独立标量排序/概率参考回归，覆盖 FP32/BF16、两个入口、
renormalize 开关、非连续输入以及变输入 Graph。修复前 CPU
**8 FAIL**、HCU **12 FAIL**，均无收集错误或跳过；修复后 CPU
算子与并列合同 **46 PASS**、HCU gate 文件 **22 PASS**。
原形状真实 logits 在三个入口各重放 12 次，修复后的实际 consumer
与影子算子的全部专家 ID 均与社区 Triton 完全一致，重复运行的
全部 ID/权重也逐位一致；对社区的权重最大差为 `5.9604645e-8`。
受影响的正式 MoE L1.5 子集 **10/10 PASS**，包含完整 pytest
相位及 subtest 检查；源码前后、603 个候选文件摘要检查通过，
L0 GREEN（31 条既有警告）。没有将这次子集计数当作修复后的
完整 54 项或全量 adapter 重跑。

修复前后原始测试 XML、真实算子重放、实际执行脚本、受测代码、
社区参考源码与最终审计归档为
`external-evidence:moe292-295-complete-raw.json.gz`，
SHA256 `0fa7f210bcd261a2cb321d483711c0479d8db5faabe1bfe282549f6954a5baea`。
本地独立核对全部文件摘要、XML 失败/通过计数及十项严格相位结果。
提交中的测试模块说明文字在采样后修正，全部测试函数体与受测文件
AST 一致；生产算子文件逐字节一致。真实模型的修复后完整原 V1
验收与同 Engine 重复验证分别留证，不据组件修复追认模型精度通过。

### 修复后真实 MoE 同 Engine 重复验证（2026-11-06）

对上述 softmax 修复的同一受测运行时代码，保持真实完整 30B
BF16 TP1、mem0.7、Graph 全关闭，执行完整 accuracy profile 的
三批六请求；在同一 Engine 通过公开 `flush_cache` 确认成功后，
原参数重复三批请求。未安装逐层观察器，未改变 gate logits 或权重。
本轮使用重新检查为空闲的设备，跨实验物理设备不相同；两次重复
均在该次同一 Engine、同一设备内完成。

六请求输出 IDs 全部相同，**48/48** 预测位置完整历史对齐，
所选 logprob 及共同 top20 最大差均为 **0**。另直接核对全部
输入/输出所选概率和完整 top20 列表，六组均逐位一致，不只检查
top1 或共同 token 的交集。社区源码前后、checkpoint 绑定及
603 个候选文件摘要检查通过。

该结果说明此前同 Engine 清缓存后的重复分叉，在本轮完整对照中
不再复现；不外推其他请求、配置或跨模式逐位一致，也没有为该轮
生成单独新增 CPU FP32 原 V1 通过结论。历史修复前输出的概率
对照另保留，不能把历史精度报告继承为新代码的模型验收结果。

原始首次/重复生成、完整历史对照、实际执行脚本及最终审计归档为
`external-evidence:moe297-complete-raw.json.gz`，
SHA256 `efdc117767d2996b1be6c1dd9493fbb89c61c39061c085d7b5a9755323402a1c`。
本地从归档重算全部 18 项对照，与远端一致，并直接核对六请求的
四类完整概率列表和输出 IDs；受测模型工具文件与本地逐字节一致。

### 真实 MLA 复合配置解析（2026-11-06）

现有完整 DeepSeek-VL2 checkpoint 的原始配置没有外层 architectures，
语言部分保存在 `language_config`。社区官方配置加载器可直接解析，
补出 `DeepseekVL2ForCausalLM`；官方文本配置为 BF16、30 层、
32 attention heads、72 routed experts、top6、词表 129280，启用 MLA。
这属于配置诊断，不是模型启动或精度通过。

验收工具此前只读取 `text_config`，会错误使用外层词表或拒绝合法的
语言 token，也会遗漏内层量化信息。现在支持 `language_config`，
与社区一致在两者同时存在时优先选择它；所选嵌套配置非法时明确拒绝，
不回退到外层元数据。带视觉配置的参考仍选择公开条件生成 AutoModel
入口，不把复合 checkpoint 当成普通 CausalLM，也不提取语言权重、
改写配置或放宽原精度策略。

回归修复前 **7 FAIL / 8 PASS**，无 error/skip；修复后连同模型
smoke、CPU 参考与 dtype 探针回归 **254 PASS**，无 error/skip。
源码前后和全部候选文件摘要检查通过，L0 GREEN（31 条既有警告）。
真实配置的词表与社区加载结果一致，未初始化 CUDA，原配置字节不变。
社区导入可注册该配置类，但当前 Transformers 条件生成模型映射
不包含该类；配置解析不代表独立参考支持该模型。真实 MLA 的启动、
生成与原 V1 精度验收仍未闭合。

修复前后测试 XML、受测代码、真实配置诊断、社区配置解析源码和
最终审计归档为 `external-evidence:checkpoint307-309-complete-raw.json.gz`，
SHA256 `58f427a0de8c355d2f3cd6c4f332c5e0123e6774527fd4e162923d5bc46c11da`。
本地独立检查全部 40 个文件的摘要、XML 计数和官方/工具词表对照，
受测生产工具与回归文件和提交逐字节一致。

### 修复后真实 MoE 原 V1 终结验收（2026-11-06）

保持真实完整 Qwen3-30B-A3B、BF16 TP1、mem0.7 和默认输出层，
修复后的完整 accuracy profile 在 Eager、Graph 均完成启动和生成。
六请求的输出 IDs、完整输入/输出所选概率及 top20 列表跨模式逐位一致；
48 个预测位置的完整历史全部对齐。该对照未安装逐层观察器。

独立 CPU FP32 Transformers eager 参考已终结，实际类为
`Qwen3MoeForCausalLM`。原 `bf16-cpu-fp32-top20-v1` 仍 **FAIL**：
96 个 teacher-forced 位置中 top1 匹配 **94/96**；
最大所选 logprob 误差 **0.326289475**（限 0.15）、
top20 logprob 误差 **3.653319359**（限 0.15）、
top20+tail 合并分布 JSD **0.041421528**（限 0.002）。
其余门内指标为 greedy regret **0.136753082**（限 0.15）、
未命中 top20 概率质量 **0.006318260**（限 0.02）、
tail 质量误差 **0.027444681**（限 0.03）。
所选误差和 JSD 最差请求均为 context-2048，top20 logprob
误差最差为 short-zh；两个模式的这些结果一致。

源码前后、25 个 checkpoint 资产绑定及 603 个候选文件摘要检查通过。
该轮沿用 softmax 修复快照，早于随后独立的 language_config 工具修复；
实际旧工具文件保存在证据内，不把新工具追认为本轮受测代码。
稳定选择消除了本轮已观察到的重复/跨模式分叉，但没有闭合原 V1
精度差距，不能据此将 Graph 或该路由修复认定为剩余精度差异的原因。
下一步在同一绑定生成序列上做独立 CPU FP32/BF16 诊断，
保持生产 dtype、输出层和原精度门槛。

完整生成、CPU 参考、配方差异、受测代码及终结审计归档为
`external-evidence:moe296-complete-raw.json.gz`，
SHA256 `37ac6843379a5defdf56e6de714ebe9b827a8a34e0fe61b3a84fcff83be9127f`。
本地直接核对六请求的四类完整概率列表和输出 IDs，重算所选误差、
逐行指标最大值及阈值判定，与报告一致；合并分布指标来自本轮
CPU 参考报告，归档没有保存可独立重算该分布的完整 CPU 概率张量。
早于终结审计的跨模式报告保留原 pending 标记，另以最终绑定关联
全部源码、权重与候选审计，不改写旧报告。

### softmax 修复后的全量 adapter 回归（2026-11-06）

以 `9b3c4eb` 的完整 603 文件快照重跑
`pytest tests/kernels tests/adapter tests/vendor -k "not stress"`，
结果为 **1692 passed / 47 failed / 137 errors / 79 skipped /
2 xfailed / 1 deselected**。逐个核对 XML，184 个失败/错误全部属于
暂缓处理的 vLLM：包括以 vLLM bridge 为宿主的 model mutation 测试；
**非 vLLM 失败/错误为 0**。跳过、预期失败和未执行项均不计通过，
本轮不是全量门禁全绿，也没有覆盖 stress。

单独执行 `run_triage.py --stage ops`：RoPE、RMSNorm、BF16 GEMM、
bridge Graph 四组通过；kernel 子组 336 passed / 2 skipped /
2 xfailed，vendor FA 53 skipped，AR gate 2 passed / 1 skipped /
1 deselected。没有失败组，但仍为 **INCOMPLETE**，退出码 2。
没有把不可用的 vendor 路径当成数值通过，也没有为减少跳过而解除守卫。
源码前后、全部候选文件摘要审计通过；CPU 契约/矩阵工具 70 PASS，
L0 GREEN（31 条既有警告）。

本轮 GPU 回归快照早于独立的 language_config 工具修复；与当前受测
运行时代码、MoE gate 测试及 ops 诊断入口逐字节一致。其后的差异为
该工具/对应回归及文档，工具改动另有上述 254 项 CPU 证据，
不把新测试计入这轮旧快照的全量计数。

完整测试 XML、失败清单、ops 日志、实际控制器、关键受测代码和终结
审计归档为 `external-evidence:full304-complete-raw.json.gz`，
SHA256 `8eb9a96a71967152b2e323a2789fec83ec74a7e15ac03a2c11f60e41072aebd3`。
本地核对全部 21 个归档文件摘要，独立重算 XML 各状态计数、
184 项宿主分类和三组未完整验证的 ops 结果。

### softmax 修复后的完整 L1.5 矩阵（2026-11-06）

对与上述全量回归相同的 `9b3c4eb` 完整候选快照，在独立目录和设备
执行五家族矩阵，**54/54 PASS**：Dense 10、MLA 17、MoE 10、
量化 10、Mamba 7。共 115 个参数节点、58 个 subtest；逐项核对
目标 nodeid、setup/call/teardown 三相及 subtest，全部真实通过，
无 collection error、skip、xfail 或缺少 call 相位。
没有把参数节点、子断言和逻辑场景累加成新的 case 数。

CPU 契约/矩阵工具 70 PASS，L0 GREEN（31 条既有警告）。
社区源码前后审计、603 个候选文件摘要检查通过。该结果闭合了
softmax 修复后的完整分层回归，仍不代表真实权重模型精度验收通过；
language_config 工具改动的快照边界与上述全量回归相同，另有 CPU 证据。

全部受测候选源码、54 项原始相位报告、矩阵/控制器和终结审计归档为
`external-evidence:matrix302-complete-raw.json.gz`，
SHA256 `526e2c3b56c160775aa9a9a67a6e5dc2001cd8bcff38ed0a65db80fdcd4ff764`。
本地核对全部 603 个受测源码文件摘要，并独立复核 54 个目标的唯一
覆盖、115 个完整相位节点和 58 个 subtest，以及前后相同的源码审计。

### 修复后真实 MoE CPU dtype 诊断（2026-11-06）

对上述修复后 30B 的同一绑定生成序列，在隐藏 GPU、HOSTnone 的新进程，
分别从完整 checkpoint 独立加载官方 CPU FP32、CPU BF16 模型；
两者均为 Transformers eager、2 threads，BF16 包含 BF16 输出层。
全部 12 行、96 个 teacher-forced 位置完成诊断，未改生产 dtype 或 V1。

HCU/CPU FP32 的四项 top20+tail 指标逐行与终结精度报告完全一致。
CPU BF16/CPU FP32 的 top20 最大 logprob 差 **3.132858276**、
合并分布 JSD **0.025420719**；HCU BF16/CPU BF16 则分别为
**3.25**、**0.020817186**，tail 质量误差最大 **0.031491076**。
CPU 自身 dtype 差异存在，但 CPU BF16 不是等价的 SGLang 基线，
HCU 与它之间也仍有差异，不能将剩余精度问题全部归因 dtype。
CPU dtype 对照以 CPU BF16 的 top20 为类别锚，另两组以 HCU top20
为锚，不能将各组最大值直接相减当作误差分解。

HCU/CPU FP32 最差 top20 项位于 short-zh、零起算输出位置 4、
HCU rank17、token399：三方 logprob 分别为 HCU **-13.375206947**、
CPU FP32 **-17.028526306**、CPU BF16 **-15.375051498**。
该项用于后续定位，不是具体算子错误的证明。两次 CPU 加载的旋转
频率与原始频率 buffer 均为 FP32，形状和字节摘要相同；
源码前后、完整 checkpoint 绑定及 603 个候选文件摘要检查通过。

诊断、原始绑定生成/精度报告、实际控制器和受测工具归档为
`external-evidence:moe311-complete-raw.json.gz`，
SHA256 `072971bb4013bf714366f160039072014708f17b578d8f48b385074ef0d05cde`。
本地检查全部 18 个文件摘要、各请求序列与原报告的对应、四项 FP32
指标逐行一致，以及 outlier 的实际标签和三方标量差。报告没有保存
完整 CPU 概率张量，不将标量复核称为独立重算整个合并分布。

### 真实 MLA 复合 checkpoint 文本启动终结（2026-11-06）

在未经文件改写的社区 SGLang 0.5.22 上，使用完整 DeepSeek-VL2
checkpoint 目录运行 BF16、TP1、triton attention、显存比例 0.7 的
accuracy profile；没有提取语言权重或改写配置。eager 与 decode full
graph 两种模式均完成六个请求，覆盖双语短输入、512/2048 token、
复用和变输入，启动及生成通过。实际 graph 模式的 prefill backend
为 disabled，不能据此声称 prefill graph 已验收。

两个模式的六组输出 token IDs 相同，48 个输出预测历史全部对齐；
但只有 short-zh 的四类完整概率字段逐位一致，其余五组存在差异。
同历史下所选 token logprob 最大差 **0.044533819**，
共同 top20 token 的 logprob 最大差 **0.468563080**。生成一致不能
替代独立精度参考，当前未建立该 checkpoint 的 CPU FP32 V1 结论，
也未执行图像请求或验收视觉计算路径。

配方参考固定为 `HYGON-AI/inference-cookbook-das`
commit `40adc4d30ec2640e3391cfd78759bd3598e95f71`。核对该版本
32 篇 SGLang 配方，未找到 DeepSeek-VL2 专用配方；
`deepseek-v3.md` 仅作 MLA/MoE 家族背景，其 W8A8、TP8/16
配置不能视为本次 BF16 TP1 的同配方验证。

控制器终结，社区源码前后与每个 worker 结束后的审计通过；
15 项 checkpoint 资产字节绑定一致，603 个受测候选文件摘要不变。
本次使用的受测快照早于最新上游 JIT 合入，不将结果外推为最新
提交的完整 GPU 回归。完整生成、逐请求模式比较、配方检索输入、
受测源码与终结审计归档为
`external-evidence:mla314-complete-raw.json.gz`，
SHA256 `1ee41b2d557217855a32d4fd313b5104a8f1a73e36f7361b749ce519c7f6cf8a`。
本地逐一核对 654 个归档文件的摘要、603 个源码文件清单，以及
六组输出与四类完整概率字段的实际对应关系。

### 非 softmax 与分组路由的并列选择契约（2026-11-06）

继续核对社区实现发现，sigmoid AOT/JIT CUDA kernel 同样明确以
较小专家 ID 打破并列；社区 Triton router 对 sigmoid、sqrtsoftplus
及分组选择也采用较小专家/组 ID 优先。原影子实现只有 softmax
采用稳定排序，其余路径及组选择仍用无序 `torch.topk`，不能保证
该契约。此处是独立组件缺陷，尚未证明它就是上述真实 MLA 概率差异的原因。

将所有 scoring 的专家选择与分组选择统一为稳定降序排序。
不改激活、bias-free 权重、共享槽位、归一化/缩放顺序、
默认模型 dtype 或 V1 门槛。新增独立 Python 标量激活/排序参考，
覆盖 72 个专家的非二次幂形状、FP32/BF16、strided 输入、bias、
grouped routing、共享专家与 renormalize，以及三次重复和变输入
graph replay。修复前隐藏 GPU 的回归为 **32 FAIL、8 PASS**；
修复后真实 HCU gate 文件 **94/94 PASS**，包括原 softmax 回归及
12 个变输入 graph 参数节点。额外双流回归 10 PASS、8 项社区源码
组合未在该进程启用；另在显式选择真实社区源码的干净 CPU 进程中，
双流与矩阵清单校验合计 **50 PASS**，无 skip/error。

前后源码审计、受测候选文件摘要和 L0 均通过。负向/正向日志、
逐节点 XML、完整正向受测代码及社区契约来源归档为
`external-evidence:gate331-332-complete-raw.json.gz`，
SHA256 `117af23c0766f2272e1f9306a1e6df106c5ce42308eef13e91afc757765cfb95`。
本地核对全部 629 个文件摘要，并独立验证负向 32 个失败均来自
新增并列选择断言，正向 94 个 gate 节点无 skip/failure/error。
社区 CUDA kernel 本轮仅核对源码，未在 NVIDIA 设备执行。

将 softmax/非 softmax 的并列和 graph 契约分别注册为四个 L1.5
exact nodeid，当前清单为 **58 case**（MoE 14）。新清单的完整
整轮运行另行验证，不把此前 54/54 结果改写成 58/58。

### 真实 MLA 的官方代码独立 CPU FP32 参考（2026-11-06）

对上述终结的 DeepSeek-VL2 原始 BF16 生成，补齐独立参考：
固定 `deepseek-ai/DeepSeek-VL2`
commit `ef9f91e2b6426536b83294c11742c27be66361b1`，
核对 11 个官方文件的 Git blob 与 SHA256，未改写参考模型源码。
在隔离的 Transformers 4.38.2 环境中，通过官方完整
`DeepseekVLV2ForCausalLM` wrapper 加载八个分片，保留视觉配置和
参数；缺失、多余、形状不匹配及加载错误列表均为空。
逐一核对全部 **6973 个 checkpoint tensor** 的原始值与 CPU FP32
加载值，执行真实 MLA eager attention。没有提取语言权重或绕开完整
checkpoint 加载；仅执行纯文本，未验收图像或视觉 forward。

本次沿用 CPU PyTorch 2.11，区别于官方依赖的 torch 2.0.1；
未安装仅视觉 forward 需要的 xformers。这些差异和完整版本记录
保存在原始证据中，不将本次环境称为官方已验证的依赖组合。
参考运行隐藏全部 GPU，未安装宿主桥；共享环境包不变，
参考源文件、社区源码与受测 adapter 文件前后摘要检查通过。

完整两种模式、12 行和 96 个 teacher-forced 位置的原 V1
结果为 **FAIL**：top1 **96/96**、greedy regret **0**，
所选 token logprob 最大误差 **0.079785109**，
top20 最大误差 **0.865669250**、合并分布 JSD **0.001434714**，
tail 质量误差 **0.005268764**、missed mass **0.000107828**。
仅 top20 最大误差超过原 **0.15** 门槛，其余五项在限内；
不能以 top1 全对或 JSD 通过豁免该失败。该生成快照早于非 softmax
并列修复，不代表修复后真实模型的最终结论。

保存五份去重后的完整参考概率张量，并在本地用独立标量实现，
对实际 HCU 所报类别重新计算全部六项指标；所选/top20 最大误差
重算差为零，JSD 与质量项差不超过 `3.34e-16`。
参考源码、加载审计、完整概率、原始 HCU 生成及终结审计归档为
`external-evidence:vl327-complete-raw.json.gz`，
SHA256 `bb29c78955f45fada7cab5c63a2d90024a47e08ab1c526cfe96f12412f5766f6`。
本地核对全部 40 个文件摘要及固定官方源文件的 Git blob。

### 真实 30B MoE 显式 FP32 配置诊断终结（2026-11-06）

保留完整 Qwen3-30B-A3B checkpoint、TP1、固定 accuracy 请求与
eager/graph 配置，仅显式传入社区 `dtype=float32`，显存比例设为
0.9 以容纳权重。两模式的六组生成及四类完整概率字段逐位一致，
48 个预测历史对齐。默认 BF16 与原 V1 阈值未改；该实验是显式配置
诊断，不是 BF16 验收或新的严格 FP32 数值策略。

独立 CPU FP32 参考的 12 行、96 个 teacher-forced 位置终结：
top1 **96/96**、regret **0**，所选 token 最大误差 **0.175325274**、
top20 最大误差 **1.724304199**、JSD **0.007823711**，
tail 质量误差 **0.002368215**、missed mass **0.000108224**。
沿用原 V1 作诊断时，前三项超限，结果仍为 FAIL；
所选/JSD 最差请求为 changed-2048，top20 最差为 short-zh。
不能认为仅改变公开 dtype 配置就已经消除剩余数值差异。

同卡 BF16、显存比例 0.9 的控制组仅 eager 完成；graph worker
在模型权重分配时发生 OOM，随后确认该组设备资源已被占用。
保留原始失败和日志，不将它当作完整的同卡两模式对照。
已完成 FP32 生成另以独立 CPU 任务对拍，原失败控制器未重启。
两次终结的社区源码及 603 个受测文件摘要检查均通过，
CPU 参考绑定与 HCU 执行的 25 项 checkpoint 资产一致。

进一步只读核对社区
`python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`
发现，该 runner 将 `compute_type` 选为 BF16（BF16 输入）或
FP16（其余输入）。尚未对本次真实模型的实际调用绑定此分支，
不能仅凭 server dtype 声称每个内部 GEMM 都使用 FP32；
后续须验证实际 runner、输入 dtype 和计算 dtype，再定位组件误差。

完整 FP32 生成、原失败 BF16 控制组、CPU 参考、受测工具、
上述社区 runner 与终结审计归档为
`external-evidence:moe317-334-complete-raw.json.gz`，
SHA256 `ee6fa84340770f586f15b854c7c5d211fa7a735832e438b1eef4de6fb7a3b30c`。
本地核对全部 39 个文件摘要、六组完整模式字段、所选 token 误差
及行最大值/阈值判定。该 CPU 参考未保存完整概率张量，
不将这些检查称为独立重算 JSD。

### 全 scoring 并列修复后的真实 MLA 生成复验（2026-11-06）

在包含全 scoring/分组稳定路由修复与最新 JIT 输入适配的快照上，
重新执行完整 DeepSeek-VL2 BF16、TP1、Triton attention、
显存比例 0.7 的 accuracy 请求。沿用上述固定 cookbook 版本及
配方差异说明，不改 checkpoint 或社区源码。Eager 和 decode graph
两模式均完成六组真实生成；server info 的 prefill graph backend
为 `None`，不据此声称执行了 prefill graph。
每个 worker 结束后及整轮前后的社区源码审计通过，
15 项 checkpoint 资产绑定一致，603 个受测文件摘要不变。

六组输出 token 全部一致，48 个预测历史对齐；四类完整返回概率
仅 short-en 逐位相同，其余五组仍有差异。模式间所选 token 的
最大 logprob 差为 **0.047582716**，共同报告类别的最大差为
**0.84375**。此处是模式一致性诊断，不能与独立参考误差混用，
也不能据此认为并列修复已解决真实 MLA 的精度问题。
本轮独立 CPU FP32 对拍另行执行，尚无最终 V1 验收结论。

完整生成、加载/模式配置、配方记录、受测源码及终结审计归档为
`external-evidence:mla344-complete-raw.json.gz`，
SHA256 `5e8cbe849bf9335023e02fb3b87451d8804b573316c42c4f821884aea0fae95e`。
本地核对全部 654 个文件摘要，并按相同预测历史重新计算模式差异。
独立参考接收原始生成报告的完整字节，SHA256 为
`ded504c53d850ed06a914056d56d7b7570532aa95f799d62d5a133d8c37dbd65`；
参考执行须重新核对实际 checkpoint 字节，不能改写报告路径或绑定
来代替资产一致性检查。

### 修复后真实 MLA 的独立精度复验终结（2026-11-06）

对上述修复后的原始生成报告，沿用固定官方 DeepSeek-VL2 源码、
完整 wrapper 与隔离 CPU FP32 环境执行独立参考。跨执行环境传递
原始报告字节，重新核对全部 15 项 checkpoint 资产；八分片加载
没有缺失、多余、形状不匹配或加载错误，全部 6973 个原始 tensor
与 CPU FP32 加载值逐一相等。参考模型源码、社区源码与 603 个
受测文件摘要前后检查通过。环境差异及纯文本验证边界沿用上述说明，
不外推视觉 forward。

两模式共 12 行、96 个 teacher-forced 位置的原 V1 仍为 **FAIL**：
top1 **96/96**、greedy regret **0**，所选 token 最大误差
**0.079785109**、top20 最大误差 **0.955446243**、
合并分布 JSD **0.002197070**、tail 质量误差 **0.008391375**、
missed mass **0.000107828**。top20 最大误差与 JSD 分别超过原
**0.15** 和 **0.002** 门槛；不能以路由组件回归通过、生成成功
或 top1 全对替代该精度结论。

保存五份去重后的完整参考概率张量，本地独立标量实现重新计算
全部六项指标：所选/top20 最大误差重算差为零，JSD 和质量项
差不超过 `2.23e-16`。完整参考概率、加载审计、原始生成、固定
官方参考源码与终结审计归档为
`external-evidence:vl356-complete-raw.json.gz`，
SHA256 `20f6bd225659d498aeb892c2ded6c4d478db2017ea140c84c66f66f6b9a44673`；
本地核对全部 40 个归档文件摘要及官方源码 Git blob。

### 真实 MoE 的 FP32 存储与内部计算路径观察（2026-11-06）

在完整 Qwen3-30B-A3B、显式 FP32、TP1、Triton attention、
显存比例 0.9、eager 的短请求中，通过社区公开 forward hooks
观察首层 MLP，并记录原社区函数的调用元数据。先未启用观察器
预热，再分别执行未观察和观察的两 token 生成；输出 ID 与四类
完整返回概率字段逐位一致。观察器不改输入、权重或计算函数。

实际捕获 prefill `[24, 2048]` 和 decode `[1, 2048]` 两次
MLP 调用，输入和输出均为 FP32 存储。两次均经原社区
`_fused_moe_kernel_sequence`，四次实际
`invoke_fused_moe_kernel` 的 `compute_type` 均为 **FP16**，
而 hidden states、w1、w2 的存储类型为 FP32。这与上述社区
runner 的分支一致，证明公开 `dtype=float32` 不能代表该路径的
每次内部 GEMM 都用 FP32。该观察只覆盖本次首层短请求，不是
对历史 accuracy 请求或全部层的追溯采样；也尚未独立验证实际
GEMM 输出的舍入误差，不能据此认定 adapter 错误或解释全部
BF16 分布偏差。默认 dtype 与原 CPU FP32 V1 门槛保持不变。

25 项 checkpoint 资产、社区源码前后审计与 603 个受测文件摘要
均通过。完整调用元数据、控制生成、源码与权重绑定归档为
`external-evidence:moe364-complete-raw.json.gz`，
SHA256 `0a316febfcdb3a2599b090a23adfef98b63ccc309ee2d6e3c3055e22d2635d1e`；
本地核对全部 626 个文件摘要与控制组完整概率字段。本项仅为
计算路径归因证据，模型精度验收仍未通过。

### 真实 MoE decode 操作数与计算舍入的分段核对（2026-11-06）

沿用上述完整 checkpoint 和显式 FP32 短请求，在首层一个 decode
MLP 边界采集写回前输入、两段 GEMM 的实际输入、八个选中专家的
权重切片、路由及返回输出。未采样 prefill；CPU clone 含同步，
因此另核对未观察控制组的输出 ID 与四类完整概率，均逐位一致。
25 项 checkpoint 资产、社区源码前后审计与 603 个受测文件
摘要均通过。采集器的隐藏 GPU 预检覆盖未启用/多行不采样、
外部 profiler 拒绝、选中专家对应、输入/权重不变及返回后保存。

实际 up/down GEMM 分别为 `[1,2048]` 与 `[8,768]` 输入，
完整权重形状为 `[128,1536,2048]` 与 `[128,2048,768]`；
保留路由相乘在 down-GEMM 输出舍入之前的顺序。
社区非量化分支直接以实际存储 dtype 加载 A/B，
`compute_type=FP16` 控制的是累加器的输出转换，不能把这一
参数解读成输入先转 FP16。独立 CPU FP64 乘积与实际输出的
最大绝对差分别为 **0.0017482405**、**0.0000874466**；
仅补 FP16 输出舍入仍有 **0.001953125**、**0.0001220703**
差异。两段实际输出均可无损往返 FP16。

保持原矩阵形状、stride、配置和全部路由元数据，将选中专家
切片逐字节搬回设备重放原社区函数；未选中专家置零，因此这是
选中操作数的原形状重放，不是完整 checkpoint 再运行。
两段各重放两次，12288/16384 个输出元素分别与真实采样
逐位一致。直接取得该重放进程实际使用的两个编译内核：
TTIR 的 dot 为 FP32 输入、`inputPrecision=tf32`，
LLVM IR 含 TF32 MMAC，输出转换为 FP16；并非 FP16 乘法。
这证明了该重放路径的 TF32 乘法、FP32 累加、FP16 输出舍入，
不追溯为所有历史请求的运行时内核绑定。

对实际输入模拟 TF32 向零截断再做 FP16 输出舍入，分别匹配
**12271/12288**、**16376/16384** 个元素，剩余最大差为
**0.0002441406**、**0.0000038147**。该参考没有模拟硬件累加
顺序，不能将其称为完全等价的指令模拟。以不导入 Torch 的独立
标量实现重算全部两段 GEMM：上述三种参考的匹配计数相同，
最大差/RMS 与 CPU 张量计算的差不超过 `1e-12`。
从实际 up 输出检查 SiLU×up 的最大差为 **5.96e-8**；
实际 down 输出的专家求和与 MLP 返回全部 2048 个元素相同。
这些局部结果不证明全部层或默认 BF16 的模型精度通过。

采样、完整操作数、两种 CPU 诊断参考与审计归档为
`external-evidence:moe379-complete-raw.json.gz`，
SHA256 `04d9079fa679720adf6c6cff78401e03f93e11ab605dd26d1b0e647102ce3ace`；
本地核对全部 657 个文件摘要。原形状重放、实际编译产物和
独立源码审计另存于 `external-evidence:moe394-complete-raw.json.gz`，
SHA256 `c5404ec3a2e5fc39cd7042872234bfcc6bd07d79f1a6a70477f7116d009b35ec`；
本地核对全部 628 个文件摘要及实际编译产物的 IR。
默认 dtype、社区函数和原 CPU FP32 V1 门槛不变。

### 默认 BF16 MoE 三层实际操作数对拍（2026-11-06）

在完整 Qwen3-30B-A3B、默认 BF16、TP1、Triton attention、
显存比例 0.7、eager 配置中，对短英文请求执行未观察控制组与
观察组的两 token 生成。只采第 0、24、47 层各一个 decode
MLP 的实际输入、选中专家切片、路由、两段 GEMM 返回值与
MLP 输出。两组输出 ID 和四类完整概率字段逐位一致；
六次实际 GEMM 的 `compute_type` 均为 BF16。
所采六次调用未使用显式 FP32 实验的 FP16 输出转换路径。

隐藏 GPU 的独立 CPU 进程，对全部六段 GEMM 以实际 A/B
计算 FP64 乘积，保留 down-GEMM 路由相乘，并分别记录原始
乘积误差及 FP32→BF16 输出转换后的误差。转换后共
**86006/86016** 个元素逐位相同；六段不匹配数依次为
**2、2、1、4、1、0**，最大绝对差不超过 **7.6293945e-6**。
独立标量实现不导入 Torch，以原始 BF16 存储重算全部乘积及
转换，得到相同计数，最大差/RMS 与 CPU 张量实现的差均小于
`1e-12`。十处残差中九处相差一个 BF16 表示步长，另一处位于
接近零的结果、相差三个步长；没有模拟硬件累加顺序，不能将
这些残差声称为已经完成指令级归因。

分别以实际 up-GEMM 输出计算 FP64 SiLU×up 后转 BF16，
三层共 18432 个激活元素逐位一致；以实际 down 输出做 FP32
专家求和再转 BF16，共 6144 个 MLP 输出元素逐位一致。
这些检查只证明所采短请求、层和 decode 步的局部数学关系，
不证明全部层、prefill、graph、中文/长输入或整模型 V1 精度通过。
下一步仍须在原失败请求的相同预测历史上核对路由及计算链。

25 项 checkpoint 资产、社区源码前后审计与 603 个受测文件
摘要均通过。完整采样、实际操作数、CPU 参考与终结审计归档为
`external-evidence:moe399-complete-raw.json.gz`，
SHA256 `b03b5d2d746b7d7ab5f4cf3ca857fb3192b8d969d94b5e79709f0fd74902e2ab`；
本地核对全部 642 个文件摘要及控制组完整概率。
默认 dtype 和原 CPU FP32 V1 门槛保持不变，整模型验收仍未通过。

### 中文相同预测历史的 BF16 路由与 MoE 核对（2026-11-06）

沿用完整 Qwen3-30B-A3B、默认 BF16、TP1、Triton attention、
显存比例 0.7 和 eager 配置，对短中文请求生成八个 token，
采集第 0、24、47 层在零基输出位置 4 的 decode。
观察组与未观察控制组的输出 ID、四类完整概率逐位一致；
六段 GEMM 均实际报告 `compute_type=bf16`。
社区 10594 个跟踪文件的前后完整性、603 个受测文件摘要及
25 项 checkpoint 资产审计均通过。

额外保存实际 gate 权重、gate 输入、路由 logits、选中专家和权重。
独立 CPU FP64 gate 乘积经 FP32→BF16 转换后，三层共
384 个 logits 全部匹配。对实际 logits 以分数降序、较小专家
ID 优先独立排序，全部 24 个选中专家 ID 匹配；实际路由输出
与两段 GEMM 的路由输入逐位一致。标量 softmax 及选中权重
归一化参考与实际权重的最大绝对差为 **1.9355240e-8**。
这里没有 bias 或 softcap，实际启用 renormalize。

对全部六段实际 GEMM，以 FP64 乘积及 FP32→BF16 输出
转换核对，共 **86006/86016** 个元素逐位一致；六段不匹配
数为 **4、2、1、2、0、1**，最大绝对差 **0.001953125**。
三层 SiLU×up 的 18432 个元素、FP32 专家求和后转 BF16
的 6144 个输出元素全部匹配。无 Torch 的独立标量实现从
归档实际存储重算全部六段乘积、384 个 gate logits 和路由：
GEMM 匹配计数相同，最大差/RMS 与 CPU 张量参考之差小于
`1e-12`，gate 舍入及选中专家也全部匹配。十处 GEMM 残差
中九处相差一个 BF16 表示步长，一处接近零、相差三个步长；
尚未完成硬件累加顺序的归因。

与原始中文失败运行比较，prompt 和 checkpoint 字节相同，
全部八个生成 token 相同，因此目标位置的预测历史相同。
但原始运行的 batch size 为 2，中文位于第二行，本轮为单行；
四类完整概率字段均与历史运行不同。不能将本轮声称为原执行
形状或原数值失败的完整复现，也不能据局部匹配豁免 V1。
下一步需恢复原两行批次并核对逐行路由和计算链。

完整操作数、控制组、历史比较、CPU 参考及审计归档为
`external-evidence:moe409-complete-raw.json.gz`，
SHA256 `f9f5c06a5ce7c815c2e0450e7e5a764d3c55d2bcd3d205889df453498ae6a3aa`；
本地核对全部 646 个文件摘要。独立标量结果另存为
`external-evidence:moe417-independent-summary.json`。
默认 dtype、社区源码和原 CPU FP32 V1 门槛保持不变，
整模型验收仍未通过。

### 恢复双语两行批次后的目标 decode 核对（2026-11-06）

恢复历史短请求的两行批次顺序：英文、中文，各生成八个 token。
默认 BF16、TP1、Triton attention、显存比例 0.7 和 eager 保持
不变，采第 0、24、47 层、零基输出位置 4 的实际操作数。
同时记录 forward batch 的输入 token、序列长度及请求槽位：
三层均为输入 `[151643,198]`、长度 `[28,27]`，与两条请求的
第 4 个输出 token 及 prompt 长度逐行对应。两行中相同专家
只保存一份权重，按实际专家 ID 逐条恢复全部 16 条路由。
观察组与控制组的两条输出 ID 和四类完整概率均逐位一致。

隐藏 GPU 的独立 CPU FP64 参考核对全部六段 GEMM，
保留 down-GEMM 的路由相乘及 FP32→BF16 输出转换：
**172007/172032** 个元素逐位相同，六段不匹配数依次为
**3、7、4、1、4、6**，最大绝对差 **0.00390625**。
实际输入与权重计算的 gate logits 经舍入后 **768/768** 匹配；
逐行独立排序的 **48/48** 个选中专家 ID 匹配，路由权重最大
绝对差 **1.9355240e-8**。三层两行的 SiLU×up **36864**
个元素及专家求和后转 BF16 的 **12288** 个输出全部匹配。
独立无 Torch 标量实现从归档重算全部 GEMM、gate 和路由，
匹配计数相同，GEMM 最大差/RMS 与张量参考之差小于 `1e-12`。
25 处 GEMM 残差中 22 处相差一个 BF16 表示步长，另三处
分别相差 2、4、7 个步长；没有完成硬件累加顺序归因。

与历史生成 JSON 逐项比较，两条请求的全部输出 ID、目标位置
预测历史及输入阶段两类完整概率相同；decode 阶段两类完整概率
仍不同。两条请求的所选 token logprob 最大差分别为
**0.032894731**、**0.004318263**，共同 top20 候选最大差为
**1.374848366**、**1.6875**，候选集合也有变化。因此恢复批次
形状仍未逐位复现历史执行，不能把变化全部归因批次大小。
原异常中文 token399 在位置 4 的本轮 logprob 为
**-13.312736511**；使用相同 checkpoint、prompt 和全部生成
历史绑定的既有 CPU FP32 参考 **-17.028526306**，该候选误差
仍为 **3.715789795**，超过原 0.15 门槛。这是单个既有参考
候选的核对，不是完整 V1 重新运行。

本轮与上述单行中文采样的实际 MLP 输入比较，第 0 层最大差
**0.001159668**，第 24/47 层增至 **0.232421875/0.515625**；
第 0 层路由 logits 和专家 ID 全部相同，第 47 层则有一个
选中专家变化（两行批次为 7、单行为 122）。这描述了所采
执行之间的数值差异，不证明具体算子错误或整模型失败归因。
后续需沿层边界对齐独立 CPU FP32 参考，定位差异的传播。

源码前后 10594 个跟踪文件、603 个受测文件摘要和 25 项
checkpoint 资产审计均通过。完整采样、CPU 参考、行绑定、
历史候选参考与比较归档为
`external-evidence:moe422-complete-raw.json.gz`，
SHA256 `2ff3b3c2ee591c2270097615994d6ad73a97f050a68678520278603bb46155aa`；
本地核对全部 653 个文件摘要。独立标量和批次组件比较另存为
`external-evidence:moe425-independent-summary.json`、
`external-evidence:moe433-batch-component-comparison.json`。
默认 dtype、社区源码及原 CPU FP32 V1 门槛不变。

### 官方 CPU FP32 层边界参考与 gate 局部变量核对（2026-11-06）

在隐藏所有 GPU、禁用 adapter host 的独立进程中，以官方
Transformers 5.12.1、PyTorch 2.11.0、CPU FP32、eager attention、
八线程加载同一完整 checkpoint。两条短请求分别输入完整
prompt 加八个已生成 token，关闭 KV cache，采集零基输出位置 4
对应的全部 48 层 MLP 输入/输出、gate logits/权重/专家 ID、
attention 输入/输出及层输出。checkpoint 前后字节匹配，
实际官方模型源码另存并核对摘要。CPU 参考是完整序列计算，
HCU 是缓存 decode，不能把它们称为相同执行实现。

与上述双行 HCU 的三层实际采样对齐，英文的三层专家 ID 全部
相同；中文第 0、24 层全部相同，第 47 层仅重合 7/8 个专家。
中文 MLP 输入最大差依次为 **0.004982293、0.075684631、
0.668693423**，MLP 输出最大差为 **0.000282519、0.009505281、
0.849285126**。本次独立 CPU 计算的 token399 logprob 为
**-17.028533936**，与既有参考相差约 `7.63e-6`，未声称逐位
重复历史 CPU 执行。只对输出位置 4 的完整 CPU 概率及实际 HCU
top20 核对，英文/中文最大 top20 logprob 误差分别为
**0.606075287/3.971648216**，仍超过 0.15；这不替代六请求、
两模式完整 V1。

对中文三层 gate 单独改变输入和输出转换，保持实际 gate 权重，
以 FP64 乘积分别检查 CPU 实际输入、CPU 输入转 BF16、HCU
实际输入，并分别保留原始乘积或补 FP32→BF16 输出转换。
第 47 层的 CPU 输入即使补上述两次 BF16 转换，仍选专家 122；
HCU 实际输入的未舍入乘积已经将专家 7 排在 122 前，差约
**0.004548693**。补输出转换后两者 logits 均为 **-3.953125**，
较小 ID 规则继续选 7，与实际路由一致。因此这个采样点的
专家差异不能仅归因最终 gate 输出舍入；输入隐藏状态的差异
已经改变了两者顺序。尚未定位此前哪个算子或层产生主导差异，
也未将这一局部检查宣称为整模型误差归因。

原始 48 层 CPU 张量、官方源码、三层 HCU 输入、比较脚本与
gate 变量检查归档为 `external-evidence:moe436-complete-raw.json.gz`，
SHA256 `42dd1931bded9f819511c03377220bdcf1bc1f16501f32a3c3b8ae611f9a459e`；
本地核对全部 24 个文件摘要。无 Torch 标量实现另从实际存储
重算三层八种 gate 检查及两条请求全部三层边界 max/RMS，
结果差小于 `1e-12`，保存为
`external-evidence:moe442-independent-summary.json`。
下一步需补齐 HCU 全部层边界，沿相同预测历史定位差异传播。

### HCU 全部 48 层边界与首次路由集合差异（2026-11-06）

保持上述双语两行批次、默认 BF16、TP1、Triton attention、
显存比例 0.7 和 eager 配置，采集输出位置 4 的全部 48 层
attention 与 MLP 输入/输出，共 96 次模块调用，并保存每层
实际 gate 权重、路由 logits、权重及专家 ID。每次调用绑定
输入 token、序列长度、请求槽位；attention 另核对 positions。
只采 decode，不将延后处理残差的 decoder 返回值当成 CPU
完整层输出。采样器的隐藏 GPU 预检覆盖两种模块、非 decode/
错误批次跳过、外部 profiler 保留和目标步采样。

观察与控制两行的输出 ID、四类完整概率逐位一致；本轮与前述
三层采样运行的控制组完整概率也全部相同。96 次实际 GEMM
均报告 BF16，路由输入与捕获的路由输出逐位一致。以实际输入
和 gate 权重独立计算 FP64 乘积后转 FP32→BF16，全部
**12288/12288** 个 logits 匹配；逐行独立稳定排序的全部
**768/768** 个选中专家 ID 匹配。

与已保存的官方 CPU FP32 层边界对齐，在本次输出位置上，
英文首次专家集合不同是第 **27** 层，集合不同的层为
**27、35、43**；中文首次为第 **30** 层，集合不同的层为
**30、31、33、39、44、47**。这些是当前位置的首次集合
差异，不代表此前没有隐藏状态偏差或整个请求的首次差异。
首层 attention 输入恰与 CPU 归一化输出转 BF16 后全部
**4096/4096** 个元素匹配；更后面的 attention/MLP 边界
已有偏差。英文第 26→27 层 MLP 输出相对 RMS 差从约
**0.031876** 增至 **0.302064**，中文第 29→30 层从
**0.067535** 增至 **0.247911**；这是伴随路由变化的观测，
不能单凭相邻层数值证明因果。

对首次不同层的 gate 作独立局部变量检查：保持 CPU 实际
MLP 输入及实际 gate 权重，仅补 gate 的 BF16 输出转换，
英文边界的专家 51/41、中文的 71/38 分别成为并列，较小 ID
规则选中 41、38，选中集合均与实际 HCU 相同。中文将 CPU
输入先转 BF16 后、不补 gate 输出转换时也会选 38。实际
HCU 输入的未舍入乘积已经具有不同排序，因此这些检查表明
局部转换能重现相同集合，不将它解释成唯一误差来源或整模型
失败的完整归因。此结果与前述末层的 gate 检查范围不同。

目标位置的英文/中文 top20 最大误差仍为
**0.606075287/3.971648216**，没有通过原 0.15 门槛。
源码前后完整性、603 个受测文件及 25 项 checkpoint 资产
审计通过。完整 HCU/CPU 张量、官方源码、采样/比较脚本和
审计归档为 `external-evidence:moe447-complete-raw.json.gz`，
SHA256 `e9ac3c18a1e41f9b5f35a33676db0a88d73f6d18e2a21b540edde666b3961b62`；
本地核对全部 976 个文件摘要。独立无 Torch 标量实现重算
全部 48 层两请求的边界 max/RMS、gate 乘积/排序/权重、
首次不同层的八种 gate 检查和首层输入舍入关系，结果与张量
参考之差小于 `1e-12`，保存为
`external-evidence:moe451-independent-summary.json`。
下一步应分解早期 attention 的实际投影、Q/K 归一化、RoPE
和缓存读写计算，不能以当前 gate 局部匹配替代整链验收。

### 首层实际 attention 操作数与指数权重转换（2026-11-06）

沿用上述双语两行批次、默认 BF16 和输出位置 4，只观察第 0 层
attention 一次 decode。保存实际输入、投影权重及 QKV 投影输出，
在原地 Q/K 归一化和 RoPE 前分别采样，另保存实际 backend 使用的
Q/K/V、物理 KV 索引、被引用的历史缓存、context 和输出投影。
按 token、positions、序列长度绑定两行；当前 K/V 与各请求历史缓存
尾行逐位相同，投影输入/context/模块输出链逐位相同。
本次 QK 归一化与 RoPE 未融合。观察/控制的输出 ID 和四类完整
概率逐位一致，且与前述全层控制一致。源码前后完整性、
603 个受测文件和 25 项 checkpoint 资产审计通过。

以实际操作数计算独立 CPU FP64 参考，再作 FP32→BF16 转换：
QKV 投影 **10239/10240** 个元素匹配，最大绝对差
**1.9073486328125e-6**；Q/K 归一化分别 **8192/8192**、
**1024/1024**，Q/K RoPE 分别 **8192/8192**、**1024/1024**，
输出投影 **4096/4096** 全部匹配。V 与原始投影对应部分也逐位相同。
按实际历史缓存长度 **28/27** 作理想 softmax attention，
context 转换后仅 **6296/8192** 个元素匹配，最大绝对差
**0.000244140625**。无 Torch 标量实现从原始存储独立重算全部
16 项 FP64/转换后 max、RMS 与匹配数，与张量参考之差小于
`1e-12`；额外 FP32 算术变体单独保留，不混称为此 16 项复核。

进一步检查固定社区 commit 的
`python/sglang/kernels/ops/attention/decode_attention.py`：
grouped stage1 的普通分支先计算 `exp(score - max)`，在 P·V
前将该指数权重转为 V 的 dtype，分母仍使用未转换的 FP32 权重
求和。源码还保留 gfx1250 的不同分支，不能将普通分支泛化为所有
硬件行为。实际 split 元数据为 **7/7**、上限 **8**；
`MIN_BLOCK_KV=32` 使这两个短请求各只有一个非空 split，
不涉及多个非空 split 的合并误差。

用实际 Q 和历史 K/V 构造对应的数学变体：FP64 QK 乘积转 FP32、
FP32 scale/指数运算，指数权重转 BF16，分母使用未转换权重，
P·V 与求和以 FP64 计算，最后转 FP32→BF16，context
**8192/8192** 逐位匹配。远端隐藏 GPU 的 CPU Torch 与独立
无 Torch 标量实现均得到该结果；同样 score 下不转换指数权重时，
仅 **6295/8192** 匹配。仅保持 FP64 score 再转指数权重时
**8189/8192** 匹配；将归一化后的概率转 BF16 是另一种数学变体，
不能替代上述内核转换位置。

这将本次首层 context 对理想参考的残差收敛到与社区计算顺序一致
的局部解释，没有据此修改内核或放宽验收。它使用实际历史缓存，
尚未独立验证此前每步 K/V 的生成，也未模拟 MFMA 累加顺序、
硬件指数近似或多 split 归约；不能由输出匹配反推每条设备指令。
原整模型 V1 失败仍保留，下一步应对齐历史缓存的生成，并沿后续
attention/MLP 边界检查误差传播。

实际张量、观察器、CPU 参考、受测代码与审计归档为
`external-evidence:moe462-complete-raw.json.gz`，
SHA256 `20114160b250b0492c94542ee7e62ff781e855c78064bbe4476afd089c9d8ceb`，
本地核对全部 636 个文件摘要。独立标量复核为
`external-evidence:moe465-independent-summary.json`。
社区内核原始字节、两种实现的变体脚本与结果另存
`external-evidence:moe473-context-reference-supplement.json.gz`，
SHA256 `5e38259b9988798bd2dbca9381a4a908819a641f4508abaa9d5e66108ba49d08`，
核对全部 11 个文件摘要。该内核字节与上述社区 commit 一致，
SHA256 `7ccbaaf523930e2f46b512e5218ed00e20de366703947b9e1ff42a22652d70e7`。

### 首层历史 KV 的独立生成与 RoPE 缓存精度（2026-11-06）

在隐藏 GPU 的独立 CPU 进程中，从同一 checkpoint 重新读取首层
embedding 行、输入归一化与 attention 权重，按两条请求完整
prompt 加八个已生成 token 运行官方 FP32 首层计算。当前目标位置的
输入与 attention 输出均与前述官方完整 48 层模型参考逐位相同；
因此新提取的前 **28/27** 个 token 的 K/V 与该 CPU 参考有明确
联系。checkpoint 全部 25 项资产前后字节核对通过，官方模型源码
字节与已归档版本相同；加载的投影与 Q/K 归一化权重也逐位等于
实际 HCU 观察中的权重。

实际历史缓存与 CPU FP32 历史直接转 BF16 的结果并不相同：
英文 K/V 分别 **7627/14336、7599/14336** 个元素匹配，
中文分别 **7361/13824、7455/13824**。这种比较遗漏中间计算的
转换，不能由此判定缓存写入错误。另从实际 checkpoint embedding
和权重独立计算输入归一化、QKV 投影、Q/K 归一化及 RoPE，在各
对应输出边界作 FP32→BF16 转换。目标位置的输入全部匹配，
QKV 输出英文 **5119/5120**、中文 **5120/5120** 匹配。

首次数学变体把 RoPE 的 cos/sin 也转为 BF16，历史 K 仅匹配
英文 **12527/14336**、中文 **12054/13824**，目标 Q 也有明显
残差。实查已保存的实际 cos/sin 缓存是 **FP32**。改用官方 CPU
FP32 cos/sin 后，历史 K 匹配英文 **14334/14336**、
中文 **13824/13824**；英文剩余两个元素最大绝对差
**0.00390625**，位置为 `(4,1,84)`、`(7,3,44)`，
三元组分别是零基 token、KV head、head 内坐标。
目标 Q 分别匹配 **4095/4096、4096/4096**。
CPU 与实际目标位置 cos/sin 本身仍有最大约
**9.54e-7/8.34e-7** 的差异，不把 CPU 三角函数当成实际缓存的
逐位复制，也未据此唯一归因剩余 K 差异。

独立分阶段参考的历史 V 分别匹配英文 **14332/14336**、
中文 **13820/13824**，各保留四个差异及其坐标；
最大绝对差分别为 **0.0001220703125、3.814697265625e-6**。
目前没有采集此前每步实际输入归一化和投影输出，尚不能区分这些
差异来自历史 GEMM、归一化舍入还是其他执行细节。这里验证的是
历史缓存内容与独立生成的接近程度，不是对每次历史写入过程的
逐条证明，更不替代后续层或整模型原 V1。

原始张量、实际 checkpoint 权重、官方源码、脚本及前后审计结果
归档为 `external-evidence:moe476-complete-raw.json.gz`，
SHA256 `00c27f43832d4f2ade9cdb64a37238b0714cfe514ab16163f84ab775147a07ac`，
核对全部 7 个文件摘要。无 Torch 标量实现从 checkpoint embedding
与权重独立重算全部历史输入归一化、K/V 投影、K 归一化及两种
精度的 RoPE，并复核 CPU 参考比较指标、FP32-trig K 差异坐标；
22 项基础指标与 4 项 FP32-trig 指标的 max/RMS 差小于 `1e-12`。
标量实现复用已保存的 CPU cos/sin，不另宣称重算其生成算法。
脚本、独立结果和 FP32-trig 张量保存于
`external-evidence:moe485-history-reference-supplement.json.gz`，
SHA256 `4eddd045a003836059f4ced587a7058f62a90e3ffb75f3939907a6000a8ccf0a`，
核对全部 11 个文件摘要，并绑定此前实际 HCU 与官方完整 CPU
参考归档摘要。下一步需观察历史实际投影边界，定位剩余元素，
再沿后续 attention/MLP 检查误差传播。


### 实际历史投影边界与 decode KV 写入核对（2026-11-06）

在同一双语批次、默认 BF16、TP1、Triton attention 和 `disable_cuda_graph` 条件下，重新启动真实权重模型，观察第 0 层从 prefill 到目标位置的 **1 次 prefill + 4 次 decode**。观察组和控制组的输出 ID 及四类完整概率字段逐位一致；社区 pristine 源码前后审计、候选文件摘要和 checkpoint 25 项资产审计全部通过，GPU 任务终态正常且未留下自有进程。

保存每一步实际输入、QKV 投影、Q/K 归一化、RoPE 后 Q/K/V 及实际 RoPE 缓存。prefill 记录 47 行，四个 decode 各记录两行；token、position、请求顺序与控制输出逐行绑定。实际 decode 的 normalized K 在四步均 **1024/1024** 匹配独立 FP32-RoPE 历史参考，RoPE 后 K 也均 **1024/1024** 匹配。实际 V 在四步分别为 **1024/1024、1023/1024、1024/1024、1024/1024**；差异为单个极小舍入残差。Q 的差异保留在少量元素，最大绝对差不超过 **0.015625**；RoPE 缓存自身与 CPU FP32 三角函数的差异约为 `1e-6`，不能把 CPU 三角函数当作硬件逐指令复制。

目标 decode 的物理 KV 索引和历史缓存尾行由观察器直接核验。以独立 FP32-RoPE、分阶段 BF16 的历史参考对齐实际缓存：英文 K **14334/14336**、V **14332/14336**，中文 K **13824/13824**、V **13820/13824**。这与此前只对最终缓存做的比较一致；实际边界进一步显示，当前 decode 的 K/V 生成和目标 KV 索引读写没有出现批次行错位或尾行写错。剩余 2 个英文 K 和 8 个 V 元素的差异仍需获取历史每一步的实际投影输入才能精确归因，不能据此修改生产代码。

真实 GPU 操作数、控制器/观察器、源码和 checkpoint 审计以及逐字段比较脚本归档为 `external-evidence:moe508-complete-raw.json.gz`，SHA256 `aff71c46aeac25d0daac4c0294f69bf18fb993443cab9df0b71e21bfc30d49f8`，核对全部 30 个归档文件。追加的逐字段独立比较和链接摘要归档为 `external-evidence:moe513-actual-history-boundaries-supplement.json.gz`，SHA256 `11f5d749ff34c2c2792ee142dcdb8f1b1ef74cc9ab7917367ddc92884db670a7`，核对全部 32 个文件。该证据只覆盖首层及这四个 decode 步，不宣称整模型功能或精度验收完成。


### Prefill K/V 与 decode 历史缓存逐位链接（2026-11-06）

利用同一批次的真实 prefill 和目标 decode 操作数，按两条请求的实际序列长度，把 prefill 产生的前 24/23 行 K/V 与目标 decode 读取的历史缓存前 24/23 行逐行对齐。两条请求合计 47 行，K 和 V 均 **24064/24064** 逐位匹配；每个请求分别为 K/V **12288/12288** 和 **11776/11776**。这直接证明本次观察中的 prefill 写缓存、物理索引映射和目标 decode 读缓存之间没有发生行错位或尾行替换。

因此此前目标历史缓存相对独立 CPU staged 参考的 2 个英文 K 与 8 个 V 元素差异，不来自已采集路径中的 cache write/read 连接。剩余问题收敛为 HCU 实际 prefill 算术与 CPU staged 参考的舍入边界，或尚未采集的历史计算阶段；不据此修改 adapter kernel 或放宽 V1 门槛。

逐位链接脚本和结果保存于 `external-evidence:moe516-prefill-cache-link-supplement.json.gz`，SHA256 `76d763187944c8d175faacd8fd94781ca87153e6bcbe6948ccb7810d5b8ce4a5`，核对全部 32 个文件，并绑定真实 GPU 操作数归档摘要。


### 历史 KV 残差坐标归因（2026-11-06）

对真实 GPU508 的 prefill 和四个 decode 边界逐字段列出与 staged FP32-RoPE/BF16 参考的差异坐标。英文 K 的两个差异全部位于 prefill 的 token 行 **4/7**、KV head/dim **(1,84)/(3,44)**，并与目标历史缓存的两个 K 差异逐项相同。英文 V 的四个差异全部位于 prefill 行 **1/5/9/10**；中文 V 的三个差异全部位于 prefill 行 **1/5/9**，另有一个差异来自第 2 个 decode。四个 decode 的 K 均无差异，目标 decode 的 K/V 结论因此可进一步收敛到少数历史 token 的算术舍入。

这组坐标与 prefill/cache 逐位链接结果相互印证：差异在 prefill 实际输出产生，并原样进入历史缓存；没有证据指向 attention softmax、KV 物理索引、cache 写入或 decode 读取路径。该结论仍是实际操作数与独立 staged 参考之间的局部数值归因，尚未模拟 HCU GEMM/归一化每条指令，也不替代整模型 V1。

坐标脚本和结果保存于 `external-evidence:moe519-residual-coordinate-supplement.json.gz`，SHA256 `4c636775bf4632d89915ecdb9d2c07eab84a5ed673ba9f419c23742fe755619f`，核对全部 32 个文件。

### 历史投影输入与累加顺序敏感性（2026-11-06）

仅使用已归档的实际 HCU 操作数及独立 staged 参考，在无 Torch、无设备调用的 CPU 标量进程中核对 prefill 与四步 decode 的全部投影输入。共 **112639/112640** 个元素逐位匹配；唯一差异位于 prefill 英文行 **5**、hidden dim **50**，实际值为 `5.745887756347656e-5`，参考为 `5.7220458984375e-5`。保持同一实际权重，以实际输入作 FP64 乘积累加再转 FP32→BF16，可重现该行 V 残差 `(head=1, dim=86)`；换回 staged 输入则重现 staged 投影值。这将该坐标的差异与投影前输入联系起来，但尚未解释该输入本身的生成差异。

对上述 2 个 K 和 8 个 V 残差对应的 **10 个投影坐标**，分别计算 FP64 累加、FP32 顺序累加及 FP32 二叉树求和，再转 BF16。按坐标分别选择变体，有 **6/10** 个可以重现实际值；这不是一个统一算法重现全部坐标。英文及中文相同前缀的一个 V 坐标和另一中文 V 坐标可由顺序累加重现，一个英文 K 坐标与第 2 个 decode 的中文 V 坐标可由二叉树求和重现，另一个 V 坐标为上述输入差异。其余 **4 个**坐标未由这些变体重现，不能据此将全部残差归因于归约顺序。

以实际 raw K 投影作独立 FP64 RMSNorm 并转 FP32→BF16，两处 K 归一化残差坐标均重现实际值。因此这两处 normalized K 相对 staged 参考的差异已在投影输出边界出现，不能直接归因为 K 归一化错误。本实验没有模拟 HCU MFMA 指令、证明每个累加步骤或验证后续所有层；默认 BF16、社区源码、生产代码和原 V1 门槛均未改变，整模型验收仍未通过。

核对两个来源归档全部 **37 个文件**摘要；脚本、结果及来源摘要保存于 `external-evidence:moe523-arithmetic-attribution-supplement.json.gz`，SHA256 `31b23a7376b609724fda425aba8552872db1f431a961090b55d4b397967415aa`，补充归档全部 **8 个文件**摘要核对通过。下一步需解释唯一投影输入差异及未重现的投影坐标，并继续沿后续 attention/MLP 边界定位误差传播。

### 首层输入归一化与 attention 局部敏感性（2026-11-06）

从已绑定 checkpoint embedding 与输入归一化权重重算上述 prefill 英文第 5 行。FP64 RMSNorm 后转 FP32→BF16 重现 staged 参考，匹配实际值 **2047/2048**；采用 FP32 平方归约、均值/逆平方根及逐乘转换的两种数学变体，顺序归约和二叉树归约均重现实际 **2048/2048** 个元素，包括 dim 50。这为唯一投影输入残差提供了 FP32 算术边界的局部解释，没有唯一确定设备归约顺序或证明每条硬件指令。

在此前已绑定的双行首层目标 decode 操作数上，固定 attention scale、物理历史和输出投影权重，按社区指数权重转换顺序做离线敏感性分析。实际 Q/K/V 的数学参考再次逐位重现 **8192/8192** 个 context 和 **4096/4096** 个输出投影元素。只换成独立 staged Q 时，context 和输出均不变；只换 K 时，context 改变 **1** 个元素、输出改变 **4** 个元素，输出最大变化 `2.384185791015625e-7`；只换 V 时，context 改变 **4** 个元素、输出改变 **41** 个元素，输出最大变化 `6.103515625e-5`。同时换 Q/K/V 时，context 改变 **5** 个元素、输出改变 **43** 个元素，输出最大变化仍为 `6.103515625e-5`。

实际输出与全部替换后的输出，对已保存官方 CPU FP32 首层输出的全批次最大绝对差均为 `0.00048661231994628906`；RMS 差分别约 `3.61359e-5`、`3.61260e-5`。因此这些局部替换没有消除该位置相对 FP32 的全部首层差异，不能据此解释或豁免整模型 V1。Q/K 替换同时包含 CPU FP32 三角函数与实际缓存的差异；实验只覆盖当前目标位置，没有将替换结果送入后续层或重新运行模型。

NumPy CPU 的独立矩阵表达式逐位复核全部 **5 组** context/输出张量，每组 **8192+4096** 个元素；三个来源归档全部 **673 个文件**摘要核对通过。脚本、完整离线输出、结果及复核归档为 `external-evidence:moe528-context-sensitivity-supplement.json.gz`，SHA256 `775078ed6f32e103643168bfc5595b9e1e6cf188d05fd7b598bdf87eab27361c`，补充归档 **8 个文件**摘要核对通过。没有新增 GPU 运行、生产修改或精度通过结论；后续应继续核对 attention 后残差/归一化与 MLP 边界的误差传播。

### 全层实际残差与归一化合同（2026-11-06）

在同一真实双语 BF16、TP1、eager 请求的输出位置 4，补采全部 **48 层、96 次**输入/attention 后归一化。计算前独立保存输入、原残差和权重，计算后保存归一化输出及写回残差，同时保存 attention/MLP 输入输出；没有通过物化待执行贡献来改变边界。采样器的 9 项 CPU 预检通过，真实观察/控制输出 ID 与四类完整概率字段逐位一致。社区源码前后审计、受测快照 603 个文件及当前 473 个 Python 文件摘要核对、checkpoint 25 项资产审计均通过；控制器与 worker 正常终结。

实际调用记录显示全部 **96 次** norm 进入社区 `forward_cuda`；样本中均未启用 HF 乘权重前转换或 FP32 residual。按实际输入构造 FP32 平方二叉树归约、均值/逆平方根及逐乘转换的数学参考，归一化输出 **393216/393216** 个元素逐位匹配；**95 次**带残差调用的写回结果 **389120/389120** 个元素逐位匹配。48 层内 norm→attention、attention→post norm、post norm→MLP，以及 **47 处**前层 MLP/残差到后层输入 norm 的连接逐位一致。独立 NumPy 向量表达式重新计算全部输出和写回值，得到相同结果；这验证当前样本的数学合同，不将二叉树匹配解释为设备指令序列的唯一证明。

社区默认融合合同以未舍入的 FP32 和计算方差，以写回激活 dtype 后的残差作分子。保持该转换位置而改用 FP64 方差/乘积时，输出匹配 **393215/393216**，唯一差异位于第 27 层 post-attention norm、英文 dim 1197，最大绝对差 `0.0078125`；上述 FP32 变体可重现该元素。使用未舍入分子或使用已舍入残差计算方差的其他变体，分别仅匹配 **309583/393216、361909/393216**。不能用这些不同合同判定当前融合实现错误，也没有据此修改 norm 内核。

本轮 attention/MLP 的 **192 组**输入输出与既有全层采样逐位一致，已返回概率的相同前缀也逐位一致。由下一层实际输入 norm 的写回残差，直接核对前 47 层结束状态与此前绑定的官方 CPU FP32 参考：第 0 层后英文/中文相对 RMS 差约 **0.003667/0.003833**；英文第 26→27 层后约 **0.013565→0.035486**，中文第 29→30 层后约 **0.035950→0.056736**。这些变化与此前专家集合差异位置相邻，但没有执行路由替换实验，不能单凭相邻指标证明因果。第 47 层后的最终写回残差不在本轮采样范围。

真实张量、采样器、前后审计及受测字节归档为 `external-evidence:moe538-complete-raw.json.gz`，SHA256 `ede90b45c9734e88c64e36b020137a589ab7a22bc4993dfa6a969740bb29cf39`，核对全部 **778 个文件**摘要；CPU 标量分析、独立 NumPy 复核及跨运行比较保存于 `external-evidence:moe548-residual-contract-supplement.json.gz`，SHA256 `94651abd8f9edea065ced454107615f1a5318aa527c3d368d9d004adfb954a2c`，补充归档 **9 个文件**摘要核对通过。当前样本没有残差连接或归一化合同错误的证据；整模型原 V1 失败仍保留，下一步应继续核对 MLP 计算及路由变化附近的误差传播。

### 首次专家集合变化附近的真实权重局部对照（2026-11-06）

对上述目标位置的英文第 26/27 层、中文第 29/30 层，用已绑定 checkpoint 的真实专家权重、实际 MLP 输入、专家 ID 和路由权重执行官方 CPU FP32 experts。逐项固定输入、专家集合或路由分数，得到每个位置 **8 组**局部输出；单行计算与既有官方完整序列 CPU FP32 输出的最大差不超过 `1.79e-7`。独立 FP64 逐专家表达式复核两组端点，最大差不超过 `1.92e-7`。本轮隐藏 GPU，只重放局部专家，没有运行整模型路由替换。

实际 HCU MLP 输出与原 CPU FP32 输出的 RMS 差，在英文第 26/27 层分别约 **0.00115109/0.01244372**，中文第 29/30 层约 **0.00635370/0.01832522**。固定实际 HCU 输入、专家 ID 和路由权重后，CPU FP32 experts 与实际 HCU 输出的 RMS 差分别仅约 **0.00011818/0.00012842/0.00034991/0.00025903**。这些局部差仍然存在，不能据此证明对应 BF16 专家内核逐位正确。

在实际 HCU 输入和 HCU 路由分数固定时，仅将 CPU 专家集合换成 HCU 专家集合，英文第 27 层输出 RMS 变化约 **0.01242358**，与实际总误差向量的余弦约 **0.99612**；中文第 30 层分别约 **0.01740054/0.94570**。前一层专家集合相同时，这一项变化为 **0**。固定 HCU 分数而使用 CPU 专家集合的局部输出，相对原 CPU 参考的 RMS 差分别约 **0.00108144/0.00595717**；使用实际 HCU 路由时约为 **0.01244183/0.01832993**。因此专家集合变化显著放大了这两个位置的局部 MLP 偏差，但还不能唯一归因于路由实现、解释整模型 V1，或将 RMS 分量当作可直接相加的误差占比。

checkpoint **25 项**资产前后指纹、受测快照 **603 个文件**、官方模型源码及 **15 个**历史输入文件摘要均核对通过。标准库标量计算独立复核 **92 组**完整指标，并将保存的输入、输出、ID 和权重与原始采样逐位对齐。完整归档为 `external-evidence:moe553-route-replay-raw.json.gz`，SHA256 `3433a66740ff51b1512feba260b93b65c1a12fbb63184a1c0644b3c3d167674b`，**31 个文件**摘要核对通过；复核脚本和结果补充归档为 `external-evidence:moe557-route-replay-supplement.json.gz`，SHA256 `00116d7e99108d810430fb3bd374445a4bffb449402d852e75dba5e808043b12`，**9 个文件**摘要核对通过。默认 BF16、生产代码及原始精度门槛未变；下一步需核对这些层实际专家 GEMM/激活/归约边界，保留整模型原 V1 失败。

### 专家集合变化附近的实际计算链（2026-11-06）

在同一默认 BF16、TP1、eager 双语请求的输出位置 4，补采第 **26/27/29/30 层**实际两段专家 GEMM 的输入、所选专家权重、输出、激活边界和最终归约。沿用已验证观察器的 8 项 CPU 预检全部通过；真实观察/控制的输出 ID 和四类完整概率字段逐位一致。社区源码前后、checkpoint **25 项**资产前后、快照 **603 个文件**和当前 **473 个 Python 文件**摘要核对通过，GPU 和后续 CPU 重算任务均正常终结。

每层两行请求实际共选中 15 个不同专家。采样中的 gate 权重和两段所选专家权重，与 checkpoint 的 **184 个张量**逐位一致；输入、输出、gate 和 router 的 **24 组**边界与此前全层采样逐位一致，返回概率的相同前缀也全部一致。以实际输入和权重作独立 FP64 乘积，再转 FP32→BF16，gate logits **1024/1024** 个元素匹配；64 个专家 ID 与实际分数排序一致，路由权重相对独立标量 softmax/重归一化的最大差约 `1.73e-8`。

实际第一段 GEMM 输出经独立 SiLU×up 并转 FP32→BF16，重现全部 **49152/49152** 个激活元素；实际第二段输出经 FP32 路由维求和并转 BF16，重现全部 **16384/16384** 个最终 MLP 输出元素。两段 GEMM 本身相对上述 FP64→FP32→BF16 参考，共匹配 **229347/229376**：第一段仍有 **13/98304** 个差异，最大绝对差 `0.00390625`；第二段仍有 **16/131072** 个差异，最大绝对差 `0.0001220703125`。保留这 **29 个**算术残差，不将近乎逐位匹配解释为设备指令重放或全层内核正确性证明。

不导入 Torch 的独立 NumPy 表达式重新计算全部 8 段 GEMM、激活、归约、gate 和路由，复核 **36 组**完整指标，得到相同统计。原始操作数、受测字节、观察器及前后审计归档为 `external-evidence:adapter-moe559-complete-raw.tar.gz`，SHA256 `afeec3a7ab19696d4dee05ed387916042a32ef3af3211326c3912fcc3254cb93`，**659 个文件**逐项摘要核对通过；复核与资源审计补充归档为 `external-evidence:moe571-four-layer-components-supplement.json.gz`，SHA256 `9c1ebb5c9004de731f83d0fff635c7b9bbf952ed479f4474c04828f6cfc3f32b`，**12 个文件**摘要核对通过。自有任务已退出并释放设备；没有修改生产代码、默认精度或验收门槛。局部路由变化对整模型失败的贡献仍需验证，原始 V1 失败继续保留。

### 真实 serving 目标位置的路由干预（2026-11-06）

使用私有诊断 hook，在原 BF16、TP1、eager 双语请求的输出位置 4，仅将英文第 27 层或中文第 30 层的专家集合替换为对应 CPU FP32 集合；保留该位置实际 HCU 输入与 gate logits，按原分数重归一化所选权重，再执行其余模型层。此操作主动改变该请求的计算语义，只用于因果诊断，未进入生产代码，也不是模型验收或建议默认配置。

22 项隐藏 GPU 的预检通过。真实运行包含无干预观察、同集合重算权重、仅英文替换、仅中文替换和双请求替换五组。无干预观察、同集合重算及移除干预后的控制请求，与初始控制的 ID 和四类完整概率字段逐位一致；每组干预前四个输出 token、相应概率前缀及完整输入概率不变。未干预请求全部不变，双请求干预的各行与对应单请求干预逐位一致。保存的 **10 组**目标路由记录验证输入行、token/position、分数、集合与权重，两段实际 GEMM 均使用干预后的路由。

对已绑定官方 CPU FP32 的同一目标位置，英文最大 top-20 logprob 差从 **0.60607529** 降为 **0.41857719**；中文从 **3.97164822** 升为 **4.42731380**。只比较各自控制与干预共同出现的类别，英文 **20 个**、中文 **19 个**类别仍得到上述最大差，因此不能单凭局部 MLP 输出靠近 CPU，推导最终分布必然改善。两种干预仍超过原 `0.15` 门槛；该位置 greedy regret 均为 0、其他指标较小也不能覆盖失败项。top-20/tail JSD 仍只是按各观察类别合并尾部的指标，不是全词表散度。

不导入 Torch 的标准库表达式独立复核 **72 个**指标值、10 组路由与原 V1 策略判定，核对同集合对照、请求隔离及恢复控制均一致。社区源码前后、checkpoint **25 项**资产前后、快照 **603 个文件**、当前 **473 个 Python 文件**及 **16 个**输入/私有脚本摘要核对通过。完整归档为 `external-evidence:moe587-complete-raw.json.gz`，SHA256 `65d9feb198392c8eaf644b8f0d2c6575ce97baffddc7d03d9bc3710a452b3cf0`，全部 **667 个文件**摘要核对通过；复核与资源审计补充归档为 `external-evidence:moe591-serving-intervention-supplement.json.gz`，SHA256 `120d487152acc588c03f4a31f8bc559299693637574929aea9e91706f0141e0f`，全部 **13 个文件**摘要核对通过。任务已终结并释放自用设备；不采用强制路由作为修复，原始完整 V1 失败继续保留。


### 完整 VL2 图像请求与视觉路径（2026-11-06）

在原样社区 SGLang 0.5.22、完整 DeepSeek-VL2 checkpoint 上完成 BF16、TP1、Triton attention、eager、显存比例 0.7 的六组图像请求：红/蓝正方形、重复红图、同图 base64、绿色宽长方形和红蓝双图。四组颜色/形状固定样例均回答正确；每组保存 24 个强制解码位置的 token、所选概率和 top20，包含 EOS 后位置，不将其称为 24 个有效回答 token。沿用上述固定 cookbook commit；该版本没有 VL2 专用配方，不能作为同配方认证。

只读 hook 捕获四组新图请求的视觉编码器末层和投影层，共 **8 组**有限 BF16 输出；红蓝图的特征摘要不同。另保存四份真实输入/输出张量，尚未独立重算其数值。重复图片未再次触发这两个 hook；相同 token 序列不等于概率一致：相对首个红图，重复文件/base64 的所选 logprob 最大差分别为 **0.018984318/0.056817532**，共同 top20 最大差 **0.4375/0.249987602**；首个 EOS 前仍为 **0.312483788/0.189036846**。这是同设备重复请求差异，不替代独立 CPU FP32 V1 比较，也不据此单独判定图像缓存错误。

初次私有探针虽然完成六组生成，但两个 hook 使用相同文件名，且错误要求重复图片重新执行视觉层，因此保留为探针失败。修正日志唯一性及新图/复用请求断言后，隐藏 GPU 的观测器预检验证四个独立记录、张量归档与输入/输出不变；真实复测正常终结。两轮社区源码、15 项 checkpoint 资产、603 个快照文件与当前 473 个 Python 文件前后核对通过，全部自有进程终结并释放设备。

完整成功归档 `external-evidence:vl604-complete-raw.json.gz`，SHA256 `7608ef371122ebbeb3c9e660e9c52bca05e9cab3b421b3f0f62c22dd6d2cebd1`，**650 个文件**；失败探针归档 `external-evidence:vl599-complete-raw.json.gz`，SHA256 `209f979ddfa8f7472c670ad252b29dd0676d32d837036e02a4aa2fca4b80b14b`，**639 个文件**。标准库独立验证全部摘要、两轮 12 请求/288 位置、固定样例、重复概率差和视觉记录；补充归档 `external-evidence:vl608-image-functional-supplement.json.gz`，SHA256 `1f932b357fa1aaaaa96b921ce6215068d3a0ec464b0a2a31b466edde9822fb7e`，**13 个文件**。本轮只补齐上述样例的 eager 图像功能证据；视觉独立 CPU FP32、图像请求 graph 验收及既有整模型 V1 失败仍未闭合，没有生产代码修改或整模型通过结论。


### VL2 图像请求的实际 decode graph（2026-11-06）

对上述同一完整 checkpoint、相同图片字节及六组请求，保留 BF16、TP1、Triton attention 和显存比例 0.7，启用 decode graph 并限制最大 batch 为 2。六组请求及四组固定颜色/形状检查完成，实际捕获 batch 1/2；服务报告的 decode graph 分配字段为 **0.266**，eager 为 **0**。本轮没有验收 prefill graph。

eager/graph 的 **144/144** 个输出 token 相同，四组新图的视觉编码器末层及投影层，共 **8 组**输入/输出摘要全部相同；但六组请求的所选概率与 top20 均非逐位一致。相同历史下，所选 token logprob 最大差 **0.056288540**，共同 top20 最大差 **1.34375**；只取各请求首个 EOS 前位置时，共同 top20 最大差仍为 **0.562137127**。这些是两种设备执行模式间的诊断差异，不是独立 CPU FP32 V1 结果；一致的视觉边界不足以唯一定位后续差异，且 eager 自身重复请求也存在概率变化，不能单独归因于 graph。

控制器与 worker 正常终结，社区源码、15 项 checkpoint 资产、603 个快照文件及当前 473 个 Python 文件审计通过，graph 任务释放设备。完整归档 `external-evidence:vl610-complete-raw.json.gz`，SHA256 `24c9205786b0f45bfa9737be5a218d6e2204d719e9633a81de9d1b6538a4d1eb`，**650 个文件**；标准库独立复核两模式的完整概率、实际分配、同历史比较和视觉摘要，补充归档 `external-evidence:vl617-image-graph-supplement.json.gz`，SHA256 `568c6d8a65385ce8bd5e11616d98ce2f2936ea36bef522b45111021b269bbd14`，**7 个文件**。上述样例的 decode graph 图像功能证据已补齐；视觉独立精度参考、重复概率差与原整模型 V1 失败仍保留。


### VL2 视觉与投影边界的独立 CPU FP32 对照（2026-11-06）

在隐藏 GPU、未加载 SGLang/宿主桥的独立 CPU 进程中，按社区所用 timm 架构严格加载同一 checkpoint 的 **342 个**视觉张量，并用另外 **4 个**投影张量执行独立 FP32 2×2 分组、Linear、GELU、Linear。视觉加载值逐项核对，投影权重直接由绑定 checkpoint 转为 FP32，保留原图、FP32 像素和强制 SDPA MATH；参考库源码未改写。该参考使用 timm 和独立投影表达式，不是官方完整 VL2/xformers wrapper，也没有执行语言解码或整模型 V1。

对红/蓝图，CPU FP32 完整视觉输出相对实际 HCU 的相对 L2 差分别为 **0.087212765/0.095354095**，余弦 **0.996206863/0.995477816**；将输入像素先转 BF16 再转 FP32 的诊断变体，相对 L2 差仍为 **0.086542693/0.094085969**。完整参考投影输出的相对差为 **0.046647056/0.048896958**；固定实际 HCU 投影输入后，仅为 **0.002180118/0.002163012**。这些边界差异没有独立的验收阈值，不能用于豁免原概率门禁或唯一归因于某个内核。HCU 原始像素输入尚未捕获，需继续核对实际输入、裁剪与图文拼接。

初次私有参考在 timm 的 meta 初始化阶段失败，未执行视觉前向；改用普通 CPU 构建后正常完成。标准库限制性张量读取器按真实 stride/offset 解码，独立重算两图 **72 个**指标值，并核对 PNG CRC、全部 RGB 像素及 **1769472/1769472** 个 FP32 归一化元素；没有以此声称再次独立执行整个视觉网络。checkpoint、社区源码、参考库源码和受测 adapter 字节前后核对通过；自有任务终结。

完整 CPU 参考及失败构建证据归档 `external-evidence:vl618-complete-raw.json.gz`，SHA256 `3b58f862bc2eb8518b4f7c7d6267ea244b2465f65ce9c5b23f471e2e50cc1de7`，**662 个文件**；独立标量验证补充归档 `external-evidence:vl623-vision-reference-supplement.json.gz`，SHA256 `4e942cadb8ba1e08a4322855633262aaee6764cee8ec37e98990a3916f0f940b`，**8 个文件**。本轮没有生产代码修改、独立图像概率 V1 或整模型通过结论。


### VL2 实际像素与首层图文嵌入对齐（2026-11-06）

在相同原样社区源码、完整 checkpoint、BF16/TP1/Triton/eager 配置下，重复六组图像请求并增加两次无采样标记的红图控制。先用无图文本预热安装首层归一化的计算前 hook，再捕获实际像素、视觉末层、投影和首层输入，共 **156 条**边界记录；四组固定颜色/形状检查通过。首轮私有脚本在预处理阶段提前初始化设备，触发自身断言，未执行模型推理；改为隐藏 GPU 的独立预处理子进程后，完整采样正常结束。未修改社区源码或生产代码。

红/蓝两图实际像素分别有 **884736/884736** 个元素等于社区处理器 FP32 输出转 BF16，处理器 FP32 像素又与此前独立 CPU 参考逐位相同。以实际 HCU 投影输出、同 checkpoint 的 newline/separator 和文本 embedding 重建各自 **437×2560** 的首层输入，其中 421 个图像位置、16 个文本位置，全部 **1118720/1118720** 个元素逐位一致；两图后续各 23 步的文本 embedding 也逐位匹配对应生成 ID 的 checkpoint 权重。上述检查仅覆盖两张方图，不代表宽图、多裁剪或任意图片的拼接均已验证。

使用原 CPU FP32 视觉/投影结果重建的完整首层输入，相对实际 HCU 的 L2 差为 **0.046295199/0.048513934**，最大绝对差 **0.155847669/0.193399191**；两图各 40960 个文本元素全部相同。实际投影输出也与此前 eager 采样相同。由此在这两组样例中未发现像素预处理、文本查表或固定实际投影后的拼接错误，但不能把视觉算术差异唯一归因于硬件，也尚未获得完整图像解码概率的 CPU FP32 验收结果。

两次无标记控制的 24 个输出 ID 相同，概率仍不同：同历史所选 logprob 最大差 **0.056274652**、共同 top20 最大差 **0.5625**，首 EOS 前为 **0.561938286**。六组观测请求的 cached tokens 和 retractions 均为 0；与此前 eager 的 144 个输出 ID 全部相同、概率仍有差异，因此不声称采样对概率的影响已被逐位排除。标准库独立读取真实 stride/offset 张量，逐元素复核两图像素和图像拼接、文本位置，并重算首层 FP32 差异及控制请求比较；没有独立重放整个模型。

社区源码、15 项 checkpoint 资产、603 个快照文件、当前 473 个 Python 文件及输入摘要核对通过；GPU/CPU 任务均终结并释放自用资源。完整归档 `external-evidence:vl646-complete-raw.json.gz`，SHA256 `6157a68bb713f36c01a94d2cb53636e0553c12287163f0b7a9e60fa9be1ae0a3`，**894 个文件**；独立复核与资源审计补充归档 `external-evidence:vl654-input-alignment-supplement.json.gz`，SHA256 `98d299e24e4166ca03386c026ad7a236fabbae813e462b13610ea2b84592f0e3`，**14 个文件**。默认精度和原 V1 门槛不变，既有整模型精度失败仍保留。


### VL2 图像请求的完整语言概率组合参考（2026-11-06）

在隐藏 GPU、未加载 SGLang/宿主桥的独立 CPU 环境中，以此前严格绑定 checkpoint 的 FP32 视觉/投影和图文嵌入，连接未改写的官方 DeepSeek-VL2 语言解码器，完成红/蓝两张方图各 24 个固定历史输出位置的完整词表概率对照。官方参考源码 commit 为 `ef9f91e2b6426536b83294c11742c27be66361b1`；完整模型的 **6973 个**加载张量逐项等于原 checkpoint 转 FP32，使用 eager MLA。该组合参考没有执行官方自定义 xformers 视觉 wrapper，也不代表所有图片、模型家族或配置矩阵的验收。

主参考使用原 FP32 图像嵌入，两图 **48/48** 个参考 top1 与实际输出 ID 相同；最大所选 logprob 差 **0.116638129**，最大 top20 logprob 差 **1.437827897**，最大 top20/tail JSD **0.002406400**。后两项分别超过原 `0.15/0.002` 门槛；即使只查看各图首个 EOS 前位置，红/蓝最大 top20 差仍为 **0.570733911/0.289894767**。没有用 greedy 一致或排除 EOS 后位置来豁免原完整比较的失败。

另以实际 HCU 首层图文输入转 FP32，固定视觉输入后执行相同官方语言解码器，作为独立诊断。最大所选 logprob 差为 **0.038723725**，最大 top20 差仍为 **0.976472369**；首 EOS 前红/蓝分别为 **0.615331323/0.434464051**，仍超过原门槛。不能用该固定设备输入的结果替代主参考，也不能据此把全部差异归因于视觉路径。此前重复红图与 base64 请求的首层归一化输入和输出，共 **48 组**跨请求比较全部相同，而最终概率仍不同，需继续定位后续计算边界。

首轮参考的 CPU FP32 `log_softmax` 有一行概率和偏差约 **0.000113973**，被既有 `1e-4` 校验拒绝；保留失败记录后，重跑保持模型计算 FP32，并保存原始 FP32 logits，使用 FP64 计算概率归一化。未改模型精度、生产代码或原 V1 阈值。标准库从原始 logits 独立重算四组共 **12410880 个**概率元素及全部 V1 指标，归一化计算的最大差不超过 **3.56e-14**；首轮旧概率按独立标量重新归一化后，与新参考最大差约 **2.15e-6**。主参考和固定输入诊断的原策略判定均为失败，top20/tail JSD 仍仅为合并尾部的分布指标。

CPU 任务正常终结；checkpoint 15 项资产、官方参考源码、603 个 adapter 快照文件、当前 473 个 Python 文件及输入摘要审计通过。完整归档 `external-evidence:vl668-complete-raw.json.gz`，SHA256 `daedbe63658d908df51a276c85c5ae8289a517de078df337523ad488c335c328`，**683 个文件**；独立概率重算、首层重复比较和 CPU 资源审计补充归档 `external-evidence:vl679-image-reference-supplement.json.gz`，SHA256 `3fd0c7763614edf9bc62cfeb5b0a1a9763cee1a32b60e87a28f4fac4887e5f9d`，**11 个文件**。本轮补齐上述两图的组合概率参考证据，既有整模型精度失败仍保留。


### VL2 重复请求的首个解码差异边界（2026-11-06）

在相同 BF16/TP1/Triton/eager 配置下，对六组真实图像请求增加全部 30 层的输入/注意力后归一化、最终归一化及 logits processor 观测。无图预热安装 61 个计算前归一化 hook，共捕获 **8940 条**边界记录；预检验证计算前克隆、kwargs、tuple、唯一记录及 logits 数据结构，四组固定颜色/形状检查通过。六组请求 cached tokens 和 retractions 均为 0，没有修改社区源码、生产代码或默认精度。

对重复红图和同图 base64，输出 ID 各 24 个全部相同，但概率仍不同：共同 top20 logprob 最大差分别为 **0.374937057/0.562057495**，首 EOS 前分别为 **0.315783501/0.562057495**。标准库独立核对两组比较的 **2928 组**归一化输入/输出边界：prefill 时，零基第 0、1 层的输入归一化与注意力后归一化均逐位相同，首次输入差异出现在 **第 2 层输入归一化之前**。此处只有第 0 个张量参数不同，残差参数与 kwargs 相同，范围缩小到第 1 层 MoE 输出及其至下一层的衔接，尚不能据此唯一指认某个路由或专家内核。

两组重复比较的全部 **48 个**位置，logits processor 输出的原始 logits 摘要也不同。另从保存的红/蓝图各 24 份 FP32 原始 logits 独立归一化，所选及 top20 logprob 与服务返回值的最大差不超过 **1.17e-6**。因此这些样本的最终概率差异已经体现在采样前 logits，不能仅解释为返回概率的格式或归一化误差；同步观测仍不能作为完全无扰动的证明。后续需要捕获首个 MoE 的实际输入、路由分数、专家 ID/权重和专家输出，区分路由变化、专家计算与输出衔接。

社区源码、15 项 checkpoint 资产、603 个快照文件、当前 473 个 Python 文件及输入摘要前后审计通过，任务正常终结。完整归档 `external-evidence:vl687-complete-raw.json.gz`，SHA256 `89cba6cbc36babaaea56bc8f83f4c8cb99d17c0c029d42cf5d8c8ba6a7f702da`，**9752 个文件**；独立比较、源接口核对和该任务资源审计补充归档 `external-evidence:vl689-decoder-trace-supplement.json.gz`，SHA256 `26f42c402a1fe17990123b22e5011eb5d7e4edc2a1986017e727008bc1549fe4`，**17 个文件**。本轮只缩小重复差异的位置，主 CPU FP32 参考及既有整模型 V1 失败继续保留。


### VL2 首个 MoE 的专家集合与排列差异（2026-11-06）

在相同 BF16/TP1/Triton/eager 的完整模型观测中，增加零基第 1 层 MoE 的 gate、TopK、routed experts 和 shared experts 计算前/后边界，共保存 **9660 条**记录。重复红图及同图 base64 的 prefill 输入、gate logits 和 shared-expert 输出分别逐位一致；首次输出差异位于 TopK，随后 routed experts 与下一层输入出现差异。逐行比较实际 **437×6** 路由结果后，两个比较分别有 **39/55 行**排列变化，但 **437/437 行专家集合全部相同**，没有越界或重复专家 ID。按专家 ID 对齐后的权重最大差仅 **2.980232239e-8**；不能将原始 ID/权重摘要不同解释为选错专家。

使用保存的真实输入和同 checkpoint 的 FP32 correction bias，在相同 adapter/社区环境中直接回放实际 grouped-TopK 派发。该回放的模块标志为 HIP、非 CUDA、未启用 AITER，模拟 uniform/round-robin 路由与 JIT grouped-TopK 开关均为 false；完整模型服务配置也未启用 EPLB/waterfill。checkpoint bias 的 72 个值全部为 **4.125**，独立标量排名与实际专家集合一致，所有行第 6/7 名最小分差为 **3.337860107e-6**，没有边界并列。该形状的结果不支持非二次幂 padding、随机模拟路由或排名并列的归因。

实际派发、编译 native 和未编译 native 各 **32 次**回放均出现 **32 种**完整排列，专家集合不变。另一次对照回放也复现此现象；只在内存中的独立函数副本将 TopK 的排序参数设为 true，保留原编译装饰器、分数、专家数、归一化和模型精度后，**32 次**排列及权重逐位一致，所选集合仍与原始记录相同。这仅说明固定顺序能稳定此边界；尚未证明整模型重复概率差由此完全引起，更未证明原 CPU FP32 V1 精度失败已解决。生产代码和社区磁盘文件均未修改，整模型排序干预另行验证。

完整模型任务正常结束，源码、checkpoint、603 个快照文件、当前 473 个 Python 文件及输入审计通过；两个最小回放进程均已终结。初次私有 AST 提取遗漏编译装饰器依赖，在执行 TopK 前失败，保留后改用真实导入函数完成回放。标准库独立读取原始张量，复核专家集合、排名、对齐权重及全部五组各 32 次回放的排列统计，不导入 Torch，不以此替代整模型重放。

完整观测归档 `external-evidence:vl703-complete-raw.json.gz`，SHA256 `b13c15be347b91f891cb5f0fde57094af4a73e6b0a7a7f37fd4fcfa6ade1eba7`，**11247 个文件**；回放与独立专家集合验证补充归档 `external-evidence:vl710-topk-supplement.json.gz`，SHA256 `b8856910c59c7ecab402582d0d4fbae428bbf963d5a8acce713bcf32fdf47aaf`，**46 个文件**；回放统计独立验证归档 `external-evidence:vl712-replay-verification.json.gz`，SHA256 `c4d50528ade042a8ee34bd2b2530437411ba8724133903cbadd1935c1ba46e68`，**3 个文件**。上述定位不授予整模型、模型家族或配置矩阵验收，既有 V1 失败继续保留。


### VL2 固定 TopK 排列的整模型诊断（2026-11-06）

在相同原样社区源码、完整 checkpoint 和 BF16/TP1/Triton/eager 配置下，只对进程内 native grouped-TopK 函数副本启用排序，保留原编译装饰器、专家集合、归一化及其他计算。六组图像请求各 24 个位置，均实际经过 **29×24=696 次**诊断函数；运行时记录确认模拟路由、waterfill 均关闭，实际 TopK 为 72 专家选 6、单组 sigmoid、FP32 correction bias。该干预属于私有因果诊断，没有安装为生产默认或改写社区文件。

相对首个红图，重复文件、同图 base64 及两次无观测标记控制的 **24/24 个**输出 ID、所选 logprob 和 Top20 logprob 均逐位一致，最大共同概率差 **0**。标准库另核对重复文件/base64 各 **1608 组**计算边界，输入、kwargs、输出摘要全部相同，共 **3216 组**，包含归一化、首个 MoE 及原始 logits；四组固定颜色/形状检查通过。这说明固定排序在这些样例中消除了重复差异，但不外推为任意请求、graph 或多卡配置通过。

严格确认两图与既有 CPU FP32 组合参考具有相同 checkpoint 绑定、图片字节、实际首层图文嵌入和全部输出历史后，复用原参考的完整词表概率重新执行原 V1 比较。主参考最大所选 logprob 差 **0.115541225**、greedy regret **0**，最大 Top20 差 **0.604931366**、Top20/tail JSD **0.002448766**，后两项仍超过 **0.15/0.002**。固定实际 HCU 嵌入的诊断最大 Top20 差 **0.547583222**，同样失败。因此重复一致性改善不能替代原 FP32 精度验收，也不能据跨运行最大值的变化声称精度问题已修复。

任务正常终结；社区源码、checkpoint、603 个快照文件、当前 473 个 Python 文件及输入摘要审计通过，自有进程均终结。标准库独立重算四组概率比较的全部原策略指标和判定，并验证重复结果、实际调用次数与边界记录。初次比较探针仅因私有脚本导入位置错误提前失败，修正后复用原参考完成，没有重跑或修改 CPU 参考模型。

完整归档 `external-evidence:vl719-complete-raw.json.gz`，SHA256 `fbbe48494c49fbc4fe93dd66471ddba57594ef6e583b958e5d728a50855c185a`，**11248 个文件**；独立验证与资源审计补充归档 `external-evidence:vl722-sorted-model-supplement.json.gz`，SHA256 `a654bb1ff242bc59d466048f524205ce9a8b6713c2fcc8aa0d00cad0fc9e45f0`，**18 个文件**。下一步将已验证的稳定排序收敛为保留函数别名、可验证源码完整性和配置边界的 adapter 修复，再回归真实模型；原整模型精度失败仍需继续定位。


### 原生 grouped TopK 稳定排序的正式运行时修复（2026-11-07）

新增 `hcu_adapter.moe_topk_order`，由 SGLang 宿主桥在启动期安装，默认启用 `HCU_MOE_STABLE_TOPK`，新进程设为 `0` 可关闭。原生 grouped router 的两处 `sorted` 关键字操作数经指令与签名核对后，仅在 GPU 私有函数副本改变对应布尔常量；原指令、Torch 模块、编译入口和已有函数别名均保留，CPU 调用原副本，其他路由内核不受此桥影响。排序发生在原归一化之前，保留分组、偏置、激活分数、共享专家与缩放；平局 ID/顺序继续遵守社区未指定的 TopK 契约。没有新增社区磁盘写回。

实现过程中，真实 Torch 构建的额外包装、预热缓存对通用可变参转发的不兼容，以及私有 Torch facade 的 compiler guard 失败均已保留为失败证据。最终方案保留所有包装和原参数帧，改用上述窄常量适配。远端隐藏设备的 CPU 契约与基础算子 **46 passed**；设备专项 **17/17 passed**，覆盖 FP32/FP16/BF16、单组/多组、偏置选择与权重隔离、归一化/缩放、平局合法性、已有别名、预热编译、新解释器和变输入 graph。真实社区已编译 router 的 **32 次**回放与原排序诊断的 ID/权重全部逐位一致，标准库独立读取原始输出复核通过。快关另在真实 SGLang 导入路径验证桥未安装；隐藏设备的初次导入因社区 capability 探测失败，未冒称快关成功。

完整 DeepSeek-VL2 checkpoint 在正式桥、BF16/TP1/Triton 配置下完成 eager 与 decode graph 两轮，各六组图像请求和两次无标记控制。两轮各自的重复文件、base64 及控制请求 **24/24** 个 ID、所选概率和 Top20 概率均逐位一致；两模式的八组结果也全部一致。graph 实际捕获 batch 1/2，报告 decode graph 分配 **0.243**，eager 为 **0**；不代表 prefill graph 验收。eager 的重复文件/base64 共 **3216 组**输入/kwargs/输出边界全部相同，两轮固定颜色/形状检查通过。默认精度、采样和原 V1 门槛不变。

按相同 checkpoint、图片、实际首层嵌入及完整输出历史复用原 CPU FP32 组合参考，正式桥的主参考最大所选 logprob 差 **0.115541225**、Top20 差 **0.604931366**、Top20/tail JSD **0.002448766**；固定实际 HCU 输入诊断 Top20 差 **0.547583222**。标准库独立重算原 V1 指标和判定均仍为失败，未以一致性修复替代数值验收。两轮社区源码、checkpoint、605 个候选文件及 475 个 Python 文件审计通过，自有模型与组件进程终结；提交前仅有环境变量说明文字更新，与受测行为源码逐项绑定。

远端另执行 `tests/kernels tests/adapter tests/vendor -k "not stress"`，结果 **1738 passed、47 failed、137 errors、137 skipped、2 xfailed、1 deselected**。逐项 XML 分诊将全部失败/错误定位到 vLLM 测试组（含依赖 vLLM 环境的 model-mutation）：缺失目标源码环境、当前安装版接口差异、设备已初始化断言和子进程超时等，按当前用户范围保留，不称全量通过。ops 分诊四组通过；Triton 组 **409 passed、1 skipped、2 xfailed**、FA 场景组 **53 skipped**、AR 组 **2 passed、1 skipped、1 deselected**，仍记 incomplete。L0 与 diff 检查通过，已有 31 项裸环境变量警告保留。

组件和失败实现归档 `external-evidence:moe744-components-raw.json.gz`，SHA256 `1b90a952b236c4ef17ed0e2e36944082df06728bd30ce27ca9785b8ffd1d61fe`，**3062 个文件**；正式模型两模式和广泛回归归档 `external-evidence:vl749-models-raw.json.gz`，SHA256 `cc1e4c87126c148e9f6bb3422d3c06f419c1f64adbbbd162a587bc1c86515602`，**11826 个文件**；独立复核、分诊及快关补充归档 `external-evidence:moe758-stable-routing-supplement.json.gz`，SHA256 `692f4655f7bd86a05d384f99164da1f3a0218817f8707f55275ba04758b51fb2`，**35 个文件**。本功能关闭已复现的 TopK 排列不稳定问题，整模型 FP32 精度、多模型及其余配置矩阵验收继续推进。


### VL2 输出层的固定隐藏状态分解（2026-11-07）

对正式稳定排序桥的 eager 红/蓝图各 24 个输出位置，复用实际采样前 BF16 隐藏状态与原始 logits，在隐藏设备、未导入 SGLang 的独立 CPU 进程，用同 checkpoint 的 `language.lm_head.weight` 做完整 FP32 输出层乘积，再以 FP64 归一化；另保存该乘积舍入到 BF16 的合成对照。实际 head 不与输入 embedding 绑定。本诊断没有重跑完整语言模型，也没有改变生产默认 BF16 或启用 FP32 head。

两图共 **6205440 个**实际 logits 均可由 BF16 精确表示，其中 **6204611 个**与 FP32 head 乘积舍入 BF16 后逐位相同；其余最大差 **0.03125**。对实际 Top20/输出 ID 的并集，保存对应权重并使用独立标量 FP64 点积复核 **10344 个**结果，逐项误差为 **0**；FP32 与 FP64 子集最大差为 **7.95813e-6**。该子集检查不代表完整词表 FP64 GEMM 验证。

实际输出相对固定隐藏状态的 FP32 head，最大所选 logprob 差 **0.036620530**、Top20 差 **0.094825149**、Top20/tail JSD **0.000237545**，局部比较在原阈值内。将该 FP32 head 结果与主 FP32 组合参考比较，Top20 差仍为 **0.610455771**、JSD **0.002234099**；与固定 HCU 图文输入的 FP32 语言参考比较，Top20 差仍为 **0.508096525**。因此输出层 BF16 舍入不足以解释剩余整模型偏差，需继续检查 decoder；尚不能唯一归因某个 attention、MoE 或累加内核。局部通过不替代原整模型 V1 失败。

独立标准库验证不导入 Torch，重算全部 FP64 子集点积、完整词表归一化、BF16 舍入位模式和四组原 V1 比较的六项指标，结果与报告一致。checkpoint 前后字节审计及输入摘要检查通过，CPU 任务正常结束，进程已释放；没有新增 GPU 运行或生产代码改动。原始分解归档 `external-evidence:vl769-head-raw.json.gz`，SHA256 `32ce423dc23f7951dedb1a7a6c44e9dc57f059b849d9fe6e1755503fcad721bf`，**11 个文件**；契约、独立复核及资源审计补充归档 `external-evidence:vl772-head-verification-supplement.json.gz`，SHA256 `90160450e2af2e9011154615bcd7bdaed0a031644a03db0d789de59ba82d3374`，**20 个文件**。精度默认值、原门槛及整体验收未完成状态保持原样。


### VL2 固定图文输入的 FP32 decoder 边界参考（2026-11-07）

在独立隐藏设备的 CPU 环境，使用相同官方参考源码和 checkpoint 重取红/蓝图的语言计算边界。完整模型 **6973 个**加载张量逐项等于 checkpoint 转 FP32，固定实际 HCU 图文嵌入、437 个 prompt token 和相同 24 个输出历史，执行 461 token 的 FP32 eager causal forward；两图全部原始 logits 与此前固定输入参考逐位相同。采集所有 30 层的输入/注意力后归一化及最终归一化，前三层保存完整行及 attention/MLP/层边界，后续中间归一化只保存末 25 行。该参考没有新增视觉 wrapper 执行或 HCU 模型运行。

与正式桥已归档的 HCU 运行对齐时，分别比较 prefill 全部 **437 行**和 **24 个**实际输出位置，并区分融合归一化相加前的两个参数、未舍入 FP32 和及实际 BF16 更新残差，不能把原始第一个参数当作完整隐藏状态。两图首层图文输入逐位相同；零基第 1 层注意力后归一化输出，相对 FP32 的全 prefill 相对 L2 差为 **0.003780/0.003789**；该层 MoE 输出的不同输入流比较为 **0.050886/0.047346**。这些比较包含前序输入与 dtype 差异，尚不能唯一归因于该 MoE 内核或路由。

最终归一化隐藏状态的全 prefill 相对 L2 差为 **0.274193/0.270126**，而 24 个实际输出位置为 **0.020212/0.016893**，对应余弦 **0.999795760/0.999857318**；保留两种行范围，不用最终输出位置的局部接近替代所有中间行一致。CPU 完整固定历史 forward 与 HCU 增量请求具有不同 GEMM 形状，本轮观测只是定位证据，不授予任何新的数值通过。下一步用完全相同的实际 MoE 输入，分别固定路由及独立计算路由，拆分输入漂移、专家选择与局部计算的影响。

标准库独立读取原始张量，重算 **44 组**边界的最大差、RMS、相对 L2、余弦和 BF16 舍入计数，并将每组 CPU 端逐元素匹配到原始 hook 捕获，全部一致；没有导入 Torch 或重跑模型。初次读取器遗漏 tuple 递归，已保留失败并修正私有读取器，原数据不变。CPU 任务 **404.05 秒**正常结束，控制器和 worker 均已退出，社区源码、官方参考源码、checkpoint、adapter 快照与输入审计通过。CPU 报告附带的历史观测比较不作为正式桥的新验收；正式桥原主参考 V1 失败仍按前述证据保留。

完整边界归档 `external-evidence:vl784-decoder-boundaries-raw.json.gz`，SHA256 `35796cf6d23b2934595c532aea1d0ee2c9e96a6c1b824b43b04df62a447ca011`，**31 个文件**；独立复核、契约与资源审计补充归档 `external-evidence:vl792-decoder-verification-supplement.json.gz`，SHA256 `4bae8b493490004c3d8d6805c0231f6b1ce2654a33a00fd54b9b130a75692c50`，**22 个文件**。生产代码、BF16 默认、原 V1 阈值和整体验收未完成状态均保持原样。


### VL2 首个 MoE 的同输入与不同计算流对照（2026-11-07）

固定正式桥保存的实际 BF16 归一化输入转 FP32，红/蓝图各覆盖 prefill 437 行及后续 23 行，共 **920 行**。在隐藏 GPU 的独立 CPU 环境，从同 checkpoint 逐项加载并核对官方第 1 层 MoE 全部张量，执行官方 FP32 gate、routed experts 和 shared experts；另以实际专家 ID/权重固定路由，独立报告局部计算与完整模型不同输入流的比较。这只是单层诊断，没有改变生产路由或重跑 HCU 完整模型。

相同输入下，**920/920 行**专家集合与实际 HCU 选择一致，gate logits 最大差为 **8.10624e-6**。核对实际 runner 的缩放为 **2.0**：HCU TopK 捕获权重和约为 1，官方 FP32 gate 返回权重已乘 2，按专家 ID 对齐并还原该缩放后，权重最大差 **2.38419e-7**。首轮私有固定路由对照遗漏 runner 缩放，保留该不完整 counterfactual，不能把它误报为生产计算错误；以精确二次幂重组已保存的 FP32 路由结果后，固定路由与独立 gate 的 routed 输出最大差 **4.76838e-7**。实际总输出与这两种同输入参考的相对 L2 差分别约 **0.003477/0.003552**，未据此授予原整模型 V1 通过。

若分别沿 HCU BF16 流及官方 FP32 流传播，进入此 MoE 的 460 行输入相对 L2 差约 **0.003804/0.003820**；两图分别有 **19/460、22/460 行**专家集合不同，其中 prefill 为 **18/437、21/437 行**，24 个实际输出位置各 1 行。独立标量统计显示，这些专家集合变化行分别包含该层不同计算流输出平方误差的 **98.09%/97.71%**。这只是输入漂移、路由变化和输出误差的关联，尚未对路由作因果干预，也不能唯一指认前序 attention、融合残差或归一化存在错误。

标准库独立核对全部专家集合、缩放权重、固定路由重组、各输出指标和条件误差统计，通过；未导入 Torch，没有独立重放全部 expert GEMM。初次读取器未处理包含额外末行的连续 storage view，保留失败后按 stride/offset/shape 读取有效区域，原数据不变。两个 CPU 任务分别 **216.60/204.28 秒**正常终结，自有进程均退出，checkpoint、官方参考源码、adapter 快照和输入审计通过。下一步检查进入首个 MoE 前的误差传播，必要时用固定输入的路由干预分离专家选择的影响；原主 FP32 V1 失败、BF16 默认与原门槛继续保留。

完整同输入与路由传播归档 `external-evidence:vl795-fixed-input-moe-raw.json.gz`，SHA256 `4f72b4a5c4656d0dfb47b91ddeb161b49bdaca1bc1f16667577648d0ebfddaaf`，**26 个文件**；契约、缩放核对和独立验证补充归档 `external-evidence:vl797-moe-verification-supplement.json.gz`，SHA256 `7e28e61e072b8d1b10b5da8abaa5e4ae6f19c045e0e0e9e634b5137d61006a3e`，**26 个文件**。本轮新增定位证据，未新增生产修复，也不扩大为模型家族或配置矩阵验收。


### VL2 融合归一化语义与 MoE 路由放大效应（2026-11-07）

在已绑定的真实红/蓝图捕获上，固定第 0 层输入归一化以及第 1、2 层融合归一化的 460 行输入，复算多种社区语义。融合路径的残差写回逐位等于 `FP32(input+residual).bfloat16()`；与实际输出最接近的方案是 **FP32 求和用于方差、残差写回 BF16 后作为分子、再乘权重并写回 BF16**。第 1 层该方案相对 L2 为 **1.51e-5/8.73e-6**，第 2 层为 **6.52e-6/6.11e-6**；先舍入后算方差约 **0.00115/0.00127**，未舍入分子约 **0.00280/0.00280**，乘权重前转 BF16约 **0.00331/0.00332**。第 0 层非融合路径的 adapter 公式相对 L2 为 **4.00e-7/1.49e-5**。这支持当前 adapter 的融合归一化语义，不能据此新增生产改动或放宽精度门。

在 CPU FP32 单层因果对照中，使用实际 HCU 第 1 层 MoE 输入和完整 checkpoint，建立三种结果：A 保留实际输入与官方路由，B 保留实际输入但只替换为另一条计算流的专家集合并按实际输入重新计算权重，C 同时替换集合和另一条流的权重。相对另一条计算流的 FP32 MoE 输出，A 的相对 L2 为 **0.027653/0.025553**；B 降至 **0.002412/0.001776**，C 降至 **0.001817/0.001738**。红/蓝分别仅有 **19/460、22/460** 行专家集合不同，但对应的路由干预显著改变误差。该实验说明前序微小输入漂移被 MoE 离散选择放大；专家集合变化来自不同输入流，非并列 tie，不能用固定排序或改路由规则强制对齐，也不能唯一归因 attention、残差或某个 kernel。

两个 CPU 任务分别约 **214.81 秒**，控制器和 worker 均正常退出；checkpoint、源码、输入和候选字节审计通过。标准库独立契约复核通过，未导入 Torch 进行二次模型执行。完整诊断归档 `external-evidence:vl808-norm-route-raw.json.gz`，SHA256 `c5d83e4f38cbe2cd725a63b6f0d7e8c5f4ad3789e358b0de8a68b019a59fc69a`；本轮无生产代码变更，原整模型 FP32 V1 失败、BF16 默认和原阈值继续保留。下一步继续沿首个 MoE 之前的 attention/残差边界定位输入漂移。


### VL2 首个 MoE 前 layer 1 attention 边界对照（2026-11-07）

在相同真实 BF16/TP1/Triton 配置下，对红/蓝图各 24 个位置补采 `language_model.model.layers.1.self_attn` 的实际输入和输出；六个请求阶段共保存 **144** 组摘要，其中红、蓝、重复文件和 base64 四阶段的 **96** 组保存完整张量。模型生成、稳定排序桥、重复请求一致性、源码和输入审计均通过，真实任务 **383.11 秒**正常结束；没有修改社区源码或 adapter 生产代码。

使用此前同 checkpoint、同实际 HCU 图文嵌入和固定输出历史的 CPU FP32 官方 decoder 边界作为诊断参考，比较 layer 1 attention 的 **437 行 prefill + 23 行 decode，共 460 行**。红/蓝 attention 输入相对 L2 为 **0.296172%/0.294548%**，attention 输出相对 L2 为 **0.419594%/0.423582%**，这两组数值均为全部 460 行，旧报告字段 `prompt` 的命名不准确。单独 437 行 prefill 的输入为 **0.295681%/0.294045%**、输出为 **0.421564%/0.421548%**；24 个实际输出位置的 attention 输出为 **0.403794%/0.448351%**，最后一个 prompt 行为 **0.484630%/0.672258%**，后续 23 行为 **0.401428%/0.440971%**。输入误差在 attention 前已存在，attention 输出继续产生同量级差异，仍不能唯一归因于 MLA、RoPE、KV cache、残差或某个 HCU kernel。

CPU 参考边界实际已包含 layer 1 attention；一次重复挂钩的私有 CPU 探针在模型 forward 期间触发重复捕获断言，保留为工具错误证据，未计入成功任务，也没有重跑 GPU。初次独立检查仅验证完整归档 **252 个文件**的哈希和指标范围，并未独立重算数值。后续标准库读取原始 HCU/CPU 张量，核对连续位置 0–459，分别重算两图输入/输出的全部 460 行、prefill 437 行、实际输出 24 行、最后 prompt 行与 decode 23 行，共 **20 组**最大差、RMS、相对 L2、余弦及精确相等计数；旧报告对应数值均一致，全程未导入 Torch。

另核对两次运行的 checkpoint 清单及 HCU 前后字节审计，重新哈希 CPU 参考绑定的 **6 个输入文件**；两图各 24 个输出 token 历史一致，进入语言模型首层的 437 行实际嵌入摘要一致。这里按首层输入绑定，不能将 logits processor 返回的 `mm_input_embeds` 字段直接当作同一捕获边界。该比较跨越 CPU FP32 固定历史与 HCU BF16 增量路径，只作定位证据，不构成整模型精度验收。

完整归档 `external-evidence:vl851-attention-raw.json.gz`，SHA256 `7b1076328717e4613bab607cbdff70aae35d64df8f367e6854638b6108b1a1c0`，**368390554 字节、252 个文件**；数值独立重算与历史/输入绑定补充归档 `external-evidence:vl908-attention-verification-supplement.json.gz`，SHA256 `913e340aa8054b02344a48025073bc59d7da12e5c3523e8109d9c2b68a09efd3`，**19 个文件**。历史 `vl852` 报告的指标复核声明以本次明确范围为准。原整模型 FP32 V1 失败、BF16 默认和原阈值继续保留，下一步沿 layer 1 attention 的输入来源与 MLA/KV 边界继续定位。


### VL2 layer 1 attention 的同输入 FP32 对照（2026-11-07）

在隐藏设备的独立 CPU 进程中，仅实例化官方 `DeepseekV2Attention`，按解析后的配置确认 `q_lora_rank=None`、32 个 head、KV rank 512、非旋转/旋转 head dim 为 128/64。从原 checkpoint 严格加载本层全部 **5 个**权重张量并逐项比对；实际使用的配置、索引和单个权重分片前后 SHA256 均匹配已有绑定，官方源码与输入文件保持一致。本次没有重跑完整模型或 HCU 路径。

先使用此前保存的官方 FP32 输入、位置与 causal mask，单层输出与两图完整 **461 行**的原 decoder attention 输出逐位一致。再把前 460 行替换为实际 HCU 捕获输入转 FP32，保留被 causal mask 屏蔽的第 461 行以维持原 GEMM 形状，比较前 437 行 prefill 与后 23 行 decode。实际 HCU 输出相对同输入官方 FP32 输出，全部 460 行的相对 L2 为 **0.347483%/0.348261%**，24 个实际输出位置为 **0.345256%/0.373406%**；固定官方 attention，仅改变输入计算流产生的输出相对 L2 为 **0.223379%/0.222311%**。原不同计算流全部行比较仍为 **0.419594%/0.423582%**；这些误差指标的参考分母不同，不能直接相减作为贡献比例。

任务 **27.30 秒**正常结束，自有进程已退出。标准库独立读取匹配张量，重算三种比较、五个行范围、两张图共 **30 组**指标，并逐元素核对实际输入/输出及原 FP32 输入/输出与此前原始归档一致。该复核没有独立重实现 attention 的 GEMM、RoPE、归一化或 softmax；同输入对照仍跨越 BF16 增量路径与 FP32 完整 causal 路径，因此不能将剩余局部偏差直接认定为某个 kernel 错误，也不能替代整模型 V1 验收。

原始单层对照归档 `external-evidence:vl914-fixed-input-attention-raw.json.gz`，SHA256 `891d4bf36f3c040e6d35a9b94fdea6bba0991d72f5867c5ad2f56aabf962eb23`，**11 个文件、38417043 字节**；独立标量复核归档 `external-evidence:vl916-attention-scalar-verification.json.gz`，SHA256 `db14bbd97202945db46377402aa323f7722e5b631bea00f1acd9bbfcdf21cd9d`，**2 个文件**。下一步继续拆分实际 MLA 投影、RoPE 与缓存/attention 累加边界；原整模型 FP32 V1 失败、默认 BF16 和原阈值继续保留。


### VL2 实际 MLA 子模块与同输入组件对照（2026-11-07）

在相同 BF16/TP1/Triton eager 配置下，观察目标层的 Q/KV 投影、KV 归一化、RoPE、attention 上下文和输出投影；子模块观察仅在目标父层调用内生效，避免共享模块混入其他层。六个图像请求阶段各实际调用 **1 次 MHA prefill、23 次吸收式 MQA decode**，`kv_b_proj` 每阶段只调用 1 次，其余 Q/KV-a/归一化/RoPE/输出投影各 24 次。该绕过来自社区吸收式路径，不要求 decode 调用 `kv_b_proj`。六组图像请求及两次无标记控制的全部输出 ID、所选概率和 Top20 概率与此前运行逐位一致；源码、checkpoint、候选文件和输入审计通过，真实任务 **385.52 秒**正常结束。

隐藏设备的独立 CPU 进程逐项核对运行时全部 5 个 attention 权重与原 checkpoint，并验证 `w_kc/w_vc` 等于 KV-b 权重的原切分/转置；实际使用的配置、索引和权重分片前后哈希一致。对每个组件固定其实际输入，计算 FP32 参考及其 BF16 舍入对照；投影参考将 437 行 prefill 与 23 行 decode 拼接，GEMM 形状与实际单步 decode 不同。RoPE 将等价的交错/半分坐标排列对齐后比较，不能直接比较两种布局的原始数组。Q/KV-a/输出投影、KV 归一化和 RoPE 的全部 460 行相对 L2 约 **0.154%–0.167%**，参考舍入 BF16 后均降至 **0.00572% 以下**。decode 的 Q/V 吸收矩阵乘法相对其同输入 FP32 结果舍入 BF16 后的差异均在 **0.000572% 以下**；这些局部数值不替代模型精度门禁。

展开 MHA prefill 的实际上下文相对同输入 FP32 参考，相对 L2 为 **0.176568%/0.176508%**，参考仅在最终舍入 BF16 后仍为 **0.138653%/0.139481%**。吸收式 MQA 的 23 行 decode 为 **0.174813%/0.175231%**，最终舍入后为 **0.131971%/0.133819%**。decode 参考按捕获的当前压缩 KV 与旋转后 K 拼接语义历史，逐项确认当前 K/V 对应，但没有独立读取物理缓存，故不能排除缓存或唯一归因累加内核。社区 Triton 源码含中间概率转 V dtype 的路径；本轮未绑定实际 kernel 分块与精度参数，不能把上述残差全部解释为该舍入，也不能据此改默认计算。

CPU 对照 **24.71 秒**正常结束，两任务自有进程均退出。标准库独立重算 **116 组**指标、核对 BF16 舍入位模式，将实际投影/归一化/上下文重新绑定到原始子模块捕获，并抽样复算 **512 个 FP64 投影点积**；抽样 FP32 参考与 FP64 最大差 **4.01684e-7**，未独立重放全部 GEMM、RoPE 或 attention softmax。原始归档 `external-evidence:vl937-mla-components-raw.json.gz`，SHA256 `9cb94248797416b8ccad37134364c1e667be71c37667cbd650a614b5ab8097e4`，**1443 个文件、377983704 字节**；验证补充归档 `external-evidence:vl940-mla-components-verification.json.gz`，SHA256 `04afaeb5f9b8e60e14645604d9c1d84cfadf7383b40bc414955359a2c6401cb7`，**8 个文件**。本轮无生产代码改动，原整模型 FP32 V1 仍失败；下一步绑定实际 attention kernel 的中间精度、分块及物理缓存证据，继续区分社区计算语义与 adapter 偏差。


### VL2 实际 kernel 参数、物理缓存与 online softmax（2026-11-07）

在相同真实 BF16/TP1/Triton eager 配方中，仅于目标层的红/蓝 attention 调用期间启用只读 Python profile，记录实际 Triton launch 解析后的 constexpr、grid、选项和源文件摘要，并按 decode 入口本次使用的物理索引读取所需缓存行；不复制整个缓存，不改 launch hook 或社区文件。共绑定 **2 次 prefill、46 次 decode**：prefill 的 `BLOCK_M/BLOCK_N=64/64`、QK dim 为 128+64、V dim 为 128；decode 的 `BLOCK_N/MIN_BLOCK_KV=32/32`、QK dim 为 512+64、V dim 为 512，调度 split 数为 7、分配上限为 128，尾部空片按原分支跳过。记录到的内核均为社区原文件，普通 BF16 路径在 FP32 累加前将未归一化的中间概率转为 BF16；不是将最终归一化概率再转 BF16。

对两图全部 23 次 decode，将实际物理索引读出的压缩 K、RoPE K 与 V，逐行对齐到本请求从 437 行 prefill 起的捕获历史，**46 次、累计 20654 行全部逐位一致**。本轮直接补充了前次语义历史参考缺少的物理缓存证据，范围仅限这些请求和 page size 1 的 eager 路径，不外推为其他布局、前缀复用或 graph 验收。六组图像请求及两次无标记控制的完整输出 ID、所选概率和 Top20 概率仍与前次运行逐位一致；源码、checkpoint、候选文件和输入审计通过，真实任务 **397.32 秒**正常结束，自有控制器、模型进程及 CPU 进程均已退出。

CPU 对照按各次实际分块与分片重建 online max、未归一化概率、累加及 split 合并。保留 BF16 中间概率并在最后舍入 BF16，红/蓝 prefill 上下文相对 L2 为 **0.003716%/0.005390%**，23 行 decode 为 **0.006349%/0.002527%**；相同分块但使用 FP32 中间概率的诊断，最后舍入后仍为 prefill **0.138658%/0.139420%**、decode **0.131973%/0.133834%**。这支持此前局部残差主要来自参考没有包含社区中间概率舍入，而非这些已核对行的缓存历史不一致；剩余差异还包含 CPU/设备 FP32 归约顺序和指数近似，未据此宣称逐位相等或修改生产默认。

CPU 进程总耗时 **162.31 秒，包含等待真实采集审计完成**。标准库独立读取归档，重算 **16 组**指标与 BF16 舍入，复核全部物理历史、实际输出及内核源摘要，并按观察到的分块/分片抽样复算 **180 个 FP64 输出值**；抽样与 CPU FP32 重建最大差 **4.99860e-9**。该抽样未独立重放完整矩阵或设备归约，诊断容差不替代原整模型策略。原始归档 `external-evidence:vl967-physical-cache-online-raw.json.gz`，SHA256 `8fe30ae75a8f6c532549f3c4db8a89d06cd09c4c9b14fd476928c5d7bbb6eb68`，**716 个文件、233778913 字节**；验证补充归档 `external-evidence:vl969-online-verification-supplement.json.gz`，SHA256 `a92e573c9e88a4b635d37f281b0d9e4d54a7c4275dcdf47c6286b9ea23e31b8b`，**12 个文件**。原整模型 FP32 V1 失败、多模型及配置验收未完成状态保持；下一步核对现有 L1.5 矩阵是否覆盖这些实际 MLA 形状、分片与缓存语义，并继续定位整模型输入漂移和 MoE 放大效应。

### 真实 MLA 压缩缓存 decode 回归（2026-11-07）

核对同版社区测试后，grouped decode 的 576/512 形状已有覆盖，但主要使用独立连续 K/V、split4，并以 normal/grouped 两个设备 kernel 互相比对；MLA backend 另有布局和 Graph 用例，不能将其等同于已覆盖真实32头、共享576行距V视图和split7组合。新增 `tests/kernels/test_sglang_mla_decode.py`，直接调用社区公开 `decode_attention_fwd`，不更换内核或默认配置。三个参数节点覆盖长度 **33/437/460**、K576/V512共享缓存、随机物理索引、split7与空尾分片；未引用行和scratch填NaN，输出须有限。独立CPU FP32 dense softmax要求原始allclose `atol=rtol=1e-2`、relative L2 `<1e-2`、cosine `>0.9999`，并要求同值连续V与原视图输出逐位一致。Graph先重放原输入，再同地址改变Q、缓存、索引、长度和split数，仍执行全部数值断言。

HCU专项 **3 passed，pytest 3.81秒**；含启动和源码摘要复核的子进程阶段共 **40.46秒**。7502个社区Python文件前后摘要不变，另按社区commit `baa9befa381511cd4d56f83980951e5a45fffc6d` 审计全部10594个跟踪文件及额外源码通过；候选Python摘要匹配既有受测快照。归档 `external-evidence:vl976-mla-decode-regression.json.gz`，SHA256 `220f8d6ef2a914dbed4c38b4fbff133b27c46270b4999e4798aef74d4a88fffb`，**12个文件**；本地核对全部文件摘要、当前测试字节、三个JUnit节点的无失败/错误/跳过状态，以及受测社区文件与源码审计的绑定。该测试由现有kernels段自动收集，不改变58项家族清单或原整轮结论；尚未覆盖本专项的prefill、其他页大小或多卡，整模型FP32 V1及多模型验收仍未完成。

### VL2 固定层输入的 BF16 串联传播（2026-11-07）

复用既有红/蓝图真实采集，不新增HCU请求。在隐藏设备的独立CPU进程中，固定零基第1层实际归一化输入及进入attention前的残差，将投影、压缩KV归一化、RoPE、展开MHA prefill、吸收式MQA decode、输出投影和融合残差归一化串联。各次GEMM保留437行prefill与23次单行decode的形状，按实际kernel的分块、split7和中间概率BF16转换计算；所有对应存储边界显式舍入BF16，不再为每个组件重新喂入实际中间结果。五个attention权重及吸收矩阵与checkpoint/runtime逐项一致，实际使用配置、索引和分片前后哈希匹配；官方参考源码及候选Python摘要不变。

在全部460行上，红/蓝实际attention输出相对该串联参考的相对L2为 **0.067911%/0.062823%**，融合后归一化为 **0.045604%/0.040348%**。仅在CPU诊断中将online中间概率改为FP32、保留其他存储边界，attention偏差为 **0.166796%/0.164183%**，归一化为 **0.108004%/0.106932%**。这一对照支持纳入社区中间概率舍入更接近实际链路，但仍有CPU/设备归约、指数近似和存储边界附近的残差，不能称为完整设备内核重放或逐位复现。

使用同checkpoint的官方CPU gate，BF16中间概率参考相对实际归一化输入的专家集合在红/蓝分别 **2/460、4/460行**不同，全部位于prefill；最后prompt行与23次decode组成的24个输出位置均未变化。FP32中间概率诊断分别为 **2/460、8/460行**不同，亦无这些输出位置变化。实际归一化输入与既有真实MoE采集逐位一致，920行实际专家集合全部匹配。该实验进一步显示很小的串联残差即可改变离散专家选择，不能把其他输入流的路由强制回填到生产或据此授予整模型通过；本轮没有执行专家输出或其余decoder层。

CPU过程 **24.08秒**正常终结，自有进程均退出；私有读取器最初误解跨请求调用编号，两次未完成运行保留，修正后以新进程完整执行两图。标准库独立复算 **48组**指标和全部路由变化行，将实际输入、attention输出、归一化与专家集合绑定到先前不可变原始归档，并抽样计算 **48个FP64投影点积**、**72个FP64 gate点积**：投影舍入BF16全部匹配，gate与CPU FP32最大差 **1.46116e-7**。未独立重放所有attention/norm运算，抽样诊断界限不替代原V1。原始归档 `external-evidence:vl992-attention-chain-raw.json.gz`，SHA256 `ee347bd56bb8978edd088b3014029d448513e09d0b855ef6b19cc57d104a5a19`，**17个文件、108998210字节**；复核补充 `external-evidence:vl994-chain-verification-supplement.json.gz`，SHA256 `225eec48f3941840eb49f6a0604bce489de8a278877bfc4bf0cb815747b7e7a9`，**7个文件**。本轮仅补充诊断与方法文档，并回填已独立重哈希的历史路由归档摘要；生产默认及原整模型FP32 V1失败、多模型/配置验收未完成状态保持。

### VL2 从图文嵌入经过首层 Dense 的 BF16 传播（2026-11-07）

在相同真实 BF16/TP1/Triton eager 配方中，将只读观察范围前移至零基第0层 attention、Dense MLP 及下一层输入归一化，仍保留首个 MoE 边界作为控制。六组图像请求和两次无标记控制的各24个输出ID、所选概率及Top20概率与既有运行全部逐位相同；红/蓝图四个归一化模块的 **192组**输入、kwargs和输出摘要也相同。社区源码、checkpoint、候选文件与输入审计通过。实际模型任务 **390.00秒**正常结束，自有模型与控制器进程均退出；没有改写社区文件或干预计算结果。

独立隐藏设备的CPU参考从实际图文嵌入开始，串联输入归一化、attention、融合残差归一化、Dense gate/up、融合SiLU乘积、down投影及下一层输入归一化。保留437行prefill和23次单行decode的GEMM形状、实际attention分块与中间概率转换，以及各BF16存储边界；五个attention权重、吸收矩阵和Dense权重与checkpoint/runtime逐项相同。两图 **46次decode、累计20654行**物理缓存再次与语义历史逐位一致。固定实际gate/up输出时，融合SiLU乘积的 **11304960个元素**与CPU参考全部相同；这不等于前后两次GEMM或整个Dense链逐位相同。

全部460行上，红/蓝实际attention输出相对从嵌入串联的参考，相对L2为 **0.039378%/0.041033%**；经过Dense后的下一层归一化为 **0.053001%/0.057200%**。仅在CPU诊断中从实际attention输出重新锚定后，下一层归一化差降至 **0.005738%/0.007671%**。该对照说明前序边界残差会向后传播，未运行整模型替换，不能将差值解释为各组件的因果贡献。与同输入的既有完整FP32语言参考比较，实际下一层归一化相对L2为 **0.296172%/0.294548%**，串联BF16参考为 **0.296163%/0.294714%**；两条误差向量的cosine为 **0.983987/0.981154**。因此这一层与理想FP32的差异并非只出现在设备实现中，但尚不能据此豁免原整模型V1或唯一归因剩余偏差。

CPU计算与审计 **41.16秒**，含等待采集完成的进程总时长 **142.17秒**；标准库独立验证另用 **172.40秒**，不导入Torch，重算 **136组**指标、绑定实际边界和既有FP32原始张量、复核物理缓存，并抽样复算 **80个FP64值**（32个gate/up、24个down、24个SiLU乘积），舍入BF16后全部匹配。未独立重放全部attention/norm或整模型。原始归档 `external-evidence:vl1009-layer0-raw.json.gz`，SHA256 `9d31187fe8912f91e739e1be77019899bb595cbb32f273c0e628755770f70d28`，**1444个文件、1046097461字节**；独立验证补充 `external-evidence:vl1017-layer0-verification.json.gz`，SHA256 `9adb6bdc578fed490fa427cd5becc299e1fc0b96b9f10733ebdf660789c00651`，**12个文件**。本轮仅补充诊断证据，BF16默认值、原FP32 V1阈值和整体验收未完成状态保持。

### VL2 Dense 真实宽度的融合激活回归（2026-11-07）

既有 `tests/kernels/test_original_silu.py` 覆盖中间维度3072、2/47行，尚未包含本次真实Dense中间维度12288及437行prefill/单行decode。参数矩阵新增这两组形状，各跑FP16/BF16，保留原有四个节点，合计 **8 passed**。仍直接调用社区公开融合SiLU入口，以CPU FP64参考和最终存储前的FP32算术误差区间检验，提前舍入激活的负控须失败；原相对L2阈值不变。每个节点覆盖三组输入的eager和同地址变输入Graph，并核对输入未改写、输出边界哨兵及两模式逐位相同。此专项只覆盖融合激活，不补记为完整Dense计算链或模型通过。

HCU专项pytest **4.20秒**，含导入及源码摘要核对的子进程阶段 **42.14秒**。7502个社区Python文件前后摘要不变；全部10594个跟踪文件及额外源码按原社区commit审计通过，候选运行时Python摘要匹配既有受测快照。原始归档 `external-evidence:vl1021-silu-regression.json.gz`，SHA256 `f02b1b275ddaca03869261f5f6b3bfe131266056af98143c3435ff6c85ada548`，**10个文件、525090字节**；本地复核全部文件摘要、当前测试字节、8个JUnit节点及源码审计。测试由kernels段自动收集，不改变家族manifest或原整轮结论；原整模型FP32 V1、多模型与配置验收仍未完成。

### VL2 首层残差向首个 MoE 路由的串联传播（2026-11-07）

将已归档的第0层BF16串联参考产生的下一层归一化输入及更新残差，同时传入第1层attention与融合归一化，再执行同checkpoint的官方CPU gate。实际归一化输入、attention输出、MoE输入、专家ID与权重逐项绑定前次独立验证的真实边界；传播输入与残差逐项绑定首层原始张量。保留437行prefill、23次单行decode、实际分块/split及BF16中间概率转换，不重新锚定到实际投影或归一化值；本轮没有新增HCU请求、执行专家输出或改变生产路由。

全部460行上，红/蓝实际第1层attention输出相对串联参考的相对L2为 **0.228884%/0.225033%**，attention后归一化为 **0.159457%/0.159149%**。官方gate得到的专家集合分别有 **5/460、8/460行**不同，全部位于prefill；最后prompt行与23次decode组成的24个输出位置集合未变化。集合相同不证明输入、路由权重或最终模型输出相同；本轮也未比较传播后的完整专家链、后续decoder或整模型分布，不能将局部变化外推为整模型失败的唯一原因。

隐藏设备的CPU过程 **21.55秒**正常结束；不导入Torch的独立标准库验证 **38.93秒**，重算 **48组**指标和全部集合变化行，复核实际/传播边界，并抽样计算 **48个FP64 gate点积**，与CPU FP32最大差 **2.00293e-7**。输入、所用checkpoint资产与候选运行时摘要不变，自有进程均退出；未独立重放完整attention/norm计算。原始归档 `external-evidence:vl1028-two-layer-chain-raw.json.gz`，SHA256 `6063e6bb255a8706af50fcae8ecacbf6a085ff068df497230ff3382782b8d9ef`，**8个文件、124351083字节**；验证补充 `external-evidence:vl1029-chain-verification.json.gz`，SHA256 `fa5217563489d924f19f91e2d7a1b380a47238f9ce7748b214a401b410d40178`，**6个文件**。本地核对归档内文件摘要，并将验证器加载的全部张量绑定到本轮或此前已验证归档。原BF16默认、FP32 V1阈值和整体验收未完成状态保持。

### 社区 Dense MLP 完整模块回归（2026-11-07）

新增 `tests/kernels/test_original_dense_mlp.py`，直接执行未改写的社区 `DeepseekV2MLP`，补齐融合激活专项之外的gate/up投影→激活→down投影组合。使用公开replicated模式、hidden2560/intermediate12288和合成权重，覆盖FP16/BF16、单行decode及437行prefill，共 **4 passed**。独立CPU FP32参考在每个中间存储边界转回受测dtype；BF16要求relative L2 `<1e-2`、cosine `>0.9999`、按参考最大绝对值归一化的最大误差 `<2e-2`，FP16分别为 `<2e-3`、`>0.999998`、`<3e-3`。每个节点执行三组新输入，要求eager/Graph输出逐位一致、输出随输入变化、完整输入存储未变。

首次四个节点在计算断言前因缺少 `tp_group` 分配上下文失败；replicated模式虽不执行通信，社区RowParallelLinear仍读取该上下文。夹具改为公开 `get_parallel().override(tp_group=None)`，作用域结束恢复原状态，没有改生产代码或社区forward。复测pytest **19.21秒**，含导入及源码摘要检查的子进程阶段 **65.42秒**；7502个社区Python文件前后摘要相同，全部10594个跟踪文件及额外源码完整性审计通过。归档 `external-evidence:vl1044-dense-regression.json.gz`，SHA256 `166d4dfa7ab4a19a68133314bc00ab6f5f45aed3d2c9111fae208f54bed50276`，**17个文件、1085241字节**，保留初次失败；本地核对全部摘要、当前测试字节、最终4个无失败/错误/跳过JUnit节点及源码审计。该专项由kernels段收集，未新增家族manifest case；合成权重、单卡replicated模块不替代真实权重、残差、TP通信、原整模型V1或多模型验收。

### VL2 首个 MoE 的真实权重 BF16 专家链（2026-11-07）

在隐藏设备的独立CPU进程，加载并逐项核对第1层全部官方MoE权重，从已绑定的红/蓝图实际MoE输入及前两层串联参考输入，执行72个专家中的所选6路和两个共享专家。每个prefill批次按所选专家聚合有效行，23次decode各自执行；gate/up投影、融合SiLU乘积均存BF16，down投影的FP32累加值先乘归一化路由权重再存BF16，六个槽位以FP32顺序求和、乘routed scale 2后存BF16，最后与BF16共享分支相加。共享中间维度3072、单专家中间维度1536。该数学参考没有重放设备的padding tile、指令或小批次编译归约树，也没有改变真实模型执行。

对同一实际输出分别保留四条参考：固定实际输入及ID/权重；只换传播输入、保持ID/权重；传播输入加同一专家集合上的新分数归一化；传播输入加新专家集合/分数。新集合按偏置后分数排序再归一化，集合逐项等于先前官方CPU gate的结果；没有把事后排列或强制路由回填生产。全部460行上，红/蓝固定实际输入参考的MoE总输出相对L2为 **0.014309%/0.015327%**，只换输入为 **0.291715%/0.225335%**，固定集合重算分数为 **0.292130%/0.228895%**，重选专家后为 **1.717357%/1.837556%**。后者仍只有原来的5/460和8/460行更换集合，全部位于prefill；24个实际输出位置的总输出差为 **0.269161%/0.160865%**。这支持离散路由能放大前序残差，但这些局部干预、不同分母的范数不能相减为贡献比例，也没有证明最终模型概率失败的唯一原因。

CPU worker **225.73秒**，含启动的控制器总时长 **231.89秒**；完整checkpoint前后字节审计、官方参考源码、候选与输入摘要通过，自有进程终结。不导入Torch的标准库验证 **62.37秒**，重算 **120组**指标，复核全部原始MoE字段、传播输入和专家集合，并抽查 **81920个位置**的combine和共享相加输出及 **384个**FP64投影/激活/路由加权值。抽样BF16最大差为0 ULP，down的CPU FP32与FP64最大差 **1.48933e-8**；32条采样记录的权重行另从checkpoint重新读取并逐项绑定，所用分片前后摘要相同。抽样不替代全部设备GEMM、gate或剩余decoder验证。

原始归档 `external-evidence:vl1043-staged-moe-raw.json.gz`，SHA256 `bfa68eef6cfb20fee47134d72ca50345886001263591c9ea9914ced2b75d30ff`，**13个文件、277073444字节**；完整验证补充 `external-evidence:vl1053-moe-verification-complete.json.gz`，SHA256 `ef894e512a060c7b97aa9e3e68fb01fb3e1625a489d34942acc387cbadb3ecfd`，**17个文件、749389字节**。本地核对归档及文件摘要，将验证器加载的7个张量文件全部绑定到本轮或此前不可变归档。本轮未新增生产修复；BF16默认、原整模型FP32 V1失败、多模型及配置验收未完成状态保持。

### VL2 实际形状的 routed 专家链回归（2026-11-07）

现有小形状MoE链之外，新增 `tests/kernels/test_original_moe_experts.py`，直接调用社区 `fused_experts_impl`，覆盖72专家、hidden2560/intermediate1536、top6及BF16的单行decode/437行prefill，共 **2 passed，pytest 11.37秒**。独立CPU FP32参考在gate/up、融合SiLU乘积、乘路由权重后的down投影分别存BF16，六个槽位以FP32顺序求和、乘2再存BF16；要求relative L2 `<1e-2`、cosine `>0.9999`、按参考最大绝对值归一化的最大误差 `<2e-2`。三组新输入同时改变激活、专家ID和权重，eager与Graph输出逐位一致，完整激活存储、路由ID和权重均未改变。一个始终未选中的专家填NaN，输出须有限，覆盖空专家与padding隔离。

含导入及两轮完整源码审计的子进程阶段 **78.69秒**。7502个社区Python文件前后摘要相同，全部10594个跟踪文件及额外源码按原社区commit审计通过；受测adapter快照摘要不变。归档 `external-evidence:vl1060-moe-regression.json.gz`，SHA256 `17f030306447170d53c1d4169938cdf1fb2e4d6951bb94d8afb70de3195cf332`，**11个文件、537937字节**；本地核对全部文件摘要、当前测试字节、两个无失败/错误/跳过的JUnit节点及源码审计，并保留最初在pytest前因私有脚本缺少测试路径赋值而失败的脚本。测试由kernels段收集，不增加家族manifest计数；合成专家链不执行gate、共享分支、TP通信或模型生成，原整模型FP32 V1失败、多模型/配置验收未完成状态保持。

### VL2 固定实际图文输入的完整 CPU BF16 decoder 对照（2026-11-07）

在隐藏全部设备、未导入SGLang或宿主桥的独立CPU进程，使用官方参考源码与同一checkpoint加载完整BF16模型，逐项核对 **6973个张量**，全部等于checkpoint转BF16，且没有额外权重数值变化。固定实际HCU图文嵌入、437个prompt token及24个输出历史，执行461 token完整语言decoder；不重新运行视觉/projector，也不进行CPU自由生成。保留native BF16输出层及同一最终隐藏状态的FP32输出层，与此前逐位复现的同历史CPU FP32 logits比较。CPU架构、Torch构建、实际dtype和官方源码commit均保存在外部环境报告。

旧采集与正式稳定路由桥的HCU输出ID相同，概率却不同。因此原历史比较保留，另逐位核对正式桥两图的437行实际首层嵌入及24个输出历史，并用正式运行的所选概率、Top20类别重算三方指标；没有重复执行模型或用旧报告替代正式观测。以下红/蓝指标均为该固定输入诊断，不是主图像FP32参考的新验收。

CPU native BF16相对CPU FP32的最大Top20 logprob差为 **0.502712/0.458664**，Top20/tail JSD为 **0.000418280/0.000636687**；相同最终BF16隐藏状态改用FP32输出层后，Top20差仍为 **0.480889/0.506500**。24个输出位置最终归一化隐藏状态相对FP32的relative L2为 **2.013795%/2.481405%**。这说明独立CPU BF16执行也会产生超原Top20阈值的偏差，不能将所有BF16/FP32差异直接认定为adapter错误；CPU与设备的GEMM、归一化、MoE存储合同及完整prefill/增量decode形状仍有差别，不能据此定义正常误差界或豁免原门禁。

正式HCU相对同输入CPU FP32的Top20差为 **0.547583/0.525276**，相对CPU native BF16为 **0.437472/0.377966**，相对BF16隐藏状态加FP32输出层为 **0.462394/0.360500**。仍存在未解释的实现与计算流差异，不能由这几组最大值相减估计某个kernel的因果贡献。原主参考的Top20 **0.604931366**、JSD **0.002448766**及V1失败不变；本轮没有生产修复、默认dtype或精度阈值调整。

CPU子进程 **874.92秒**正常结束，社区源码、官方参考源码、checkpoint、候选和输入前后审计通过；自有控制器及worker均退出。不导入Torch的标准库独立复核 **33.40秒**，从原始logits重算 **144行**完整词表归一化、逐元素核对96行保存概率、复算10组比较及48个输出位置，并再次绑定正式桥实际输入与全部token历史。该复核没有二次运行decoder或重放全部GEMM；组件、固定历史与局部指标均不替代多模型和配置验收。

原始归档 `external-evidence:vl1063-dtype-diagnostic.json.gz`，SHA256 `73c2ec829fac5b4d8289bfd191ad8eb86b6d5840d972c83e08657592362a470c`，**36个文件、339513506字节**；脚本、加载审计及跨归档绑定补充 `external-evidence:vl1068-dtype-verification-supplement.json.gz`，SHA256 `27b4ecc8cdca1e7cabb42a6527c4300d97d2aeef8d9218e705a3d08f63a9cb5a`，**21个文件、62410字节**。本地核对原始归档全部文件摘要、提供脚本规范化LF后与实际执行字节的一致性，并重哈希四份新旧不可变归档，将报告输入及验证器加载张量的 **30个路径**全部绑定到归档记录；最初本地脚本因Windows/Linux换行字节不同而拒绝打包的过程另保留，未重跑数值任务。原整模型FP32 V1失败、BF16默认及多模型/配置验收未完成状态保持。

### VL2 保留中间存储边界的完整语言 decoder 数学对照（2026-11-07）

将此前已验证的前两层参考扩展到全部30层：固定正式HCU红/蓝图的实际图文嵌入和24个输出历史，以437行prefill和23次单行decode的形状，执行独立CPU FP32运算与显式BF16中间存储。分别保留融合残差归一化的分子/方差约定、MLA online softmax在P·V之前转换未归一化权重、gate/up及融合SiLU乘积存储、路由加权down存储、六槽位顺序求和后乘2，以及共享专家相加。全部 **6973个**加载张量逐项等于绑定checkpoint转FP32。此诊断不执行视觉/projector、自由生成、Graph或TP，也不是生产候选。

两图各五个首层边界与既有参考逐位相同；第二层attention/归一化、旧460行gate及完整专家链也逐位复现，共 **12项严格控制**。主链将gate改为437行加23次单行计算，首个MoE的专家集合不变，权重最大差为 **4.47035e-8/5.96046e-8**；全部58组逐层路由的第6/7名边界无精确并列。只有前两层的实际attention内核调度已采集，后28层依据同构几何沿用该数学调度，不能称为已观察所有层的实际dispatch，也未核对全部HCU运行时权重变换或模拟设备指令、padding tile及归约树。

正式HCU相对该参考的最大Top20 logprob差，BF16输出层为 **0.500000/0.315564**，同一隐藏状态的FP32输出层为 **0.558331/0.327695**。最终24个位置的参考归一化隐藏状态相对原固定输入FP32参考的relative L2为 **2.043972%/2.044510%**。保留实际形状和这些存储边界仍未消除实现与计算流差异，不能据此唯一归因某个算子、把最大值相减当作贡献比例，或以固定历史诊断替代原图像参考。

CPU计算及其权重核对 **512.73秒**，控制器含启动与前后源码审计 **543.25秒**，自有进程已终结。首次尝试在官方CPU模型加载阶段以SIGSEGV退出，尚未进入数值对照；保留失败原件，开启故障堆栈的独立重试完成，不能将该退出解释为adapter数值错误。不导入Torch的标准库独立复核 **30.54秒**，重算 **144行**完整词表归一化、核对96行保存概率及 **6205440个**输出层BF16舍入元素，复算6组指标、120个层边界和两组最终归一化指标，并绑定实际输入及路由不变量；没有独立重放30层计算或验证后续层调度假设。

原始与复核归档 `external-evidence:vl1085-complete-staged-decoder.json.gz`，SHA256 `c1ce77ccda5a77365a43a1fb3e8ccdad9208e43ef42112254b8344b810468e87`，**48个文件、113305878字节**。本地核对所有成员摘要及两次实际执行的worker/controller字节。原主参考Top20 **0.604931366**、JSD **0.002448766**与V1失败不变，多模型及配置验收仍未完成。

### VL2 全层实际归一化张量与串联参考（2026-11-07）

历史正式运行的后续层大多只留摘要，不能直接推算数值误差。本轮在同一受测候选、原样社区源码及checkpoint上，重新执行原六组图像请求及两次无标记控制，为红/蓝两图的全部30层输入/attention后归一化及最终归一化保存计算前输入、原残差和返回张量。六组请求的输出ID、所选logprob和全部Top20概率与原正式运行逐位相同；观察没有替换输出。采集子进程 **390.66秒**正常结束，前后社区源码、候选、权重和输入审计通过。

不导入Torch的独立比较读取 **2928次归一化调用**，按437行prefill的最后一行和23次decode对齐24个输出位置，对全部120个层边界及两组最终边界同时比较实际HCU、完整分段存储参考和原固定输入FP32参考。以分段参考为分母，红图输入归一化的relative L2由第3层 **0.316739%** 增至第4层 **1.624066%**；蓝图由第4层 **0.382958%** 增至第5层 **1.681913%**，层号从0开始。跳变发生于前一层MLP/残差到下一层输入归一化的区间，尚未隔离其路由、专家GEMM、残差或上游传播，不能直接判定某个kernel错误。

最终归一化处，实际HCU相对分段参考的relative L2为 **1.866843%/1.904160%**，相对固定输入FP32为 **2.021153%/1.689255%**；分段参考相对FP32仍为 **2.043972%/2.044510%**。三方距离不是可相减的因果贡献。上述比较只统计24个输出位置，不能宣称比较了全部prefill行；原始全行张量另有保留。后续层的实际attention调度、运行时权重与分段参考之间的合同差异仍待逐项核实，原主图像FP32 V1和多模型/配置验收仍未通过。

比较过程 **105.42秒**，结果与实际worker终结另经完成审计核对；一次性比较脚本的状态字典误继承父任务元数据，原记录与更正说明均保留，数值报告未改写。完整采集、比较及执行代码归档 `external-evidence:vl1087-hcu-boundaries.tar.gz`，SHA256 `2853625d6a31047d795732ad7a79799f64d74cadded99bab6705ee465f06a020`，**12997个文件、1017935085字节**。本地流式核对全部成员的大小与SHA256、实际执行代码、完整压缩文件摘要及gzip CRC。

### VL2 第3/4层的实际路由与误差传播分解（2026-11-07）

继续使用此前冻结的受测候选、原样社区SGLang和同一checkpoint，补采第1/3/4层MoE的计算前输入、实际gate输出、TopK专家ID/权重、routed/shared输出及完整模块输出。六组图像请求的输出ID、所选概率和全部Top20概率与原正式控制逐位一致；采集子进程 **422.11秒**，源码、候选、输入和checkpoint前后审计通过。运行时逐项核对三个MoE的全部 **663个checkpoint张量、444组参数**，包括gate、校正bias、72个专家及共享专家；实际W13为gate-then-up，未转置、交错或AITER重排，全部与checkpoint一致。这些结果仍绑定原受测快照，不追认为随后上游合入其他功能后的新组合验收。

独立CPU前六层参考保留全部 **460行**边界，全部6973个加载权重逐项核对；48组边界的24个已存输出位置及全部路由/权重与此前30层参考逐位相同。另对红/蓝两图的第3/4层分别固定实际输入/路由、替换传播输入、同集合重算权重，以及重新选择专家，共四条流；每条均分开处理437行prefill与23次单行decode。四组完整460行的MLP及下一层归一化严格复现原串联参考，实际MLP→下一层输入、post-norm残差连接均逐位对齐。前六层CPU子进程 **311.33秒**，分解对照子进程 **259.27秒**，结束审计通过，自有进程终结。

实际gate logits重新计算的专家集合、以及固定实际输入重算gate后的集合，四组均与实际观测一致；没有第6/7名精确并列，按实际槽位重算的权重最大差为 **5.96046e-8**。传播输入的专家集合则在红3/蓝3/红4/蓝4分别改变 **29/22/37/34行**，24个输出位置分别包含 **2/0/1/1行**，其余变化位于prefill。在这24个位置，目标红3/蓝4进入MoE前的输入相对L2已为 **0.421307%/0.484412%**；固定实际输入和实际路由时，完整MoE输出相对L2仅 **0.000293%/0.015030%**。因此需继续追查前序传播，不能只由专家切换认定TopK实现错误。

在24个输出位置，传播输入与残差但保留实际专家及原权重时，红3→4/蓝4→5下一层输入归一化相对L2为 **0.434970%/0.435323%**；同集合重算权重为 **0.438070%/0.434748%**；重新选择专家为 **1.624066%/1.681913%**。固定实际MLP输出和原残差重算下一层归一化，其差为 **0.000728%/0.000930%**。这些局部干预支持专家集合变化放大已有残差；没有替换整模型路由，也不能将上述距离相减作为因果比例。下一层norm运行时权重未单独导出审计，后续层attention调度仍依据前两层几何建模，不能称为完整设备指令重放或唯一根因。

不导入Torch的独立程序耗时 **336.44秒**，复算 **320组**指标、12组专家集合变化，逐元素核对 **18841600个**槽位求和及共享相加输出，并将本轮 **2928次**完整归一化字段摘要与此前采集对齐；另直接读取safetensors原始字节，复核全部663个MoE权重张量的444组摘要。该程序没有重放专家GEMM、sigmoid或全部norm数学。两次未完成的私有CPU对照也保留：一次在专家计算前因排序参数写法退出，一次因入选专家内部同分导致槽位顺序不同而触发严格控制；最终恢复原参考TopK顺序完成，未更改生产路由。

完整原始采样、执行代码、失败记录和独立复核归档 `external-evidence:vl1098-moe-transition-diagnosis.tar.gz`，SHA256 `527d63691dae37a224f8da957f7cce07e34fc1ace8fdd2f1863136ad66b113bb`，**15006个文件、1904805530字节**；本地流式核对成员大小/SHA256、执行脚本字节、完整压缩摘要和gzip CRC。复核器原文件与换行规范化后的执行文本摘要分别绑定，不将两者混为同一字节摘要。下一步补齐第2–4层实际MLA调度与计算边界，定位触发专家切换之前的数值残差。原BF16默认、主图像FP32 V1的Top20 **0.604931366**、JSD **0.002448766**及失败状态不变，多模型和配置验收仍未完成。

### VL2 第1–4层实际 MLA 调度、权重与计算边界（2026-11-07）

在同一冻结候选、原样社区源码和checkpoint上，补采零基第1–4层attention的计算前输入、返回输出与实际内核调度。红/蓝两图每层各1次437行prefill和23次单行decode，共 **192次attention调用**。逐次核对已解析参数、张量shape/stride/dtype、启动grid、内核源码摘要及编译元数据，均与既有首层观测一致；物理存储offset和JIT缓存态的options表示保留原件但不参与此签名比较。prefill的BLOCK_N为64，decode为32，实际KV splits均为7，均符合串联数学参考。本轮将第2–4层的调度假设补为实际观测，未覆盖第5–29层、Graph或其他配置，也未重放设备指令或GEMM归约树。

依据实际共享DeepSeek权重加载器的变换，逐项核对四层共 **20个checkpoint张量、28组运行时参数**：五类原始投影/归一化权重，以及从kv_b_proj拆分、转置并整理布局的w_kc/w_vc，全部匹配。另以不导入Torch的标准库程序直接读取safetensors字节，独立重建派生张量的逻辑排列，全部摘要一致。该范围只覆盖这些attention权重，不代替完整模型运行时权重审计。

六组图像请求的输出ID、所选概率和全部返回Top20概率与正式历史控制逐位相同，两次无标记红图控制也相同；**2928次**红/蓝全层归一化字段摘要与前次观测逐项相同。独立程序核对 **384处**attention与前后归一化的张量连接，并直接从原始张量计算全部460行、prefill437行、实际输出24个位置及decode23行的 **64组**误差指标。以相应串联参考为分母，24个输出位置的第2层attention输入relative L2为红/蓝 **0.237999%/0.213370%**，输出为 **0.447267%/0.414761%**；第3层输入为 **0.316739%/0.291949%**，输出为 **0.391349%/0.366172%**。差异在后续MoE路由放大前已存在，但这些是不同传播输入流的比较，不能将范数相减作为贡献比例，也不能据此认定attention内核错误。下一步固定实际输入，继续分解第2–4层MLA局部计算。

采样子进程执行 **390.48秒**，完成全部请求、checkpoint和源码审计后，在私有脚本末尾将JSON读回对象与内存对象直接比较时触发断言，原失败状态及日志保留。后续统一使用已保存JSON逐项验证控制结果，并补齐未执行的覆盖与权重汇总检查，没有重跑GPU模型。独立比较第一次仅因假定位置参数而未读取实际hidden_states关键字参数退出，原件同样保留；修正后的标准库任务 **58.64秒**正常终结，未导入Torch。上述工具失败不改写为原采样脚本成功，数值和覆盖结论以独立复核为准。

完整采样、两次比较、执行代码及失败原件归档 `external-evidence:vl1106-mla-dispatch-boundaries.tar.gz`，SHA256 `e06f9ac654c9e55e504497a6566449ba5d50d98ee6fbb827a50e8a0eb4e3b55b`，**10660个文件、62247498字节**。本地逐项核对归档成员大小/摘要、实际执行脚本字节及完整gzip摘要/CRC，自有进程已终结。本轮没有生产修复；证据仍绑定原冻结候选，不追认为最新上游组合验收。BF16默认、主图像FP32 V1的Top20 **0.604931366**、JSD **0.002448766**及失败状态不变，多模型和配置验收继续保持未完成。

### VL2 第1–4层 MLA 的固定实际输入对拍（2026-11-07）

在隐藏全部GPU、未导入SGLang的独立CPU进程，仅实例化官方第1–4层attention，逐项加载核对同一checkpoint的 **20个**所需张量。读取前次保存的hidden_states关键字参数及positions，严格核对每层红/蓝图的437行prefill与23次单行decode位置；使用各层实际观测的调度，保留原FP32运算及显式BF16中间存储。复用的七个数学函数AST与原串联参考一致，八组完整460行传播attention输出全部逐位复现历史参考，第1层两图的实际输入/输出另与旧捕获逐位对齐。没有重跑GPU模型或完整语言decoder。

分别保留实际输出对同实际输入参考、对传播输入参考、两条参考输出之间，以及两条输入之间的比较，共 **128组**指标，区分全部460行、prefill437行、实际输出24个位置和decode23行。在24个输出位置，同实际输入下attention输出relative L2：第2层红/蓝为 **0.183437%/0.104319%**，第3层为 **0.098280%/0.114712%**，第4层为 **0.129591%/0.125492%**。原不同输入流指标逐项复现；例如红图第4层仍为 **1.062815%**。固定输入对照缩小了定位范围，但没有更改服务输出；不同参考分母的距离不能相减作为误差贡献，也不能据这些局部值定义正常BF16误差界或宣布某个内核正确。仍需沿实际投影、归一化、RoPE及attention内部边界继续分解局部残差。

CPU参考子进程 **46.38秒**正常结束，包含前后社区源码审计的控制器 **61.10秒**；所需checkpoint资产前后字节摘要与原HCU绑定一致，官方参考源码、候选和输入审计通过。另一个不导入Torch的标准库进程 **96.07秒**，直接读取原始张量，重算全部128组指标，核对 **384处**attention输入/输出连接、**3680行**位置记录和 **16个**完整历史边界张量，复核七个数学函数AST及checkpoint资产绑定。该复核未二次执行attention GEMM或online softmax，不代表设备指令、tile或归约树重放；全部自有进程已终结。

原始参考、执行脚本、审计及独立复核归档 `external-evidence:vl1112-fixed-input-attention.tar.gz`，SHA256 `2d56ba4980299db3131bdca9c50a7b6d820be2efc437518bbb8a6d467310e144`，**32个文件、75921307字节**；本地逐项核对成员大小/摘要、执行脚本及完整gzip摘要/CRC，并将所用实际捕获绑定到前次不可变归档。本轮没有生产代码修改，BF16默认、原主图像FP32 V1失败及多模型/配置验收未完成状态均不变。

### VL2 第2层 MLA 内部边界与实际缓存对照（2026-11-07）

沿用冻结候选、原样社区源码及同一checkpoint，只在零基第2层attention补采八个子模块的计算前输入和输出，以及实际设备attention操作数、物理KV历史与分段中间结果。红/蓝图各覆盖437行prefill和23次单行decode，共 **48次attention调用**。未标记预热时记录实际绑定方法的文件、函数和源码摘要，调用跟踪确认prefill执行通用`forward_normal_prepare/core`、decode执行通用`forward_absorb_prepare/core`，没有将后端名称当作ROCm方法实际执行的证明。该层5个checkpoint张量、7组运行时权重及吸收布局全部核对一致，另保存实际FP32 RoPE缓存。

GPU采样子进程 **392.02秒**正常结束；六组图像请求的全部输出ID、所选概率和返回Top20概率逐位复现正式基线，两次无标记控制也一致。源码、checkpoint及候选前后审计通过。逐次将decode实际读取的历史K/V与此前保存的归一化latent及RoPE输出对齐，**46组历史、累计20654行**全部逐位匹配；prefill的Q/K/V、attention输出与后续投影输入的连接也逐位一致。该样本没有缓存写读错位的证据，不外推其他层、Graph或配置。

CPU对照保留实际prefill/单行decode形状，并逐项固定本段的实际输入。两图decode的KV归一化共 **23552个元素**全部匹配。RoPE使用官方CPU三角函数时仍有差异；改为读取实际FP32缓存、保持相同输入重算后，红图decode的Q/K共 **48576个元素**全部匹配，蓝图仅Q的一个元素不同，最大绝对差 **2.38419e-7**；prefill仍有少量残差。这里只隔离缓存与算术边界，不能据此认定缓存生成错误或将该变体设为生产修复。

直接使用实际物理Q/K/V、已观察的分块及有效分段计算online softmax，decode context的relative L2为红/蓝 **0.002835%/0.004209%**；投影和吸收BMM的同输入对照也仍有少量差异。完整attention串联参考则逐位复现前次固定输入结果，decode输出relative L2仍为 **0.183725%/0.104682%**。局部对拍与完整串联分别保留，不将这些不同分母的距离相减作为贡献比例，也不以局部小误差豁免整模型门禁。下一步应在完整attention串联内仅替换RoPE缓存来源，再分别锚定其他实际边界，检查这些局部差异如何传播。

配置分段数为7，但长度438–448时有7个有效分段，449–460时只有5个。首个私有CPU比较脚本因假定所有配置分段均有效而触发断言，失败状态及原脚本保留。源码及实际调度确认两个设备阶段都跳过空分段；后续比较只读取有效缓冲区，复用同一GPU捕获，没有重跑模型或改动原七个数学函数。修正后的CPU子进程 **34.77秒**正常结束，含前后源码审计的控制器 **49.02秒**。独立NumPy程序 **6.06秒**，未导入Torch或SGLang，从原始张量重算 **86组**指标、复核 **358处**张量连接，并核对全部 **2928条**红/蓝历史归一化记录、六组历史请求及两次控制；没有独立重放GEMM、RoPE、online softmax或设备指令。

完整采样、两次CPU比较、独立复核和失败原件归档 `external-evidence:vl1120-layer2-attention-components.tar.gz`，SHA256 `90d742b2ae30fafcd26ac11f9374a64b11c13f15316e7d6a05f01f548cda6d53`，**11175个文件、354640667字节**；本地逐项核对成员大小/摘要、实际执行脚本字节及完整gzip摘要/CRC，所需checkpoint资产另与原捕获绑定，全部自有进程已终结。本轮没有生产代码修改；原主图像FP32 V1的Top20 **0.604931366**、JSD **0.002448766**及失败状态保留，多模型和配置验收仍未完成。

### VL2 第2层完整 attention 的 RoPE 缓存来源隔离（2026-11-07）

复用上轮实际捕获，在隐藏全部GPU的CPU进程中，对红/蓝图各执行13个完整attention变体。原七个数学函数保持不变，默认变体逐位复现历史全部460行参考；另将437行prefill和23次单行decode分别保留。缓存来源变体只替换FP32 cos/sin值，仍调用同一官方`apply_rotary_pos_emb`并使用原坐标转换、投影、归一化、KV历史、有效分段、online softmax及BF16存储边界；Q/KV投影、latent和prefill展开值与原参考逐位相同。

实际缓存与官方CPU缓存的最大绝对差为 **3.05166e-5**。仅替换缓存后，完整attention输出的decode relative L2由红/蓝 **0.183725%/0.104682%**变为 **0.069265%/0.078787%**，prefill由 **0.061188%/0.077506%**变为 **0.044926%/0.059859%**。这隔离了缓存数值对该层完整计算的影响，尚未区分缓存构造中的频率、三角函数或算术顺序，也未证明缓存实现错误或它是整模型失败的唯一原因。

另分别锚定实际投影、归一化、旋转后操作数、attention上下文及输出投影输入，观察后续传播。全部attention操作数锚定到实际捕获后，decode输出relative L2仍为红/蓝 **0.024607%/0.030769%**；进一步固定实际输出投影输入后，剩余 **0.001513%/0.001789%**，且逐位复现上轮同输入输出投影的局部参考。这些锚定主动截断前序传播，是定位边界的诊断，不能当作独立算子正确性证明、将不同分母的距离相减估算贡献，或直接作为生产修复。

CPU子进程 **51.48秒**，含审计控制器 **65.90秒**正常结束；加载的5个checkpoint张量、原样社区源码、官方参考源码、候选及输入前后核对通过。独立NumPy程序 **7.35秒**，不导入Torch或SGLang，从原始保存张量重算 **216组**指标、核对 **44处**非平凡张量连接，并验证七个原数学函数及缓存来源变体的AST。另6处上下文/投影输入锚定只核对捕获值与构造记录及源码，没有分别保存变体中间值，不计为独立张量重放。首个复核脚本因AST归一化误处理嵌套坐标转换函数而失败，原件与终结状态保留；修正仅作用于复核器，未重跑数值计算或占用GPU。

完整CPU结果、原始张量、两次复核及实际执行脚本归档 `external-evidence:vl1127-cache-source-counterfactuals.tar.gz`，SHA256 `2658b8a112b897c59ef4b683add9363f9f69f746016c51c22f839b3abea2b3cf`，**31个文件、257535328字节**；本地逐项核对文件大小/摘要、脚本字节及完整gzip摘要/CRC。自有进程全部退出，所需checkpoint资产与原捕获绑定。本轮没有生产代码修改；BF16默认、原主图像FP32 V1的Top20 **0.604931366 > 0.15**、JSD **0.002448766 > 0.002**及失败状态保持，多模型和配置验收仍未完成。下一步应分别检查缓存构造来源与相同操作数下的残差，不把缓存来源对照中的局部改善提前认定为整模型修复。

### VL2 RoPE 缓存的社区构造来源与频率阶段（2026-11-07）

实际第2层使用社区普通`RotaryEmbedding`，checkpoint没有启用YaRN缩放。逐字提取原样社区源码的`_compute_inv_freq`与`_compute_cos_sin_cache`，仅以记录型Torch委托保存构造中间值，不导入SGLang或adapter宿主桥。在新检查为空闲的设备上执行原方法后，样本使用的前460行缓存 **29440个元素**及初始4096行 **262144个元素**均与原实际捕获逐位一致。社区源码已说明默认GPU缓存构造可能与CPU参考存在细微差异；该结果将这些实际缓存值绑定到社区构造语义，未发现所用前缀被adapter改变的证据。

隐藏GPU的独立CPU进程执行相同社区方法，并加载官方普通RoPE参考；两份CPU缓存的全部4352行逐位一致，前460行也逐位复现上一轮官方参考控制。CPU/GPU的32个逆频率有 **7个**相差 **1–2 ULP**。固定GPU逆频率后，CPU位置乘积与GPU记录的全部 **139264个**相位元素逐位一致；在相同GPU相位上改用CPU三角函数，所用前460行缓存的最大差仅 **5.96046e-8**，原CPU/GPU完整构造的对应最大差为 **3.05166e-5**。这把该样本缓存差异进一步分到逆频率生成阶段和较小的三角函数差异，尚未单独重放pow/倒数指令，也不以两个距离相减估算整模型贡献。

一次性GPU构造未复现原缓存最后256行，故另执行原社区`_ensure_cos_sin_cache_length`。初始4096行在GPU默认上下文构造，再在CPU默认上下文调用扩展：逆频率在CPU生成后复制到设备，新位置乘积及三角函数仍在GPU计算，所得4352行 **278528个元素**全部复现原缓存。扩展也在GPU默认上下文执行时，则逐位复现一次性GPU构造。探针采用源码默认128对齐，以4224作为达到4352行的重建触发值；历史模型的具体扩展触发调用没有被观察，不能宣称重放了其完整调用序列。末尾256行没有参与当前前460位置的attention比较。

两次有效GPU组件子进程各 **18.09秒**，CPU子进程 **7.32秒**，均未加载整模型或checkpoint权重。最初扩展探针误把对齐值断言为256，失败原件保留；读取源码默认128后仅修正私有探针参数，生产源码没有修改。全部任务的社区源码、候选与输入前后审计通过；官方参考源码及实际配置字节另行绑定。不导入Torch或SGLang的NumPy复核 **1.25秒**，从原始张量重算 **85组**指标，检查 **17处**张量连接、FP32位置乘积、7个逆频率差异及原方法AST；没有独立执行GPU三角函数、完整attention或整模型。

原始中间张量、有效与失败探针、实际执行脚本及源码归档 `external-evidence:vl1137-cache-construction-stages.tar.gz`，SHA256 `7011ae5f43f4e55da67af7936b5c9cbe2681ed09836de44d057360a2561d28a2`，**87个文件、28295382字节**；本地核对全部成员大小/摘要、执行脚本字节与gzip完整性，自有进程已退出。缓存来源替换对完整attention的影响仍以上一轮实测为准，本轮没有新增生产精度修复；BF16默认、原FP32 V1的Top20 **0.604931366 > 0.15**、JSD **0.002448766 > 0.002**及失败状态不变。下一步固定实际attention操作数，继续核查投影、BMM与attention累加的剩余数值合同；多模型和必要配置验收仍未完成。

### VL2 实际操作数的普通 HCU Torch 投影与 BMM 重放（2026-11-07）

使用零基第2层红/蓝图的实际输入、checkpoint权重、形状和布局，以普通 `F.linear` / `torch.bmm` 重放，不导入SGLang或adapter宿主桥。保留437行prefill及23次单行decode，分别执行q_proj、kv_a_proj_with_mqa、o_proj、prefill kv_b_proj、decode Q吸收与V吸收。两图六类主要输出共 **16639488个元素**，全部与历史实际捕获逐位一致；额外不使用`out=`的V吸收控制也逐位相同，未计入该主要元素数。

逐项核对五个原始BF16 checkpoint张量及七组运行时权重。吸收后的w_kc保留形状`[32,128,512]`、stride`[65536,1,128]`，w_vc保留`[32,512,128]`、stride`[65536,1,512]`；V吸收同时保留原`out=`视图。记录 **238项**操作元数据，该数不含额外的无`out=`控制调用，不能称为全部GPU操作数。Torch的`matmul_precision=highest`、`allow_tf32=False`及`allow_bf16_reduced_precision_reduction=True`均按原环境记录，未调整开关。

这个对照表明，相同操作数下此前相对CPU数学参考的投影/BMM残差，也能由普通HCU Torch产生，不需要adapter宿主桥参与；不能由逐位复现推断历史内部调用目标、设备库正确性、Graph行为或整模型门禁已经通过。它没有重放attention、路由或完整模型，不能将这些组件残差直接定为正常误差，也没有给出生产数值修复。

组件worker **48.83秒**，控制器含源码审计 **63.49秒**；源码、候选、输入及所需checkpoint资产前后审计通过。独立NumPy复核 **14.32秒**，不导入Torch或SGLang，从原始张量重算 **64组**指标，核对 **42处**实际输出/历史CPU参考/新GPU输出连接，并直接读取safetensors的五个BF16张量、验证七组权重值与原布局元数据；未独立执行GPU GEMM/BMM。自有控制器、worker及超时包装进程均已退出。

原始输出、实际执行脚本、运行时权重清单与独立复核归档 `external-evidence:vl1144-plain-hcu-projection-bmm.tar.gz`，SHA256 `3f142c7c54415cf08de2ea78fa5568e266a39d4a9169c4234f0099650f467b14`，**29个文件、89208595字节**；本地核对成员大小/摘要、执行脚本字节与gzip完整性。下一步使用既有实际操作数及调度，检查attention的分块softmax、部分结果和归约路径，排除未初始化的空KV分段。原BF16默认、主图像FP32 V1的Top20 **0.604931366 > 0.15**、JSD **0.002448766 > 0.002**及失败状态保持，多模型和必要配置验收仍未完成。

### VL2 原社区 attention 内核重放与两阶段分解（2026-11-07）

在不导入SGLang或adapter宿主桥的进程，提取原社区prefill、grouped decode stage1及stage2的三个JIT内核与`tanh`，逐项核对原AST后执行。保留红/蓝两图的实际Q/K/V、物理KV索引、shape/stride、启动grid及warps/stages；原编译元数据也逐项相同，但不据此宣称设备指令序列相同。两次437行prefill及46次单行decode的 **4333568个最终输出元素**全部逐位复现，另有 **4489216个有效FP32分段输出、8768个有效log-sum-exp**逐位一致，共 **8831552个**主要控制元素。

原KV池按实际物理索引重建，value保留key前512维的576行距视图，逐次读回核对历史；未用KV和部分结果槽位填NaN。配置split数仍为7，长度438–448的有效数为7，449–460为5；保持原128槽位存储与归约上限，只比较初始化的有效范围。重新执行原CPU数学helper，全部分段输出、LSE及最终上下文逐位复现历史参考；没有加载模型或checkpoint权重。

以下relative L2均以实际设备输出为分母。在固定实际分段结果与LSE后，仅将第二阶段改为CPU FP32合并，红/蓝各 **376832个**最终BF16元素中仅 **13/12个**不同，relative L2为 **0.000918899%/0.001317809%**。完整CPU分段计算加合并则有 **84/167个**不同，relative L2为 **0.002835093%/0.004209494%**。相同实际分段结果交给原GPU第二阶段时逐位复现；改为FP64 CPU合并仍有 **10/13个**不同，不能将提高合并精度直接当作修复。

保留原GPU第二阶段，只换CPU参考分段上下文而保留实际LSE，有 **67/151个**最终元素不同；只换LSE有 **26/30个**，两者都换有 **82/167个**。这些实际中间边界的干预支持继续追查第一阶段QK、online softmax及P·V，不能由这些数量或距离相减分配整模型因果比例，也未唯一判定某个运算错误。prefill本轮只完成原内核重放，尚未分解其内部数值路径。

组件worker **26.70秒**，含前后源码审计的控制器 **41.61秒**；社区源码、候选和输入审计通过。独立NumPy复核 **2.53秒**，不导入Torch或SGLang，重算 **26组**指标，核对 **208处**原始张量连接、46组有效分段字段、**278条**含干预的启动记录及 **6个**原内核/数学helper AST。它未二次执行GPU或CPU attention；GPU重建池与未用NaN区间仅由执行worker校验，没有另存完整读回供独立复核。全部自有进程已退出。

原始输出、两阶段干预、执行脚本、原社区内核源码及独立复核归档 `external-evidence:vl1150-attention-interstage-replay.tar.gz`，SHA256 `d2469c22d6a11e3b47940d5e5bb8270fa07e3720fa9a54619334df9b572227f5`，**31个文件、76453570字节**；本地核对全部成员大小/摘要、执行脚本字节及gzip完整性。下一步在保持分段输出/LSE逐位一致的前提下，观察第一阶段内部的打分、指数权重和累加边界。原BF16默认、主图像FP32 V1的Top20 **0.604931366 > 0.15**、JSD **0.002448766 > 0.002**及失败状态保持，本轮未重跑整模型，多模型和必要配置验收仍未完成。

### MLA 第一阶段实际操作数与 Graph 有效分段切换（2026-11-07）

仅在原社区 grouped decode stage1 的私有副本增加调试参数和存储，保存QK主点积、合并打分、缩放值、FP32指数及其BF16转换、重缩放与P·V累加结果、分母和最大值。移除这些新增部分后，原算术AST完全一致。红/蓝图各抽取decode第1/11/12/23步，长度438/448/449/460，保留原物理KV索引、stride及配置。八次调用的 **786432个有效分段元素、1536个LSE** 全部逐位复现历史捕获；另保存 **4386624个** 内部观察值。增加存储使共享内存从 **36864变为53248字节**，因此这是带输出控制的插桩观察，不能称为历史设备指令的直接观测。采样worker **37.72秒**、含源码审计的控制器 **52.49秒**正常终结。

不导入Torch或SGLang的NumPy分析按本段实际操作数建立数学锚点：QK缩放的114880个值、实际FP32指数转BF16的114880个值、累加器重缩放的1900544个值及最终分段除法的786432个值全部匹配。QK点积/相加、指数、P·V、分母归约和log仍有残差；例如实际打分的指数相对FP64计算后转FP32的最大绝对差为 **5.96046e-8**，P·V加实际重缩放累加器为红/蓝 **4.76837e-7/9.53674e-7**。这些锚点用于划分算术边界，不是正常误差界、指令模拟或整模型因果比例。

分析 **136组逐调用指标、34组汇总、64处历史连接及8组派生控制** 后，另一标准库程序用`math.fsum`复算全部170组指标，检查136处原始数据锚点、八组有效分段/LSE位模式和16处标量点积。红图有两处实际指数与独立`math.exp`再转FP32的结果相差 **5/2个FP32 ULP**，却跨过BF16舍入中点：第1步head28/history39转换为约 **5.00679e-5/4.98295e-5**；第11步head2/history326转换为 **0.045166015625/0.044921875**，后者CPU指数恰在中点并按偶数舍入。蓝图相同固定打分的BF16指数全部匹配。分析/复核耗时 **2.81/12.54秒**；两处敏感点没有被证明是整模型失败根因，未修改生产指数实现。

补强 `tests/kernels/test_sglang_mla_decode.py`：保留33/437/460三个参数节点、独立CPU dense-softmax及连续V对照，增加有效分段有限、128槽位scratch中的全部未用部分保持NaN、eager与Graph输出逐位一致的断言。在同地址改变Q、缓存和物理索引后，固定split7执行 **448→449→448**，有效分段随之 **7→5→7**。原数值容差不变，真实HCU运行 **3/3 PASS**，pytest **0.49秒**、含bootstrap子进程 **37.81秒**、含前后审计控制器 **52.78秒**；不增加L1.5家族manifest计数。

完整采样、原始中间值、分析/复核代码及实际受测文件归档 `external-evidence:vl1158-stage1-observation-graph-boundary.tar.gz`，SHA256 `0f11efaedc2a4b590fb086562f471263e4fbbb989371335488f39c92db0dbf33`，**49个文件、66398291字节**。本地核对全部成员大小/摘要、执行脚本、仓内测试的受测字节及gzip完整性，并将 **87条输入边** 与本轮及既有原始归档绑定。社区源码、冻结候选与输入前后审计通过，全部自有进程退出。本轮未重跑整模型或完成生产精度修复；BF16默认、原主图像FP32 V1的Top20 **0.604931366 > 0.15**、JSD **0.002448766 > 0.002**及失败状态不变。此有界阶段定位已完成，下一步回到多模型、必要配置和当前版本的验收缺口，不能用继续枚举局部ULP代替完整目标。
### fused-add RMSNorm 的社区存储语义修复（2026-11-07）

在当前合并候选上重新执行五家族 58 项矩阵，结果为 **57/58 PASS**：
Dense 9/10、MLA 17/17、MoE 14/14、Quantization 10/10、Mamba 7/7。
唯一失败是 `dense-fused-add-norm-graph` 的 4 个 Triton 参数节点；
对应 4 个 Torch 节点通过。失败差异集中在 residual 分子提前舍入与未舍入
FP32 和的区别，最大绝对差为 FP16 **0.00048828125**、BF16
**0.00390625**；不是测试框架或资源失败。历史两项 MLA 失败用例在本轮
均通过，附加压缩 MLA 专项为 **3/3 PASS**。

对照绑定的社区 `fused_add_rmsnorm.cuh` 后，adapter Triton 实现恢复社区
默认合同：方差使用未舍入 FP32 `input + residual`，归一化分子使用写回
激活 dtype 后的 residual。社区显式 `cast_x_before_out_mul=True` 分支仍
使用未舍入 FP32 和作为分子，并在乘权重前舍入归一化结果。生产修改只在
`sgl_kernel/layernorm.py`，没有修改社区 SGLang 文件；测试在
`tests/kernels/test_layernorm.py` 增加了 16 个直接算子参数节点和 8 个
社区入口参数节点，覆盖普通/Gemma、Torch/Triton、FP16/BF16、宽度 128/
12289、多 tile masked tail、eager/Graph 和三轮变输入。

修复候选的 HCU layernorm 文件为 **45/45 PASS**，Dense 家族为 **10/10
PASS**；CPU 组合为 **76 passed、1 skipped、38 deselected**，L0 GREEN。
当前修复版和社区源码前后完整性审计均通过，632 个候选文件摘要及输入摘要
保持不变，所有自有进程已退出。不可变归档
`external-evidence:adapter-norm1168-storage-contract.tar.gz` 的 SHA256
为 `46277fd348c6b1b217479cd65fac323801d1c07f6de3b35fe90fea90571096cf`，
本地独立核对 **1374** 个成员、成员大小/SHA256、gzip CRC、受测生产/测试
字节及报告中的 58/10 项严格判定。

该修复只关闭了当前版本的 fused-add RMSNorm 组件差距，尚未重新运行真实
模型，也没有改变默认 BF16、原 FP32 V1 阈值或旧 VL2 整模型失败结论：
Top20 最大差 **0.604931366 > 0.15**，Top20/tail JSD
**0.002448766 > 0.002**。多模型真实权重、必要配置和整模型精度验收继续
未完成。
