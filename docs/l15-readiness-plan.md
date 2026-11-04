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

2026-11-04 用户再次确认 nmz21 可用于调试。三个节点的授权持续有效，
仍在每次作业前检查资源；本次实查 nmz21 容器正常、计算利用率为 0，
但 8 张卡显存占用 88%–89%，因此没有在该节点启动新 GPU 作业。
不能把计算利用率为 0 当作显存空闲，也不能沿用之前的空闲检查。

历史前置差距包括 `jit_*`、`dsv4_*` 的源码磁盘写回，
现已逐项迁移；Qwen2 当前组合通过严格源码完整性检查，其他模型与配置
仍须逐项验证，上述完整目标尚未完成。对象广播 `dist_bc` 已迁移为运行时只读 numpy buffer
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

nmz21 隐藏全部 GPU 的最终缓存/环境/启动结构及 CPU 算子组合回归
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

此前 nmz21 多次显存预检不足，确认有其他用户的 8 卡
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

nmz21 隐藏 GPU 的最终接口/环境/启动顺序回归 **25 passed、3 skipped**
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

nmz21 GPU 0 对原失败用例原样复测，连续/变长布局 × group32/128
**4 passed**（`iteration9-dsv4-ffi-gpu.log`）。每项由原社区 public API
实际编译，独立 FP32 SiLU×mul/分组 scale/FP8 参考对拍，三轮改变输入
幅度和专家有效 token 数后 replay；有效行检查 scale、逐元素误差和
cosine，未放宽首轮失败断言。隐藏 GPU 的 source/FFI 两臂回归
**25 passed**（`iteration9-jit-cpu.log`）；L0 GREEN（31 既有 warning）。
这是该设备上的算子证据，尚不代表真实 DSV4 checkpoint 或完整模型精度。

### 真实 Qwen3 与可重复 CPU FP32 参考探针

nmz21 物理 GPU 1 重新预检空闲显存比例 0.969343，
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
  --smoke-report results/l15-readiness/iteration9-qwen3-checkpoint.json \
  --out results/l15-readiness/iteration10-qwen3-reference.json
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

nmz21 隐藏 GPU 的绑定/参考/执行失败契约 **32 passed**
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
  --up-tree /home/lipenghui/lph-adapter/sglang-0.5.22-pristine-iter7 \
  --families checkpoint --checkpoint /model/qwen3/Qwen3-0.6B \
  --profile accuracy --modes eager,graph \
  --out results/l15-readiness/accuracy-generation.json
HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
  python3 -m tools.l15_checkpoint_reference \
  --smoke-report results/l15-readiness/accuracy-generation.json \
  --out results/l15-readiness/accuracy.json
```

首轮 `iteration12-{qwen2,qwen3}-accuracy-generation*` 失败保留：
Transformers 5.12 的聊天模板默认返回字典，导致请求传入字符串 key。
现在显式 `return_dict=False`，并在创建 Engine 前验证 token 列表和 case。
模拟该真实返回协议的回归已补入；最终 CPU 组合 **43 passed**
（`iteration13-accuracy-cpu.log`），两个真实 tokenizer 的独立预检也通过
（`iteration13-real-tokenizer-cases.log`）：短输入长度 Qwen2=31/30、
Qwen3=24/23，其他两批均为 512/2048。

nmz21 GPU 1/2 的 `iteration13-{qwen2,qwen3}-accuracy-generation.json`
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

nmz21 物理 GPU 3（预检空闲显存比例 0.969343），
`sglang_liph_adapter`，真实 `/model/Qwen/Qwen3.5-2B-Base`，
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

### 官方 FP32 输出层单变量诊断（2026-11-03）

`tools.l15_model_smoke --enable-fp32-lm-head` 显式转发社区公开 Engine
选项，默认不传入该选项。controller、worker、实际 server info 与 CPU
参考报告绑定同一配置，拒绝忽略选项或混用旧 worker 证据。此功能用于归因，
不改变 adapter 默认行为或 `bf16-cpu-fp32-top20-v1` 的任一阈值。

nmz21 同一容器、同一社区 commit，Qwen2/Qwen3 分别沿用物理 GPU 1/2；
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

证据位于 `results/l15-readiness/iteration16-{qwen2,qwen3}-fp32-head-`
前缀的 `generation.json/log` 和 `accuracy.json/log`；既有失败保留。
隐藏 GPU 的四模块 CPU 回归 **59 passed**，日志
`iteration16-final-cpu.log`；L0 GREEN（31 既有 warning）。
本次未重新执行全量 adapter/vendor 套件。

### 固定序列 CPU dtype 归因入口

```bash
HCU_ADAPTER_HOST=none VLLM_PLUGINS=none HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
  python3 -m tools.l15_checkpoint_dtype_probe \
    --smoke-report results/l15-readiness/iteration13-qwen2-accuracy-generation.json \
    --out results/l15-readiness/iteration17-qwen2-dtype.json --threads 4
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

2026-11-03，nmz21/nmz7/nmz28 均无满足预检条件的空闲 GPU，
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

资源复查发现 nmz7 GPU 已释放，选择物理 GPU4 并再次进行容器内预检：
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

nmz7 物理 GPU4，逐项预检空闲显存比例 0.969710，同一 checkpoint
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

nmz7 物理 GPU5 再次预检空闲显存比例 0.969710，使用社区公开
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

`iteration22-qwen3-device-probe.json` 在 nmz7 GPU4（预检空闲比例
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
- nmz7 CPU 是 Intel Xeon Platinum 8558P，而之前 nmz21 是
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

## 显式量化选项与参考边界（2026-11-03）

`tools.l15_model_smoke --quantization w8a8_int8` 转发社区 Engine 公开选项，
默认仍使用 checkpoint 自动检测。controller、worker、实际 server info 和
参考输入必须一致，忽略选项或陈旧 worker 报告会失败。CPU FP32 参考和
dtype 诊断同时拒绝 checkpoint 量化声明与显式在线量化，不能把未声明量化
的原始 config 当作在线量化执行的数值金标。

真实 `/model/hygon/Qwen3-30B-A3B-Channel-INT8-w8a8` 自动检测的失败保留于
`results/l15-readiness/iteration23-qwen3-moe-int8.json/log`：社区 CT MoE
构造拒绝并提示仅支持 NPU。社区另有 GPU `W8A8Int8MoEMethod`，且 cookbook
固定版本 `40adc4d30ec2640e3391cfd78759bd3598e95f71` 的同模型 BW1100
配方显式指定 `--quantization w8a8_int8`，因此先验证现有官方入口。
该入口是否完成真实加载、生成及数值验证须分别保留后续实测证据。

nmz21 隐藏 GPU 的工具回归 **114 passed**，报告
`results/l15-readiness/iteration25-quantization-option-cpu.log`。
此次新增节点授权包括 nmz7、nmz21、nmz28 的 `sglang_liph_adapter`，
代码共享目录相同；GPU 作业仍须逐次检查空闲资源。

## INT8 Linear 与量化模型验证（2026-11-03）

### INT8 Linear 公开路径的缺失符号修复

显式 `w8a8_int8` 的首次真实 30B 执行继续暴露
`W8A8Int8LinearMethod.apply -> NameError: int8_scaled_mm`，原始失败保留在
`results/l15-readiness/iteration26-qwen3-moe-int8.json` 及 jobs 日志。
此前仅给 CT scheme 补 GEMM global，未覆盖显式量化的另一个消费者。
`hcu_adapter.int8_linear` 复用现有 `sgl_kernel.gemm_int8.int8_scaled_mm`，
先验证两个消费者的方法和 globals 契约，再绑定各自模块符号。
原方法及已缓存别名保持原样；没有新增数学实现或源码写回。
catalog/UI/runtime registry 同步登记为 candidate，不据此作性能晋级。

nmz21 隐藏 GPU 的桥接/矩阵工具回归 **35 passed**，L0 **GREEN（31 warn）**，
见 `iteration31-int8-linear-cpu.log` 和 `iteration31-int8-linear-l0.log`。
设备回归与真实模型复测需以各自结果判断，不能由 CPU 绑定回归替代。

nmz21 GPU4 的既有 INT8 GEMM 与新增真实 W8A8 Linear 调用链共 **20 passed**，
后者包含 6 个参数化节点及三轮变输入 graph；见
`iteration31-int8-linear-gpu.log`。严格 MoE 家族另获 **7/7 case PASS**，
包括新 INT8 MoE，报告 `iteration30-moe-family.json` 的执行后社区源码
完整性通过。以上仍不代表真实 30B 模型或完整精度验收通过。

### 真实 30B Channel-INT8 模型启动与生成

`iteration31-qwen3-moe-int8.json` 在 nmz7 GPU4/BW1100 gfx938 上记录
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

nmz21 的原始 `iteration31-regression-full.log` 在收集期失败：
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
nmz21 后续资源检查显示显存已占用，GPU 作业在启动前拒绝；未终止其他进程。
改在已确认空闲的 nmz7 GPU5 验证全部相关文件，得到
**51 passed、1 skipped、1 deselected**，见
`iteration34-regression-fixes.log`。skip 是旧 aiter 布局才适用的 opt-in
场景，deselected 为本轮不运行的双卡 stress 项。
该证据是全量失败后的增量验证，不把原 56 failed/29 errors 改为整轮全绿。

最终增量检查：nmz7 GPU4 的量化家族严格门 **9/9 case PASS**，零
skip/xfail，执行后社区源码完整性通过，见
`iteration34-quantization-family.json`。最终 L0 **GREEN（31 warn）**，
见 `iteration34-final-l0.log`。当前保留的 vLLM 47 failed/29 errors
尚未修复；全模型 BF16/量化精度及其他既定覆盖缺口仍需继续迭代。

### BF16 逐层归因与增量解码复现（2026-11-04）

使用同一 nmz7 GPU4、Qwen3-0.6B checkpoint 字节和社区 commit
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
`results/l15-readiness/iteration37-{hf,sglang}-layers.log`、
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

nmz7 GPU4：整个计算链文件 **25 passed**，严格 Mamba 家族
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

nmz7 GPU5：完整 FP8 GEMM 文件 **17 passed**，接入后的严格量化家族
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

随后在 nmz7 GPU4 暴露单卡、真实社区与 DTK 依赖下分别启动干净
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
  --smoke-report results/l15-readiness/iteration45-ctfp8-generation.json \
  --quantized-reference ct-fp8-token-channel \
  --out results/l15-readiness/iteration47-ctfp8-bfloat16-accuracy.json
```

参考通过 Transformers 公开加载接口和 compressed-tensors 的
`run_compressed=False` 进行 CPU FP32 eager 执行。完整加载信息必须无缺失、
多余、形状不符或错误；逐个核对全部原始 tensor，量化权重必须精确等于
原始 FP8 值乘 channel scale，其余 tensor 也须精确匹配。实际 Linear 名称、
数量和量化参数必须完整匹配；Python profiler 只观察官方执行，不替换计算，
每次完整 forward 必须逐层观察到真实输入量化与 fake-quantize 调用。
仅反量化权重而移除激活 QDQ 不是此 checkpoint 的参考。

nmz7 GPU4、BW1100/gfx938、Torch 2.11.0/HIP 6.3.26113，
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

nmz7 GPU4 执行 `pytest tests/kernels tests/adapter tests/vendor -k "not stress"`：
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

nmz7 GPU5 空闲显存比例 **0.9697097** 时，完成一次完整的 32-case
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

### DeepSeek-V4.1-Flash-Channel-INT4 真实启动闭环（2026-11-04）

在 nmz7 的 `sglang_liph_adapter` 容器中，以 adapter
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
`results/l15-readiness/iteration51-dsv41-int4.json`，原始服务日志和请求
结果见同名目录。下一步先在 nmz21 做离线 EP4 dispatcher 复现，再修改
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

本功能提交为 `a7db6fc`，CPU 路由/契约回归为 **9 passed**。nmz28
容器中的社区源码目录仍为
`baa9befa381511cd4d56f83980951e5a45fffc6d`，未发生文件改写；显式以
社区 0.5.22 pristine Python 路径加载后，调用安装函数可观察到
`DeepEPMoE.run_moe_core._hcu_w4a16_deepep=True`。这只证明 seam 安装和
布局契约，尚未替代真实 EP4 权重加载、DeepEP 通信、masked GEMM 数值和
生成验证。

nmz28 当前 8 张卡仍由既有服务占用，显存约 94%，因此本轮没有启动新的
GPU 作业。下一步是在 nmz28 释放后，用真实
`DeepSeek-V4.1-Flash-Channel-INT4-w4a8` 做 TP4/EP4 启动、graph/eager
生成和与 EP1 的数值/语义对拍；若该 masked W4A16 ABI 与实际 DeepEP
buffer 形状不一致，先补离线形状/数值回归再修 adapter。
