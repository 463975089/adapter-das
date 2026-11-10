# GLM-5.3 W4A8 适配记录

本分支配套 HYGON-AI/sglang-das v0.5.22.plugin（15c7204b7a3b），模型为 compressed-tensors。

## 2026-11-04 第一批

- 默认 CT 路由改为 native，保留激活量化；旧 w4a16 仅显式启用，不能用于本任务。未知配置拒绝。
- 修复 all-reduce 接线引用已删除的 _tar_installed，避免启动 NameError。custom AR 已除役，拒绝 AITER 时应直接保留 RCCL 回退。
- 新增 examples/glm5.3-w4a8/launch.sh，保留 CT 格式；当前为 TP8/EP8 eager 探路配方，尚未验收。
- 当前底座已内置 HCUCompressedTensorsW4A8Int8DynamicMoE；组合接线及两个 GEMM 的 A8 执行仍待运行核实。

第一批的本地 L0/Python AST、bash -n、git diff --check 已检查；没有新增单测。当时 GPU 验收、原生完整 profile、精度回归均未完成；后续进展见下文。

## 2026-11-04 原生配方与初始化检查

- 以用户提供的部署文档 revision 82 为配方来源，保留 `SGLANG_OPT_USE_TOPK_V2=0`，移除额外指定的 `--dsa-topk-backend torch`。记录实际 resolve 后的 backend，不通过强制覆盖替代文档配方。
- 多节点共用 NFS Triton 缓存时发生 `_per_token_quant_int8.amdgcn` 文件缺失；编译缓存默认改用容器本地路径，并允许环境变量显式覆盖。
- 在组合容器通过真实 CT 解析与 scheme 选择检查：抽查第 3/77 层和 MTP 专家得到 `HCUCompressedTensorsW4A8Int8DynamicMoE`，普通 Linear 得到 `CompressedTensorsW8A8Int8`，原始配置未被修改。该诊断使用无权重模块，仅证明分发，不证明实际 kernel 或模型精度。
- 指定 SGLang 底座的 11417 个跟踪文件逐项哈希未改变。本轮初始化检查时尚未生成原生完整 P/D profile，后续抓取结果见下节。

## 2026-11-04 原生 P/D profile 完成

原生镜像的 P=PP2/TP8/CP8/EP8、D=两节点 TP16/DP16/EP16、EAGLE 5/1/6，完成 4095 输入 token、64 输出 token 的完整请求抓取。32 份 trace 逐份 gzip/JSON 解析通过，取回本地后的压缩文件 SHA256 逐项一致，清单见 [原生 profile 记录](glm5.3-w4a8-native-profile.json)。

已命中 prefill HIPC contiguous W4A8、decode ASM masked W4A8、FlashMLA sparse attention、LightOp INT8 Paged MQA；这不代表组合环境的同输入数值已经对齐。ROCtracer 报告 duplicate flow 警告，kernel duration 总和不用于端到端性能验收。本次只覆盖一条约 4K 请求，其他长度、缓存及并发分支仍需补齐。

原生定向短请求在温度 0 下有复读/截断；文档的温度 1.0、top_p 0.95 下三条短请求正常结束且答案正确。这些是诊断请求，不是总体精度成绩。

初始 packed INT4 Triton 探路配方显式关闭 `SGLANG_ROCM_USE_AITER_MOE`，防止底座的通用 AITER 分支按 W4A16 标志解释 CT 特有的 per-channel W4A8 组合。实际 kernel 验证仍待完成。

## 2026-11-04 CUDA 特性探针导出与 index-K 配方

`platform_seam` 已将 NVIDIA CUDA 版本探针在 `utils.common` 中设为 False，但 `sglang.srt.utils` 的提前导出仍指向旧函数。KDA 模块被 plugin 注册链导入时，旧函数在 HIP Torch 的 `torch.version.cuda=None` 上执行 `split`，中断整个插件注册，导致后续 KV pool 钩子缺失。修复将这组探针与现有平台函数一起同步到包级导出；没有伪造 Torch CUDA 版本，也没有改变 KDA 算法。

实际容器检查：通过 `common` 和包级导入的 Ampere/Hopper/Blackwell/SM90/SM100 探针为同一函数且均返回 False。配合 plugin 的原生能力复用修复，完整注册检查通过，后续 KV pool 钩子已登记；证据为任务产物 `platform-probe-v2.log`、`plugin-bootstrap-v3.log`。启动配方补齐原生文档的 `SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1`。组合 v5 已完成八卡 CT 权重加载，约 47 GB/卡，但 warmup 尚失败于上述缓存接线，修复后的完整生成仍待验证。

## 2026-11-04 editable 安装后的冷缓存预热自锁

节点 21 的独立 RoPE 检查在 tvm-ffi 预热时停住：父进程持有构建 flock，子进程即使移除 `PYTHONPATH`，仍通过 editable 安装找到 `sitecustomize`，重新进入 adapter 并等待同一锁。实际进程链和内核等待点为父进程等待构建子进程、子进程 `locks_lock_inode_wait`，未进入 RoPE 数学检查。

构建子进程环境现同时移除 `PYTHONPATH`，并显式设置公开及传播的 host 为 `none`；预热窗口中显式传入的子进程 env 也按此处理。父进程环境保持原样。重跑后冷缓存成功生成 `libtorch_c_dlpack_addon_torch211-rocm.so`，主诊断继续运行；证据是 `rope-jit-check-v1-deadlock.log`、`rope-jit-check-v2.log`。不改动镜像中的 Torch/DTK 算子库，模型生成验证仍待完成。

镜像的 `/usr/local/bin/ninja` 是 Python 入口，JIT 编译调用它时再次执行已安装的 `sitecustomize`。启动配方通过已安装 `ninja.BIN_DIR` 将同一 wheel 的原生二进制置于 PATH 前部，保留编译命令、工具版本和父服务的 adapter 配置。正在运行的模型继续等待其原有 JIT 完成，未为此重启。

## 2026-11-04 FlashMLA code object 兼容诊断

组合 v8 已越过 RoPE/concat，在镜像内 FlashMLA sparse prefill 报 `no kernel image`。同卡、同输入的原生镜像调用通过；指定 DTK26.10 镜像内的 8/64 heads、带/不带 `topk_length` 均失败。设备代码中有 gfx936，不能据此推断可执行。当前节点的 HYHAL 为 `6.3.31-V1.5.7 20261029-124220`；同一 DTK 编译器生成的最小 HIP 核中，code object v5 可查找函数并执行，v6 的模块加载成功但函数查找返回 500。

在隔离目录用镜像版本对应的 FlashMLA 提交 `e7b26ad186028b07b9267e296eaf0acaa5cc7069`、CUTLASS 子模块 `33e45645599f371c2382e67920085498f5905955` 构建，显式设置 `FLASH_MLA_CODE_OBJECT_VERSION=5`，未修改算子源码、Torch 或 DTK。隔离包的 5 组 BF16 sparse prefill（tokens 1/6/17/256，heads 8/64）及更换输入后的 graph replay 均通过预先声明的 FP32 参考误差检查；最大绝对误差小于 0.0044。原安装包保持原样，完整模型生成和 P/D 仍待验证。这是当前节点/镜像组合的证据，不代表所有 v6 运行时都不支持该格式。

启动脚本现保留调用者的 `PYTHONPATH`，使此类已核实的隔离包可在同一镜像中验证；三份任务源码仍排在前面。复现构建时，在上述固定源码与子模块目录执行：

```bash
export HCU_ADAPTER_HOST=none FLASH_MLA_CODE_OBJECT_VERSION=5 FLASH_MLA_OPT=1 MAX_JOBS=8
python3 -m pip wheel "$FLASH_MLA_SOURCE" --no-deps --no-build-isolation -w "$WHEEL_DIR"
python3 -m zipfile -e "$FLASH_MLA_WHEEL" "$VENDOR_OVERRIDE_DIR"
```

仅对需要该验证的启动设置 `PYTHONPATH="$VENDOR_OVERRIDE_DIR"` 与 `FLASH_MLA_ROOT_DIR="$VENDOR_OVERRIDE_DIR/flash_mla/asm"`。先确认实际导入的 `flash_mla.cuda.__file__` 和哈希，再继续服务验收；解包尚未完成时启动的一条早期探针导入了原包，该条未作为隔离包验证。证据清单为任务产物 `flashmla-cov5-provenance.json`、`code-object-check.json`、`flashmla-cov5-validation-v1.json`，原始失败记录保留。

## 2026-11-04 单机真实请求与 INT8 专家指令确认

组合 v9 完成 TP8/EP8 加载及 warmup。与原生使用相同固定输入 ID 和文档采样参数的三条请求均正常 stop，最终答案为 `323`、`H₂O`、`18`。水分子式的 Unicode 下标使第一版仅匹配 ASCII 的检查失败；保留原响应和失败记录，按 NFKC 表示等价重新检查同一响应后通过，没有重新生成或调整数值误差阈值。本轮仍为 MTP/graph 关闭的单机诊断。

额外抓取一条 33 输入、2 输出 token 的完整诊断请求，8 个 rank 的 gzip/JSON 均解析通过，本地归档内压缩文件 SHA256 逐项一致。trace 共命中 2400 次 `fused_moe_kernel_gptq_awq` 和 9312 次 `_per_token_quant_int8`。对应 8 份实际 Triton 编译产物中，A 为 `i8` 指针，packed B 为 `i8` 字节存储，矩阵指令为 `v_mmac_i32_16x16x32_i8`，输出为 BF16。结合 CT 的 INT4 权重布局，确认当前专家计算保留 A8；不能根据旧配置文件名中的 `w4a16` 或 BF16 输出类型判断量化降级。

独立抽查真实第 3 层一个专家，DeepGEMM masked 的 gate/up 和 down 两次 INT8 GEMM 对 QDQ 参考的 NRMSE 分别为约 `5.9e-9`、`4.5e-5`。在完全相同的 BF16 输入和实际 gate/up 中间结果上，原生与候选的输入 INT8/scale、SiLU-mul 后 INT8/scale 四组张量在 17 个有效行内逐元素一致；先前 CPU 舍入参考的两次严格检查失败仍保留。这些局部证据不替代完整模型中间结果与 teacher-forced 精度对照。

版本、trace 哈希和验收范围见 [单机候选 profile 记录](glm5.3-w4a8-candidate-single-profile.json)。目标四节点第一次启动停在 RCCL 初始化：镜像的 verbs/provider 不能打开当前 SHCA，且缺少所选网络插件；原生匹配的 verbs/provider 和网络插件通过隔离路径接入后，23/24 两节点实际 SGLang PyNccl 的 FP32、BF16、INT32 归约及两次更换输入的 graph replay 均精确通过，日志确认使用 4 个 SHCA 和 RDMA 插件。完整 DeepEP、Mooncake KV 传输、MTP、目标 graph、有限矩阵及候选 32-rank profile 继续验证。

## 2026-11-04 Mooncake SHCA 与 GPU RDMA

镜像的通用 Mooncake 包按标准 verbs 头文件构建，与当前节点的 SHCA u17 ABI 不匹配；仅替换运行时 verbs/provider 后，仍在 completion vectors 检查失败。采用镜像对应的 Mooncake 提交 `6eb1b21a06cad9e83569815847655ab1673c4e78`，按其 SHCA 变体设置 `USE_HYGON=ON`、`USE_SHCA=ON`、`USE_FAKE_HIP_RPC=ON`，使用与可用 verbs/provider 配套的 SDK 头文件重新编译。1741 个跟踪源码文件逐项 SHA256 与固定提交的本地副本一致，没有修改 C++ 源码。

构建只下载并解包隔离的 JSONCPP/YAML 等依赖，没有全局安装驱动或更换 Torch/DTK。SDK 作为 system include，避免其旧枚举警告被目标自身的 `-Werror` 提升为错误。产物是 CPython 3.12 的本地 Linux wheel；没有宣称 manylinux 可移植性。任务容器中原通用包的 146 个文件先逐项校验备份，再卸载通用 distribution，并通过隔离前缀加载 SHCA wheel，保留上游对重复 distribution 和头文件 ABI 的检查。恢复时先移除隔离前缀的启动引用，再从对应节点备份恢复原文件。

23/24 两节点通过公共 `mooncake.engine.TransferEngine` 验证 GPU→GPU RDMA：40 KiB 和 4 MiB 非均匀字节模式、128 字节偏移，接收内容及前后保护区均逐元素精确一致，两进程退出码均为 0。初始化日志确认 SHCA RDMA transport，未设置 `MC_FORCE_TCP`。控制消息使用独立 TCP socket；第一版使用共享盘控制文件，在不同 NFS 服务端之间未及时可见而超时，该失败记录保留，不能作为数据传输失败或通过的判据。

固定版本、产物哈希和范围见 [通信验证记录](glm5.3-w4a8-network.json)。四节点 v2 已采用这组依赖继续启动；本节只证明两节点通信，尚不证明完整 DeepEP、KV 交接、MTP 或目标 graph 已通过。

## 2026-11-04 连续 W4A8 设备代码与 DeepEP u17 ABI

四节点 v2 的 32 个 scheduler 均越过通信初始化和权重加载。prefill 在实际 DeepEP normal 计算中报 `W4A8 WASP kernel launch failed`；decode 在 low-latency 初始化时由 RocSHMEM 的 `ibv_alloc_parent_domain` 返回 95 并终止。四个进程均已退出，未将退出码 137 直接归因为显存不足。

镜像 DeepGEMM 为提交 `b061ec70272a001fca622ea814d3ee163e24bb05`。独立真实专家检查中，默认 WASP 1006 启动失败，原 HIPC 1002 明确报 `no kernel image`。为隔离 1002，诊断构建保留该实现的 kernel body，排除 WASP include/分发并明确拒绝 mode 1006，再使用上游标准 Torch hipify。相同 HIP 源码和编译参数仅切换 code object 格式：COV5 完成 gate/up 与 down 两次 INT8 GEMM，17 个有效行对 BF16 QDQ 参考均逐元素一致；COV6 编译成功后仍报 `no kernel image`。这只验证独立 HIPC 1002 模块，尚未接入完整服务，也未验收 WASP 1006。

DeepEP 采用镜像对应提交 `a755fa7e14631af7e8a6a0d6ead217101d3bfdda` 的 `BUILD_SHCA=ON` 构建，匹配 u17 SDK、`U16_LID=OFF` 和 COV5。其 build.sh 显式 checkout RocSHMEM `b71ac40294bb10c5ffb11b94bed7d7d20529ab02`，不同于 gitlink 的 `1c22be1a9e83d6500db171b5902e3b0e3d649481`，按实际构建契约记录版本。465 个非版本元数据的跟踪文件与固定源码逐项一致；setup.py 按已声明逻辑重写 `deep_ep/version.py`，保留原始严格字节检查失败记录，不宣称全部文件未改变。

隔离 SHCA wheel 保留原 DeepEP 安装。23/24 两节点共 16 rank，以 hidden=6144/experts=384/topk=8 的合成通信参数分别完成 3-token BF16 和 17-token INT8 low-latency dispatch/combine。所有路由都跨节点，检查每个有效 token、FP32 INT8 scale 和加权 combine，16/16 rank 逐元素精确通过。这不包含完整模型、MTP 或 graph 验收。产物与验证范围见 [后续 ABI 诊断记录](glm5.3-w4a8-vendor-abi.json)；后续需将连续 GEMM 的已验证实现作为可审查的 ABI 适配接入，再继续完整 P/D、有限矩阵和 32-rank profile。

## 2026-11-04 可审查的 HIPC COV5 接入

新增独立构建器 `tools/build_deepgemm_w4a8_cov5.py` 和可选机制 `hcu_adapter.deepgemm_cov5`。前者固定实际编译的四份 vendor 源码/头文件摘要，只生成独立 HIPC1002 模块并原子发布；后者仅在模块显式出现在 Python 路径时接入，验证完整 wheel 版本、wrapper 字节及扩展 ABI 元数据。以私有 op facade 和模块局部默认值保留原函数对象、签名、校验与已缓存别名，不改动共享配置或 masked 等其他算子。新 catalog、registry、UI 快照和已有 `HCU_DISABLE_OPS` 快关同步登记，没有新增环境变量。

真实专家检查经过原公共 API，默认选择 HIPC1002，gate/up 和 down 保持 INT8/FP32 scale 并通过原 QDQ 误差检查。两次 GEMM 各做两次变输入 graph replay，沿用原误差门限通过；缓存别名、函数 code、共享配置和 masked op 身份检查通过，显式 WASP1006 拒绝。任务产物为 `w4a8-contiguous-provider-mode1002-v1.json`；该组件验证仍不替代完整服务验收。配方统一采用公开 `--cuda-graph-max-bs-decode 16`，等价于先前兼容别名的目标设置。


## 2026-11-04 两节点 prefill 就绪及实际库核对

四节点 v3 的 prefill（21/22，PP2、TP8、CP8、EP8、normal）完成真实 CT 权重加载和 warmup，主节点 `/server_info` 返回 200。逐个读取 16 个 scheduler 的进程映射并校验已映射文件，确认使用同一 HIPC COV5 provider、FlashMLA COV5、SHCA DeepEP、SHCA Mooncake、配套 verbs/provider 和 RCCL 网络插件；不能仅根据启动环境变量认定实际库已生效。组件哈希及范围补入 [ABI 记录](glm5.3-w4a8-vendor-abi.json)，原始映射为任务产物 `candidate-pd/runtime-maps-node21-v3.json`、`candidate-pd/runtime-maps-node22-v3.json`。

同次 decode 在目标 graph 捕获时被原生 W4A8 消费者的 INT8/FP32-scale 校验拒绝，退出码为 137。分发配置修复归 companion plugin PR #11，不通过删除校验、关闭 graph 或降级 W4A16 处理。此阶段尚未完成一次候选 P/D 请求，也未交付候选 32-rank profile。

## 2026-11-04 避免重复包装 DeepEP vendor API

候选 decode v4 仍在目标 graph 捕获时被 INT8/FP32-scale 校验拒绝。仅运行 bootstrap、不加载模型的检查确认，native 与 TBO 均已使用预期 plugin dispatcher，但其 `_DEEPEP_LL_DISPATCH_USES_QUANT_TYPE` 为 false：旧 adapter bridge 将原生 vendor API 包装为 `use_fp8` 签名，隐藏了 `quant_type`，该旧接口也不能表达 INT8。配方现显式设置已有 `HCU_DEEPEP_BRIDGE=0`，由指定底座及 plugin 的既有 ABI 分支接管；保留旧 bridge 供其原有宿主版本使用，没有再叠一层 wrapper。

同时纠正路径记录：本底座的 CT + DeepEP + DeepGEMM 选择 `HCUW4A8Int8DeepEPMoEMethod`，复用其专家实现，但 CT 配置、checkpoint 名称和加载格式不变；其 normal 与 low-latency 均声明 INT8。plugin 修正 DeepGEMM 选择下的 W4A8 vendor 分支，撤回首次尝试中多余的 setter 配置补写。原消费者校验和目标 EAGLE/graph 设置保留，新的完整服务复验仍待完成。

## 2026-11-04 FlashMLA 静态池索引接口复用

decode v5 已越过 INT8 契约并完成一组 graph 捕获，随后 MTP 初始化导入 `FlashMLABackend` 时因 shadow 缺少 `hcu_create_flashmla_kv_indices` 退出。镜像没有注册对应 `torch.ops.sgl_kernel` 算子，LightOp 也未提供此出口；底座已有同语义的 `create_flashmla_kv_indices_triton`。新增薄接口复用该实现，保留原参数名、显式 stride、可选 start、page64、有效位置写入及尾部不变的语义，空批次/零宽输出直接返回；没有新写索引算法或更换 attention kernel。catalog、runtime registry、UI 快照和档位矩阵同步登记为 candidate。

节点 23 经实际公开 API 检查 6 组输入：连续/带 stride 源与输出、有/无起点、长度 0/1/63/64/65/128/257/4097，以及空批次/零宽输出。所有有效页索引与整数参考精确一致，未写尾部和输出保护列保持原值；4 个非空组合各完成两次更换源表、请求索引和起点后的 graph replay，仍逐元素一致。记录为任务产物 `candidate-pd/flashmla-indices-check-v1.json` 和同名日志。未新增单测，完整 MTP 与 P/D 请求仍待复验。

## 2026-11-05 原生 HCU top-1 MTP 随机采样选择

v6 的 32 个 scheduler 均已就绪，decode 完成 target-verify、draft-decode、draft-extend graph 捕获及 16 个 attention-DP rank 的启动 warmup。首条固定 33-token、temperature1/top-p0.95 的真实 P/D 请求完成 prefill，随后 decode 进入依赖 FlashInfer 头文件的 JIT 采样并退出；此前 warmup 不能代表此随机采样路径已通过。

指定底座已有 `sample_mtp_target_ids`：HCU top-1 MTP 保留温度、top-k/top-p/min-p、请求 seed/positions，先采样 target IDs，再进行树匹配。CUDA 平台 facade 令 `eagle_utils._is_hip=False`，已有 sampled IDs 没有被后续 `_verify_uses_greedy` 条件采用。`eagle_hcu_sampling.py` 通过正式 HookRegistry 在真实 HCU、原生能力及签名均符合契约时，补齐 top-1 非 rejection 的选择条件。它先保留原 predicate 结果，非 top-1 和 rejection 请求继续使用原结果；没有新增采样算法或改写 `eagle_sample`、logits、概率及随机数。

实际 bootstrap 和 Hook 应用检查完成 9 个 selector 组合，覆盖 top-k1/2/未配置、greedy 和 rejection。已缓存的 helper 别名随 Hook 传播，公开 `eagle_sample` 函数对象及 code 未改，模块平台常量未改。结果为任务产物 `candidate-pd/eagle-policy-check-v1.json`；这只是实际接线/选择检查，真实请求复验和完整 profile 仍待完成。未新增单测，SGLang 跟踪源码未改。

## 2026-11-05 候选完整 P/D 诊断 profile 与精度边界

修正 HCU 采样分支和 attention TP/CP 校验同步后，四节点候选完成真实 P/D 请求及 EAGLE 统计，但固定短算术请求输出乱码/复读并在 512 token 截断，精度未通过。按既定优先级保留当前失败配置，抓取与原生完全相同的 4095 输入、64 输出固定请求，P/D 两侧均正常停止 profiler。32 份 trace 逐项 gzip/JSON 解析通过，每 rank 均有对应 W4A8 与 INT8 激活量化 kernel；本地 tar 中压缩文件的 SHA256/大小 32/32 一致。归档及逐 rank 哈希见[候选 P/D profile 记录](glm5.3-w4a8-candidate-pd-profile.json)。这证明完整请求执行和真实量化路径，不证明答案、完整算子归因或性能通过；decode sparse-attention/indexer 的 kernel 归因仍待对齐。

实际 config/index/tokenizer 元数据在四台候选容器和两台可用原生参考容器一致，模型为 78 层、256 个路由专家；此前 DeepEP 组件探针的 384 专家是合成通信参数，不能当作真实模型专家数。PP 两侧 trace 各含 39 层 attention，GEMM 调用数差异继续结合实际路由检查，不能据此判定少跑模型层。

原始数值响应显示 P 入口 HTTP 的 token/logprob 有时与 D 首 token 不同且含非有限值；直接拼接它们的两版探针失败记录保留，不用于数值通过判断。临时、隔离的末级 PP LogitsProcessor 观测对真实 33-token 请求记录到 final hidden states 和 raw logits 全零；这仅把异常限定到采样之前，首个清零边界仍待定位。CPU 拷贝会同步并改变时序，观测代码没有提交为正式修复。固定误差阈值未放宽，完整数值及缓存/chunk/并发/MTP 开关矩阵仍未通过。

## 2026-11-05 embedding 汇总与 Aiter CAR code object

固定 33-token 请求的 PP0 发送、PP1 接收张量在 8 个 TP rank 上逐字节一致，异常已在传输前出现。进一步观测发现，第 0 层归一化输入已经全零；在 embedding 边界增加同步快照后，加载的权重行和各 rank 本地 Triton 查表结果均与 checkpoint 精确一致，8 份本地输出的 CPU 求和也与完整 embedding 一致，但实际汇总输出在各 rank 间不同。同步观测改变了未写入缓冲区的旧内容，因此不能把观测后的非零结果视为修复。

镜像 Aiter 为 `0.1.8+dtk2610.torch2110.2611031115.ga1a2bc`。独立进程重放实际 `[40, 6144]` BF16 embedding 分片，并增加 rank 常量、倍增和负半值三组控制。8 个 rank 的 4 组 RCCL/PyNccl 结果全部精确；原 Aiter CAR 的 32 组结果全部失败，显式以 NaN 填充的输出缓冲区全部保持原值。该二进制的 AMDGPU ELF ABI 为 4（COV6）。

使用镜像内相同的 402 份源码/配置，通过原生 `AITER_JIT_DIR` 和 `AITER_CODE_OBJECT_VERSION=5` 构建隔离 CAR，未改动算子源码、Torch、DTK 或原安装文件。COV5 产物使上述 8-rank eager 检查和每后端两轮变输入 graph replay 全部精确通过。版本、哈希及验收范围写入 [ABI 记录](glm5.3-w4a8-vendor-abi.json)。这支持当前节点/镜像的 CAR code-object 兼容性归因，不外推到其他运行时或全部 Aiter 算子。

构建入口为 `tools/build_aiter_car_cov5.py`：固定输入版本和字节摘要，使用公开 JIT 接口，只重建 CAR，校验 COV5 设备代码后原子发布。其他模块通过符号链接复用镜像文件；来源变化拒绝，失败的构建目录保留供检查。未增加数学实现、运行时 hook 或单测。临时观测不进入正式修复，组件检查不代替完整 P/D 精度验收。

实际 CLI 在独立目录重建成功，其新产物也重新完成相同 8-rank eager/变输入 graph 检查，全部精确。移除观测后，prefill 16 个 rank 的实际映射均绑定已验证 COV5 CAR。完整 P/D 算术 cold/repeat 两次均答对并正常 stop；随后 32-token 水分子式请求输出无关代码、512 token 截断，检查按原规则停止，JSON 项未执行。结果为任务产物 `candidate-pd/smoke-v13/validation.json`；模型精度仍失败，继续固定前缀/rank 对照，不能把 CAR 局部修正视为全部根因已解决。

## 2026-11-05 六组固定前缀原生数值对照

移除观测的 P13/D8 与原生四节点服务完成相同六组固定 token 前缀对照。原生启动前，四节点的 3660 份冻结源码/库文件逐项一致。每次请求先清空 prefill cache，使用 greedy、输出 2 token、ignore-EOS，记录预先指定的 127 个 token 的 logprob 和 top20；误差阈值保持 RMSE≤0.1、最大绝对误差≤0.5，且两端 top1 均须出现在另一端 top5。

| 前缀 | 输入 token | Prefill RMSE / 最大误差 | 已对齐 Decode RMSE / 最大误差 |
|---|---:|---:|---:|
| arithmetic-prefix0 | 33 | 0.90439 / 2.28129 | 0.36922 / 1.37770 |
| arithmetic-prefix8 | 41 | 0.62857 / 1.87121 | 0.42803 / 1.11660 |
| formula-prefix0 | 32 | 13.36944 / 20.24423 | 首 token 不同，未比较 |
| json-add-prefix0 | 44 | 0.68710 / 2.00999 | 0.76215 / 3.00405 |
| 4k-prefix0 | 4095 | 1.35404 / 3.62945 | 0.95365 / 2.01560 |
| 4k-prefix8 | 4103 | 0.56670 / 1.82594 | 0.48873 / 1.78225 |

六组 prefill 和五组可对齐的 decode 均失败。formula-prefix0 的首 token 为候选 17351、原生 785，互相 top5 检查也失败，证明分歧在 prefill 首 token 分布已经出现；其余五组前两 token 相同，但分布误差仍不通过。未放宽阈值，也未比较已经分叉后的 decode 前缀。

本轮 COV5 的 P 入口 T0 已核对末级 PP 的序列化/重建源码，且 D 返回的重复 T0 top20 与 P 精确一致，127 个选定项均有限、完整。此前拒绝的 COV6 未初始化响应仍保留为失败证据，不套用到本轮有效捕获。[紧凑结果和请求哈希](glm5.3-w4a8-numerical.json)记录验收范围；全响应与比较脚本保留在同名任务产物中。下一步比较实际归一化等中间边界，定位首次数值偏离；目前没有把模型级问题归因于某个尚未对照的算子。


## 2026-11-05 首个异常边界：短请求第 0 层 attention

原生和候选各采集两组请求的 32 份实际 rank 中间状态，并按 CP 的有效 token 位置对齐。两端插桩后的输出 ID、127 个选定 logprob 和 top20 均与各自未插桩基线精确一致；CPU 同步观测仍不用于性能验收。原生实际走 CP v2，逻辑 32/33 token 在每个 rank 补齐至 8 行；候选走旧 CP，归一化边界保持全局 32/40 行。观察点设在实际 eager runner，覆盖绕过外层 CausalLM.forward 的 CP v2 model-body 调用。

两组请求的第 0 层 input-layernorm 输入、gamma 和输出，在所有有效位置及 8 个 rank 上均精确一致。首个偏离出现在该层 attention 出口：算术 33-token 的各 rank 最大 NRMSE 为 0.001667，水分子式 32-token 为 0.965085；对应残差仍精确一致。32-token 请求的 8 个 rank 均仅前 4 行非零，后 28 行全零，且 rank 间输出不同。其前 4 行与原生对应 CP 局部行也没有对齐，不能仅凭形状就认定补一个 gather 足够。

两组请求共享相同的前 10 个输入 token：原生这些位置的首层 attention 输出精确一致，候选对应位置 NRMSE 为 0.847931。异常已经出现在后续 dense MLP 和 routed MoE 之前，下一步只对照该 attention、投影与 CP 数据布局边界。[中间值证据](glm5.3-w4a8-first-attention.json)保留源响应一致性、逐 rank 行记录和原始文件哈希。临时 CP v2 对照仅切换 prefill 的既有开关以匹配原生实际路径，保留 CT W4/动态 INT8 和原 decode/MTP/graph 配置；其结果尚待验证，没有提前修改正式配方。


## 2026-11-05 显式对齐原生 CP v2 配方

同一生产源码和观察代码，仅将 prefill 的 `SGLANG_ENABLE_CP_V2` 设为 1；decode v9 的权重、MTP 和 graph 进程保持运行。两组实际调用均确认进入策略 CP v2，按原生方式将每个 rank 补齐至 8 行。水分子式 32-token 请求的首层 attention 最坏 rank NRMSE 从 0.965085 降至 0.001760，首两个 token 恢复为原生的 `[785, 1196]`；算术请求为 0.001899，首两个 token 仍与原生相同。该结果支持当前短请求异常与旧 CP 执行路径有关，不单独归因为某个 gather 或 kernel。

本 GLM CT prefill 配方显式设置该已有开关，保持 W4/动态 INT8 和原有算子库，其他模型及 decode 的默认值不变。该修正只验证了两组插桩短前缀，最终 logits 仍有差异；去插桩的六前缀数值阈值、完整答案及缓存/chunk/并发/MTP 矩阵继续验证，不能将首 token 恢复视为完整精度通过。


## 2026-11-05 去插桩 CP v2 六组复验与 dense0 排除

P16 使用提交 23685d8 的正式 CP v2 配方，移除观察模块与额外插件白名单，decode v9 进程保持不变。六组固定前缀均完成，两端前两个 token 全部相同、6 组 decode 前缀均可对齐，但 6 组 prefill 和 6 组 decode 的既定分布阈值仍全部失败。算术与水分子式的 prefill RMSE 分别为 0.696257、0.859662；完整逐组结果补入[数值记录](glm5.3-w4a8-numerical.json)。不能把修正短请求的部分行全零等同于全部精度恢复。

使用实际 post-attention norm 输入及 checkpoint 的首层 dense W8A8 权重，分别在两套镜像重放 M4/M8 的 gate/up（N=24576、K=6144）和 down（N=6144、K=12288）。两端 24 份共同捕获的激活、INT8 量化值、FP32 token scale 和 GEMM 输出逐元素一致；当前 ASM 与 LightOp 公共 GEMM 接口的完整输出一致，128 个输出通道的 CPU INT64 dot/scale 参考也精确通过。公共接口可能内部委派 LightOp，不能仅凭 triton_scaled_mm 名称声称实际执行 Triton kernel。

进一步用镜像里的 LightOp fused SiLU，在同一套当前镜像算子上，分别以原生和候选实测输入逐元素重建了各自的首层 dense 输出。因此该边界的差异已由上游输入带入，不修改其 GEMM 分发或激活算法。FP32 SiLU 中间值参考与原生 BF16 中间舍入不同，仅作为诊断对照，不替换生产语义。[组件证据与复现入口](glm5.3-w4a8-dense0-alignment.json)记录实际库版本及夹具哈希；下一步观察首层 attention 的投影、归一化、RoPE 与核心计算边界。

## 2026-11-05 首层 attention 投影与 RoPE 实测边界

两组固定前缀各捕获全部 16 个 prefill rank，并保留 PP0 的首层投影、归一化、RoPE、attention 和输出投影实际入参/输出。原生参考使用四节点同版本原镜像；四份 P/D 响应的 token IDs 和三类所请求 logprob 与未加观测的原生 v3 逐项一致。32 份状态及一份静态权重已下载归档，逐文件大小和 SHA256 全部核对。早期混合原生 P 与当前 D 的可选诊断因框架线协议版本不同而取消，没有产生请求状态，不作为精度证据。

首层 fused QKV 的 2624 列输出、Q/K 归一化输入与输出在全部有效 CP 行逐元素一致；静态 Q/K norm、Q-b/O 权重与 scale、吸收矩阵也逐元素一致。Q-b 输出仍有很小的 BF16 差异，最大 rank NRMSE 约 8.2e-6。RoPE 则在 Q/K/位置输入完全相同的条件下出现输出差异，Q 最大 rank NRMSE 约 0.00142，K 约 0.00170。Indexer 返回顺序不同，但每个有效行选出的索引集合完全相同，不能按索引元素位置差直接判为选错 token。

受控 CPU 复算以 checkpoint 的 theta=8000000、rotary_dim=64 生成缓存；BF16 cache 和 BF16 乘加逐步舍入能精确重建两组、八个 rank 的原生 RoPE Q/K 输出。该缓存是受控生成值，并非运行时直接捕获值；下一步检查真实分发和镜像内公共算子，不能据此直接宣称最终精度根因已修复。[边界证据](glm5.3-w4a8-attention-alignment.json)记录范围与原始产物入口。当前仍未通过六组数值合同，本轮没有改变生产量化、GEMM、激活或阈值。

GLM 配方现显式设置 plugin 的 `SGLANG_HCU_USE_LIGHTOP_ROPE=1`，选择镜像内公共 LightOp BF16 RoPE，并保留框架 FP32 cache。分发策略与 typed cache 接线位于 plugin，不在 adapter 增加模型专属算法。正式插件发现、构造和分发已验证：两组八 rank 的 32 项 Q/K 及两组更换输入后的 Graph 重放均逐元素匹配原生，FP32 cache 对象未变，关闭开关后保留原分支。LightOp 使用完整 head 分发；复用旧 sgl_kernel 的 8-head 分块会产生差异，失败探针记录保留。完整模型数值仍待复测。

## 2026-11-05 原生 RoPE 配方的完整六组复测

P18/D10 使用 adapter f29c5f7、plugin bebe2a1 的精确源码快照，四节点全部 930 个运行时跟踪资产及五个框架文件逐项一致。移除观测后完成六组固定输入、CT W4A8、P/D、MTP 5/1/6 和目标 decode Graph 的实际请求，原始捕获退出码为 0；两步输出 IDs 六组均匹配原生，但六组 prefill 和六组对齐 decode 仍未通过原有 RMSE≤0.1、最大绝对差≤0.5 的合同，比较退出码为 1。局部 RoPE 对齐不构成整体精度修复。

例如 arithmetic-prefix0 的 P/D logprob RMSE 分别为 0.8123/0.4400，formula-prefix0 为 0.8751/1.6891，4k-prefix8 为 0.4410/0.4611。完整结果保留在[数值记录](glm5.3-w4a8-numerical.json)，159934 字节的本地原始归档已核对大小和 SHA256。服务仍正常存活，没有因比较失败重启；下一步继续用相同实际 Q-b 输入/权重，对照当前 ASM、公共 LightOp 与原生输出，检查已经观察到的微小投影差异。完整有限矩阵和最终验收 profile 尚待完成。

Q-b 的两镜像对照确认公共 LightOp 输出均逐元素匹配原生，插件 ASM 快路径有四个元素不同；48 份共同量化整数、token scale 和公共输出全部一致。正式插件接口显式启用 `SGLANG_HCU_W8A8_USE_LIGHTOP=1` 后，16 组真实输入及两次更换输入的 Graph 均逐元素匹配原生，关闭开关后保持原候选结果。GLM 配方现同时启用公共 W8A8 与原生 BF16 RoPE，分发策略仍在 plugin；两项修正组合后的模型数值继续复测，不将组件通过当作完整模型通过。


## 2026-11-05 组合配方复测与残差 RMSNorm

P19/D11 同时启用原生 BF16 RoPE 和公共 W8A8，四节点的 933 份运行时资产及五份框架文件逐项一致。六组原始请求捕获退出 0，数值比较退出 1：六组 prefill 和六组对齐 decode 仍全部未通过原阈值，首两个输出 IDs 六组均匹配原生。算术 33-token 的 P/D RMSE 为 0.873326/0.315432，水分子式 32-token 为 1.514600/2.134720。111440 字节的原始归档已下载并核对 SHA256；不能用 token IDs 一致代替分布精度通过。

随后用两组、八个 CP rank 的真实 residual RMSNorm 输入对拍。原生实际入口为 `RMSNorm.forward_hip -> lightop.op.fused_add_rms_norm_opt`，能逐元素重建全部 32 份原生输出。adapter 原 Triton 内核第一遍用 FP32 残差和计算方差，同时写回 BF16 residual，第二遍却读回已舍入值作为归一化分子；它精确重建候选输出，但与同输入原生结果有 149008 个元素不同。

通用修复将 residual 写回延后到第二遍，归一化使用重新计算的 FP32 和；Torch 回退也保留 FP32 和。修正后的 Triton 仅剩 6 个不同元素，最坏 NRMSE 为 5.33e-5；镜像内现有 LightOp 分发全部 32 份输出精确匹配原生。因此 GLM 配方显式选择已有 `HCU_FUSED_ADD_RMS_IMPL=lightop`，通用默认仍为 Triton。三种实现各完成 64 份实际输入重放、4 次普通/Gemma 更换输入 Graph 重放，所有 Graph 输出及 residual 均与各自 eager 精确一致；BF16/FP16/FP32、hidden 6144/8193 的 FP32 参考误差另行记录，不声称不同归约顺序逐位相同。[组件记录](glm5.3-w4a8-fused-rms-alignment.json)包含环境、原始报告哈希和复现脚本。

这轮仅确认并修复算子边界；包含 RMSNorm 修正的六组模型复测、有限矩阵和最终验收 profile 尚待完成。


## 2026-11-05 RMSNorm 修正后的完整六组复测

P20/D12 使用 adapter 3199c22 与 plugin d31cc27 的冻结快照，显式选择 LightOp residual RMSNorm；其余量化、P/D、MTP 和 Graph 配置保持不变。四节点全部 938 份运行时/框架资产一致，旧进程均结束、32 卡各 2 MiB 后才启动新服务。六组原始捕获退出 0、比较退出 1：全部六组 prefill 和六组对齐 decode 仍失败，首两个 IDs 六组均匹配原生。

例如算术 33-token 的 P/D RMSE 为 0.657014/0.321125，水分子式 32-token 为 0.769123/0.962325，4095-token 为 1.623346/1.912347。原有阈值不变，组件修复不构成整模型验收。208876 字节的原始归档包含四节点源码核对、启动信息、复现脚本和六组响应，已下载并验证 SHA256。服务保持运行，下一步用原生实际首层 Q/K/索引输入对拍 sparse attention 核心，不在缺少证据时继续切换配置。


同输入 sparse prefill 控制已完成：将原生实测的 Q/K 两部分拼接为 576 维，以实际索引、scale=0.0625、d_v=512 调用两端公共 FlashMLA。两组、八个 CP rank 在 topk_length=None 和有效数量两种条件下，均逐元素重建原生实际输出；当前 COV5 二进制哈希与合格版本一致。该结果仅覆盖首层组件有效行，不外推为全部 attention/模型/Graph 通过。原始输出、报告和脚本的 13453710 字节归档已本地校验，见[attention 记录](glm5.3-w4a8-attention-alignment.json)。后续通过临时观察模块重新定位修正后调用链的首个差异，并先验证观察前后响应一致。


## 2026-11-05 修正后实测边界与原生 Top-K

P21 仅给原 P20 运行时增加临时观察，decode D12 进程保持不变。四份 P/D 响应的 IDs 和三类 logprob 与 P20 全部精确一致；32 份状态及一份静态权重已归档，本地逐文件 SHA256 一致。两组首层的 Q/K、Q-b 与 RoPE 已精确对齐，索引集合也一致，但原生索引为升序，候选按分数排序。attention 核心分别出现 10/12 个 BF16 元素差异；算术经 O 投影后精确一致，水分子式的 attention 出口仍有 105 个元素不同。

固定同一 Q/K，仅互换索引顺序，两套镜像各自都精确重现全部 16 组原生和候选 attention 输出，说明此处细小差异由顺序引入，不能扩大为全部模型误差的归因。追溯发现 adapter 的旧 model-env 规则以“融合算子无 HIP 实现”为前提，自动设置 `SGLANG_DSA_FUSE_TOPK=0`；当前 plugin 与镜像已提供原生 LightOp 路径。

GLM 配方因此显式设置 `SGLANG_DSA_FUSE_TOPK=1`，复用现有 plugin 分发，继续使用默认 sgl-kernel 和 TOPK_V2=0。正式插件入口在 16 组实际短行布局上精确匹配原生索引；分数是受控合成值，因这些窗口均短于 K，只验证原生“选择所有有效索引”的分支，不冒称捕获了原生 MQA logits。PAGED/RAGGED 各两次更换长度、offset/page-table 的 Graph 重放均与 eager 精确一致。初版探针在 capture 内创建 CPU 元数据失败，已改为预先分配后完成验证，不归因算子库。详见[Top-K 组件记录](glm5.3-w4a8-topk-alignment.json)。完整六组模型复测仍待执行，精度阈值不变。


## 2026-11-05 原生融合 Top-K 配方的六组复测

P22/D13 使用 adapter 388ab80、plugin d31cc27 的未加观察快照，四台实际主进程均确认 FUSE_TOPK=1、TOPK_V2=0、两项原生数学开关和 LightOp residual RMSNorm；量化、MTP、Graph 不变。全部 938 份运行时/框架资产逐项一致，旧句柄终止且 32 卡各 2 MiB 后才启动。六组捕获退出 0、比较退出 1：六组 prefill 和六组对齐 decode 仍全部未通过原阈值，首两个 IDs 六组均匹配原生。

算术 33-token 的 P/D RMSE 为 0.707731/0.451417，水分子式为 0.811182/1.252665，4095-token 为 1.202854/2.141170。209303 字节的本地原始归档已校验 SHA256，包含实际环境、启动、源码核对、请求和响应。该结果保留为失败证据，不能将原生索引组件通过等同整模型通过；下一步在修正后的调用链继续对拍中间边界。


## 2026-11-05 原生路由分支后的六前缀复测

P24/D14 使用 adapter `5a70ace` 与 plugin `9faa706` 的精确提交快照，启用已完成组件验证的 BF16 gate 与公共 LightOp 专家路由。四节点 946 份运行时/启动/框架资产及镜像、模型配置一致，旧进程已结束、32 卡均为 2 MiB 后才启动；MTP 5/1/6、目标 Graph、CT W4 与动态 INT8 保留。六组原始捕获退出 0，比较退出 1：全部六组 prefill 和六组对齐 decode 仍未通过原有门限，首两个输出 IDs 六组均匹配原生。

prefill RMSE 为 0.4409–1.5144，decode 为 0.2826–1.1189；这不支持将路由组件对齐视为全模型修复。182517 字节的原始归档 SHA256 为 `22c2e87e02dea06717e2716d6494d0b440c45b33d138bfdc8c78911d8308283c`，已取回并核对；完整指标见[数值记录](glm5.3-w4a8-numerical.json)，组件范围见[MoE 对齐记录](glm5.3-w4a8-moe-alignment.json)。服务在比较失败后仍正常运行，继续用同源码只读观测核对首个 MoE 边界；原生 gate 中间值尚未独立捕获，不能把来源分支重放写成实际捕获。最终有限矩阵和 profile 仍待验收。


## 2026-11-05 第 8 层 Q RMSNorm 的实际输入对齐

P25/P26 的只读观测保持四份 P/D 响应与未加观测的 P24 精确一致；原生同样四份响应精确匹配冻结基线，四台 3660 份原始源码与库文件在观测前后逐项一致。算术样例在第 8 层 `q_a_layernorm` 首次出现一个 BF16 元素差异，位置为 CP6 的逻辑 token 30，最大绝对差 0.0078125；其输入和权重均与原生一致，后续 Q-b 与 attention 差异从这里传播。另一组样例在该层全部观察到的字段一致。

原生 plain RMSNorm 使用 LightOp，adapter 原先默认使用 Triton。新增可选的公共 LightOp BF16 分发，保留调用方的输出 buffer、非连续布局和原地输出；其他 dtype、Gemma 和 CPU 路径保持原有实现，通用默认仍为 Triton。GLM 配方显式设置 `HCU_RMS_NORM_IMPL=lightop`，与 residual RMSNorm 的独立选择分别生效。

两套镜像的公共 LightOp 均在 32 份实际 Q/K norm 输入上逐元素重建原生结果。正式 adapter 入口的三组独立冷进程进一步确认：开启路径全部 32 份输出精确一致；关闭路径新旧版本 32 份输出精确一致；六类 dtype/Gemma/CPU 回退结果一致；五类调用方 buffer/布局检查及四次更换输入的 Graph 重放通过。211 份实际运行资产在验证后逐项一致。1840659 字节原始组件归档已本地校验，详见[plain RMSNorm 记录](glm5.3-w4a8-plain-rms-alignment.json)。本轮未新增单元测试，未运行完整测试套件。

这些证据只确认该组件差异和修复接入；修正后的整模型六组精度复测、其余算子完整审计和有限矩阵仍待完成。用户本次明确调整为先修复已定位精度问题、随后补全算子审计；精度通过后再整理 layer-split 与 VP 功能的独立 PR。


## 启动显存容量查询补充（2026-11-07）

实际 DTK26.10 启动日志仍出现 `get_device_memory_capacity → get_nvgpu_memory_capacity → nvidia-smi 缺失 → torch.cuda.mem_get_info`。此前主进程无额外 scheduler queue 的观测不能排除这条 PyTorch 初始化路径。禁止 lazy-init 的 main 组合探针已在同一调用处报错；这不是 INT8 KV 预算方法的错误。

本轮将 CUDA facade 下的容量查询接到已加载 HIP runtime 的 `hipDeviceTotalMem`，按可见 HIP ordinal 查询并取整 MiB 的最小值，保持上游容量选择语义。同步 common 和 utils 出口，不改框架文件，也不调用空闲显存接口。既有日志容量为每卡 65520 MiB；新接口数值、无初始化与新服务 context 状态待单独核验。当前正在运行的 GPQA 服务未修改，不能把该评测归为包含此新增修复。

## 官方 SGLang 0.5.22 上的运行时兼容（2026-11-10）

GLM-5.3 W4A8 的 PP2 + DSA LayerSplit 由 plugin 接入。adapter 为
`handle_model_specific_adjustments` 提供 `rewrite_resolution_function`。
它核对已经加载的函数与官方源码及元数据重写一致，再应用 plugin 的局部转换。
返回的新函数保留原 globals 和默认参数，官方源码文件保持原样。

平台适配先装好元数据、设备探针与平台语义，再导入
`runner_utils.shared_read_event`，避免其导入链提前初始化 GPU 0。
已导入的 DSA 模块继续使用现有的 `is_cuda` 别名修复。
JIT hash TopK 的设备标记使用 `kDLGPU` 匹配 Torch DLPack 设备枚举，
CPU 标量仍走原路径。

节点组合验证使用官方 `release/v0.5.22` 提交 `baa9befa` 和配套
plugin。该组合在 DTK26.10 上运行 P=PP2/TP8/CP8/EP8、D=TP16/DP16/EP16
及 EAGLE，完成三项精度评测。该结果对应当时的组合快照。
