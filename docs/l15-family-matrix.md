# L1.5 Dense / MLA / MoE / Quantization / Mamba 代表矩阵

该矩阵用于回答“社区 sglang + adapter 是否能在不加载模型权重、不启动服务的条件下，走通五类主要模型行为”。它不是性能测试，也不替代真实模型 L2。

## 当前清单与最近完整矩阵结果（2026-11-06）

当前清单扩为 **58 case**：Dense 10、MLA 17、MoE 14、
Quantization 10、Mamba 7。新增四项路由并列分数契约，覆盖
softmax、sigmoid、sqrtsoftplus、分组选择、共享专家及变输入 graph
replay；其中含 CPU 与 HCU 参数节点。

新清单首次五家族整轮结果为 **56/58 PASS，2 FAIL**：
Dense 10/10、MLA 15/17、MoE 14/14、Quantization 10/10、
Mamba 7/7。207 个 pytest 主节点及其 setup/call/teardown
报告通过，但 58 个 subtests 中 **55 PASS、3 FAIL**；
严格 runner 因此返回非零，不能将主节点计数称为整轮通过。
两个失败 case 为：

- `mla-dsv4-q-rope-large-prefill`：65536 行的两个 dtype 子测试
  分配显存失败，日志显示剩余显存不足；其余四个子测试通过。
- `mla-dsv41-low-ratio-compress`：ratio-1 子测试的 latent 数值
  不匹配，ratio-2 通过；不能仅凭同轮存在 OOM 将数值失败归为资源问题。

整轮社区源码前后审计与 603 个受测文件摘要核对通过；
完整报告、逐子测试结果、日志、受测源码及审计归档为
`external-evidence:matrix340-complete-raw.json.gz`，
SHA256 `2a743c57de0be6329f1cd5026ad0be6accc70296722705df2cde460e997fc9a2`。
本地逐一核对 674 个归档文件摘要，并按原 exact nodeid 重算严格
case 判定。L0 与同快照 CPU 算子/清单组合回归通过；
没有在本轮重跑全部 adapter 测试。

另一个空闲设备上的两项 exact case 复测均通过，包括原 65536 行
形状和全部 **8 个 subtests**。额外采样保留实际 compressor 输入与
kernel 返回 latent，ratio-1/2 原参考断言均通过；ratio-1 的实际
kernel 输入与重算 wkv 相同，index store 未改变返回 latent。
这些结果尚未解释原数值失败，不据此修改生产代码、放宽断言或将
原 56/58 改成通过。复测、采样、原始操作数与源码审计归档为
`external-evidence:compress351-355-complete-raw.json.gz`，
SHA256 `e59bd9d5d99b370e9373fa4ad68f5ab16b528f3496b886d367a08b3d7661217a`；
本地核对全部 22 个文件摘要和两项 case 的严格相位/子测试结果。

进一步在空闲设备上，不启用操作数观察器，连续执行社区原始
`test_matches_the_unfused_chain` **8 次**，每次仍覆盖 ratio 1/2
和原始数值、缓存字节断言。8 次均通过，无 failure/error/skip；
社区源码前后审计与 603 个受测文件摘要不变。此结果排除了本次
复测对张量采样同步的依赖，但仍未复现或解释原整轮的数值失败，
也不覆盖其他 case、变输入或 graph，不修改原 56/58 结论。
完整日志、原始测试、受测代码与审计归档为
`external-evidence:compress383-complete-raw.json.gz`，
SHA256 `8e0cfa9f8b7bd50e7288c52949308a24e77c435cb88f1cbc5f9833944b845c0a`；
本地核对全部 615 个文件摘要和八次运行的最终判定。

以下 54 项结果仅对应当时的清单与受测快照。

合并上游同步及 FA3、Conv1D、prefill 登记修复后的不可变快照，
完成五家族一次整轮 **54/54 PASS**：Dense 10、MLA 17、MoE 10、
Quantization 10、Mamba 7。逐项复核 115 个参数节点、58 个 subtests，
setup/call/teardown 全部通过，无 skip、xfail、错误或超时。
同快照的 CPU 组合回归 87 passed，CUDA 未初始化；
L0 GREEN，31 条已有警告。

社区 SGLang 0.5.22 commit
`baa9befa381511cd4d56f83980951e5a45fffc6d` 的全部 10594 个跟踪文件
及额外源码前后审计通过，603 个候选文件摘要保持不变；
本地另核对归档中的候选文件及当前 kernel/bridge 字节。
本轮快照不包含随后新增的模型 smoke 显存预算参数，该功能的独立
验证见 [模型配置验收](l15-model-smoke.md)。组件矩阵通过不证明
真实模型、独立 FP32 精度、其他配置或全部硬件能力通过。

原始报告、逐节点相位、日志、全部受测候选文件及审计归档为
`external-evidence:matrix272-complete-raw.json.gz`，
SHA256 `fb0d986dc83de401e352bb787677ccc241042dde68cf6a2876b15ca53521a55b`。
报告中的 `snapshot_sha256` 是排序后的文件摘要清单的 SHA256，
不是 ZIP 摘要；归档同时保留完整清单和文件字节。

## 固定边界

- 绑定社区 sglang v0.5.22 源码树；
- adapter 优先进入 `PYTHONPATH`，不修改上游源码；
- 恰好 1 张 HCU；
- 无服务、无模型权重、无外部网络；
- 只执行 manifest 内固定的 exact nodeid 或仓库相对脚本；
- 不接受任意 pytest 参数或 shell 片段。

入口：

```bash
python tools/l15_family_matrix.py validate
python tools/l15_family_matrix.py run \
  --up-tree "$HCU_UP_TREE" \
  --families dense,mla,moe,quantization,mamba \
  --out results/l15-family-matrix/run.json
```

## 代表行为

| 家族 | 核心行为 | 当前代表用例 |
|---|---|---|
| Dense | backend 构造、prefill、decode、GQA/MQA、页边界、碎片 KV、graph metadata/replay contract、fused-add RMSNorm 舍入、packed QK norm、原始 RoPE 与变输入 graph | 上游 Dense Triton 三个 exact class method + adapter 两项归一化及一项 RoPE 契约 |
| MLA | Tiny DeepSeek MLA 构造、compressed KV prefill/decode、碎片布局、graph contract、sparse virtual sink | 上游 MLA Triton 三项 + adapter sparse FlashMLA |
| MoE | Top-K 权重、grouped gate、block alignment、EP `-1` sentinel、capture-safe metadata | adapter MoE gate/align 五项 |
| Quantization | FP8 row/group/fused/scaled quant、scale 语义、异形 fallback | adapter Aiter quant 五项 |
| Mamba | Mamba2 prefill/decode、碎片布局、真实变输入 graph、状态槽位隔离、hybrid dispatch | 上游 Mamba2 四项 + adapter Mamba2/GDN 计算链与 Conv1D |

## 裁决口径

- 严格运行现强制单张真实 HIP 设备、绑定文件指纹和源码导入来源；
  逐 nodeid 的 call 阶段必须通过，skip/xfail/xpass/缺失报告均失败。
  一键入口与本轮执行计划见 [l15-readiness-plan.md](l15-readiness-plan.md)。
- sparse emu 数学项标为 `cpu-contract`，不计 HCU 内核覆盖。

- 上游 class method 内的 `subTest` 会作为一个 exact nodeid 整体裁决；这是上游粒度限制。
- 名称带 `cuda_graph` 的上游 Dense/MLA/Mamba runner 主要验证 capture/replay metadata 生命周期，不自动等价于真实 `torch.cuda.CUDAGraph` launch 证明。
- MLA 家族另包含社区 `TestQRopeStore` 的四个 exact nodeid，分别验证
  小/大 prefill 的 Q/RoPE 输出和 padding，以及真实变输入、变位置的
  `torch.cuda.CUDAGraph` replay。该测试传入 `norm=None`，不证明 RMSNorm、
  KV cache 存取、完整 attention 或模型精度。
- `mla-dsv41-low-ratio-compress` 执行社区原始
  `TestFusedLowRatioCompress::test_matches_the_unfused_chain`。它使用真实
  KV pool，覆盖 ratio 1/2 各 64 行的单步压缩，以独立公式检查归一化和
  投影数值，并与未融合链逐字节比较主 KV 与 index-K 缓存。缓存字节对照
  会复用社区量化和打包函数，不是这些函数的独立参考；该 nodeid 不包含
  跨步 pair-state 生命周期、padding/open-group 禁写或变输入 graph。
- `mla-dsv41-compress-pair-state`、`mla-dsv41-compress-padding` 和
  `mla-dsv41-compress-open-group` 分别执行同一社区类中的三个原始回归体：
  ratio-2 配对状态跨步保留和复用、ratio-1/2 padding 槽位禁写、
  ratio-2 未闭合分组不发布缓存。adapter 的 CUDA API 语义会使社区
  `is_hip()` 装饰器跳过它们，因此测试入口仅在实际 HIP runtime 和真实
  设备存在时，通过独立子类调用这三个已核对装饰器的原始方法。保留原断言、
  `subTest`、setUp/tearDown 与清理栈，不修改社区文件、类或平台判定；
  其他 skip、预期失败、错误和装饰器漂移均不能记为通过。这些用例不是
  真实 CUDAGraph replay，也不证明完整模型的压缩状态或数值正确。
  该社区 fixture 显式设置 `fused_compress=True`，因此验证的是融合路径；
  真实模型是否选用该路径须另取调用证据，不能据此排除未融合压缩链的问题。
- `mla-dsv41-unfused-pair-graph` 直接执行社区后端的未融合
  `_low_ratio_pair_partners`、真实 `CompressStatePool` 和原始
  `DeepseekV41Compressor.pool_pairs`。独立 CPU 历史保留每个逻辑 token，
  检查跨步 partner、长于 ring 的 prefill、混合请求、padding 禁写、
  未用槽位与空 sentinel；pooling 使用独立 FP64 softmax 加权公式。
  ring 4/8、head dim 32/512 四个参数节点各捕获一次真实 CUDAGraph，
  eager 和 graph 分别从空状态连续执行四轮变输入、位置及 padding。
  这是合成 FP32 投影数据的未融合状态和 pooling 契约，不包含真实
  投影权重、归一化、RoPE、index-K 或 KV cache 发布，也不证明模型路由
  或完整 attention 精度。
- MoE/Quantization 先使用 adapter-owned GPU contract，因为 v0.5.22 不同源码快照中的上游文件路径已有漂移；应在目标 HCU 树实际 collect 后再把稳定 nodeid 纳入主 classified baseline。
- 用例通过只形成 family-matrix green；晋级 `core-green` 仍需 route evidence，证明未意外走 CUDA stub 或错误 fallback。

`dense-original-silu-graph` 直接调用社区原始 JIT `silu_and_mul`，
覆盖 FP16/BF16、2/47 token 和 3072 维 gate/up。独立 CPU FP64
参考在最终 dtype 存储前应用 FP32 运算误差区间，同时检查整体相对 L2。
每组输入显式验证提前存储 SiLU 中间值的错误实现会被拒绝。
带前后保护区及非零 storage offset 的输入、输出，在单次真实
CUDAGraph 捕获后执行三轮变输入 eager/replay；检查输入与保护区不变，
且 eager/graph 逐位一致。该契约不覆盖量化激活、专家过滤或完整 MLP。

四项 Q/RoPE 用例已通过社区原始文件直接执行，并通过正式
`--families mla` 入口复测。扩展后的 MLA 家族 **9/9 case PASS**，
包含 8 个 GPU 契约和 1 个显式 CPU 契约；零 skip/xfail，前后全量
社区源码字节检查通过。另有 70 项 CPU 回归及 L0 GREEN
（34 条既有警告）。该次清单有 39 个逻辑 case，本轮仅重跑 MLA，
其余四个家族未重跑；不能将该结果解释为完整矩阵或真实模型精度通过。
原始环境、日志与机器报告保存在仓库外。

加入低压缩比用例后，清单为 **40 个逻辑 case**；正式 `--families mla`
入口重新执行 **10/10 case PASS**（9 个 GPU 契约、1 个 CPU 契约）。
所有选中节点和执行阶段通过，无 skip/xfail、错误或超时，运行前后全量
社区源码字节检查通过。新增社区原始用例单独执行也通过两个 ratio 子场景。
另有 70 项 CPU 回归及 L0 GREEN（34 条既有警告）。其余四个家族本轮
未重跑，该结果不替代完整矩阵、跨步压缩状态或真实模型精度验收；
原始环境、日志和机器报告仍保存在仓库外。

加入三项压缩状态回归后，当前清单为 **43 个逻辑 case**。正式
`--families mla` 入口整组 **13/13 case PASS**（12 个 GPU 契约、
1 个 CPU 契约），所有选中节点及 call 阶段通过，无 skip/xfail、错误或
超时，运行前后全量社区源码字节检查通过。三个原始回归体单独执行也
全部通过；直接执行带社区平台装饰器的版本则为三项 skip，保留为不完整
执行记录，不计通过。最终测试入口的 78 项 CPU 回归与 L0 GREEN
（34 条既有警告）通过。其余四个家族本轮未重跑，不能解释为 43 项
整轮通过或真实模型精度验收；原始证据和环境仍保存在仓库外。

加入未融合 pair-state / pooling graph 回归后，当前清单为 **44 个逻辑
case**。新增四个参数节点单独执行全部通过，正式 `--families mla`
入口整组 **14/14 case PASS**（13 个 GPU 契约、1 个 CPU 契约）。
全部选中节点的 setup/call/teardown 通过，无 skip/xfail、错误或超时；
前后全量社区源码字节检查通过。78 项 CPU 回归及 L0 GREEN
（34 条既有警告）通过。其余四个家族本轮未重跑；该结果不替代
完整 44 项矩阵或真实权重模型精度验收，原始证据继续保存在仓库外。

Sparse attention 的 CPU 契约现使用独立的标量 FP64 点积和显式零值
sink key，分别校验输出与 LSE，避免参考复用实现中的 FP32
`softmax` / `logaddexp` 步骤。新增手工可解的重复 selector、多重集合
排列、负索引与长度屏蔽、逐 head sink，以及只参与打分的非 value 通道
回归。当前清单为 **46 个逻辑 case**，其中 MLA 为 16 项；新增两项均标记
为 `cpu-contract`。本轮三个 sparse CPU 节点、矩阵工具与基础 CPU 回归
共 **73 项通过**，L0 GREEN（34 条既有警告），manifest 校验通过。
隔离的内存变异检查确认这些断言能分别拒绝错误 LSE、忽略 scale、
遗漏打分通道、selector 去重、忽略长度屏蔽和移除 sink 六类错误；
各次均为数值断言失败，无收集或执行错误。
这不是新的 GPU、完整矩阵或真实模型精度通过证据；前述 MLA 14/14
属于扩充前的执行记录。

加入 `mla-grouped-wo-a-graph` 时，清单为 **47 个逻辑 case**，
其中 MLA 为 17 项。新用例调用社区原始 `_apply_wo_a_bf16_matmul`，
以两组 `1024×4096` BF16 权重覆盖 4 行 decode、8 行 prefill；
独立 CPU FP64 分组点积对照有效行，使用有 padding 的 token stride、
NaN padding 与未使用 group，检查输入和权重未被改写。每个参数节点
只捕获一次真实 graph，三轮同时改变输入与权重，eager 和 graph 分别
检查数值契约，不要求跨批量形状逐位相同。

本轮正式 `--families mla` 整组 **17/17 PASS**（14 项 GPU 契约、
3 项 CPU 契约），全部节点 setup/call/teardown 通过，无 skip/xfail、
收集错误或超时；前后全量社区源码字节检查通过。另有 73 项 CPU 回归、
L0 GREEN（34 条既有警告）和 manifest 校验通过。其余四个家族本轮
未重跑，不能拼成完整 47 项矩阵通过；该组件也不替代缓存、后续量化
投影、通信或整模型精度验收。原始证据与环境继续保存在仓库外。

随后在提交 `a6540c6` 上完成了上述 47 项清单的五家族**一次整轮运行：
47/47 PASS**，其中 43 项 GPU 契约、4 项 CPU 契约；Dense 3、MLA 17、
MoE 10、Quantization 10、Mamba 7。所有参数节点和上游 subTest 均通过，
无 skip/xfail、收集错误或超时，前后完整社区源码字节检查通过。
这份整轮证据绑定修复 RMSNorm 回退之前的代码和 manifest，
不能转授为下一段新增后的 48 项整轮通过或真实模型精度通过。

新增 `dense-fused-add-norm-graph` 后，清单增至 **48 个逻辑 case**。
该项分别执行 Torch 回退和默认 Triton 的普通/Gemma fused-add RMSNorm，
覆盖 FP16/BF16，共 8 个参数节点。独立 CPU FP64 参考分别保留未舍入的
加法和与激活 dtype 存储边界：方差使用前者，归一化分子使用存回后的
残差。构造位于相邻存储值中间的加法和，明确拒绝“先舍入再算方差”；
每项只捕获一次真实 graph，三轮改变输入、残差及权重，检查 eager、
graph 输出、存回残差和权重未被改写。此项验证指定数值路径，
不替代 HF cast-before-multiply、vendor 档或整模型精度验收。

修复后的 Dense 家族正式运行 **4/4 PASS**，其中新增项的 8 个参数节点
全部 setup/call/teardown 通过，无 skip/xfail、收集错误或超时；
前后完整社区源码字节检查通过。另有 76 项 CPU 回归、25 项归一化文件
回归通过（排除 4 个 opt-in vendor parity 节点），L0 GREEN
（34 条既有警告），48 项 manifest 校验通过。这是 Dense 家族报告，
该家族报告本身不代表完整 48 项矩阵通过。

随后在修复提交 `a929a1a` 上完成该 48 项清单的五家族**一次完整运行：
48/48 PASS**。Dense 4、MLA 17、MoE 10、Quantization 10、Mamba 7，
其中 44 项 GPU 契约、4 项 CPU 契约；86 个参数节点的
setup/call/teardown 和 58 个 subTest 全部通过，无 skip/xfail、
收集错误或超时。前后全部 10594 个社区跟踪文件及额外源码检查通过。
该证据绑定 manifest SHA-256
`e12caa3265150f8bee50efcbbeddd90b01be88805fd7601d13fd545406b47b18`，
原始报告保存在仓库外。这是 L1.5 五家族矩阵通过，不代表完整 adapter
测试套件或真实模型精度通过；Qwen 默认 BF16 的 V1 精度失败仍保留。

新增 `dense-qk-norm-graph` 后，清单增至 **49 个逻辑 case**。
直接执行社区原始 `fused_inplace_qknorm` JIT 入口，覆盖 FP16/BF16、
128 维 head 的 2/47 token 与 16Q/8KV，以及 512 维 head 的 3 token
与 4Q/2KV。Q/K 是带前后保护区的同一 packed QKV 存储的非连续视图，
独立 CPU FP64 逐 head 对拍，容许至多相邻存储值的舍入差异并检查
相对 L2；单次 graph capture 后三轮改变输入和两组权重，要求
eager/graph 逐位一致、V 与保护区不被写入、权重不变。

新增项六个参数节点通过；接入后正式 Dense 家族 **5/5 PASS**，
17 个参数节点的 setup/call/teardown 全部通过，无 skip/xfail、
收集错误或超时，前后完整社区源码检查通过。另有 70 项 CPU 回归、
L0 GREEN（34 条既有警告）和 49 项 manifest 校验通过。
其余四个家族本轮未重跑，不将旧 48 项与本轮新增项拼成完整 49 项
整轮通过；直接内核合同也不代替 `apply_qk_norm` 条件选路或真实
Qwen 模型概率精度验收。

新增 `dense-original-rope-graph` 后，当前清单为 **50 个逻辑 case**。
直接执行社区原始 RoPE JIT 入口，覆盖 NeoX/GPT-J、FP16/BF16、
2/47 token、16Q/8KV、128 维 head，共 8 个参数节点。
Q/K 是共享 packed QKV 中带非零偏移和 token stride 的视图。
单次捕获后连续三轮改变输入、位置与 FP32 cos/sin 缓存，
包含位置 0/2048/4095、等系数相消、V 与前后保护区禁写，
要求 eager/graph 逐位一致，缓存与位置不被改写。

独立 CPU FP64 计算旋转及每个元素的 FP32 乘加误差区间，最后才转
输出 dtype；另检查相对 L2。相消结果的误差按两项乘积的绝对值和
界定，不能仅以接近零的最终值计算相对误差或相邻 BF16 步数。
本轮正式 Dense 家族 **6/6 PASS**，25 个参数节点的
setup/call/teardown 全部通过，无 skip/xfail、收集错误或超时；
70 项 CPU 回归、50 项 manifest 校验、L0 GREEN（34 条既有警告）
及前后完整社区源码检查通过。本轮只重跑 Dense 家族，
不将历史报告拼接成完整 50 项整轮通过；不包含融合 KV 写入分支，
也不替代实际模型的精度验收。

## 历史 HCU 实测结果

拆分前的兼容快照在与 manifest 两个 binding hash 完全一致的社区 v0.5.22
源码树上，以单张 gfx938 HCU 完整执行：**22/22 PASS**。该快照同时包含下述
生产兼容修复，现移至 [PR #27](https://github.com/HYGON-AI/adapter-das/pull/27)；
不能据此声称仅合入工具链的分支也已达到同样的 GPU 结果。

| 家族 | 通过 |
|---|---:|
| Dense | 3/3 |
| MLA | 4/4 |
| MoE | 5/5 |
| Quantization | 5/5 |
| Mamba | 5/5 |

该历史快照为取得上述结果修复了四类 adapter 兼容缺口，生产实现不在工具链 PR 内：

- 新版 paged allocator 无条件导入的 HCU kvcache ABI，回落到上游 Triton/torch correctness 路径；
- speculative 与 sparse attention 的可选 public import surface，未选择的能力允许收集，实际调用仍 loud fail；
- Mamba causal Conv1D 所需 CUB block headers 转发到 hipCUB；
- column-major FP8 scales 从无法在 gfx938 编译的 CUDA JIT 路径改为 torch correctness fallback。

该结果证明 family-matrix 行为通过；仍不自动等价于全部 case 已具备独立 route evidence 或性能达标。

## 当前明确缺口

1. Mamba2 组件级真实 CUDAGraph fresh-input replay 已补齐（见下）；纯
   Mamba2 真实 checkpoint、更多形状及 speculative/graph padding 仍未覆盖；
2. FP8 grouped expert GEMM 已增加小形状数值合同（见下），但真实 DeepEP
   多卡路由及模型精度仍未由此验证；
3. INT8 MoE 两段 GEMM 的端到端合成合同已增加，完整真实 checkpoint 精度仍缺；
4. INT4/W4A8 在 HCU 上主要是 correctness-first W4A16 降级，不能宣称原生 CUTLASS 行为；
5. v0.5.22 部分候选路径在本地后续快照已重命名，必须以 manifest 指纹绑定的目标树 collect 为准；
6. 该组件矩阵仍不覆盖权重加载、scheduler、HTTP serving 和真实模型生成；
   独立 Engine 冒烟的覆盖与实测见 [l15-readiness-plan.md](l15-readiness-plan.md)。

2026-11-03 扩展：新增 BF16 Triton MoE 的 up/SiLU/down/weighted-combine、
社区动态 INT8/FP8 Linear，以及 GDN 非方形状态池、envelope stride、
禁写和 ReplaySSM ring 验证。以上链路与 FP8/INT8 GEMM 都捕获真实 graph，
连续三轮变输入对拍。FP8 Linear 保留实际平台默认 dispatch；不伪报
CUTLASS 支持。独立 GEMM 用例验证 adapter 的 Triton 实现。
Conv1D emu 项与 sparse emu 一样明确标为 CPU 契约，不计 GPU 覆盖。

2026-11-03 真实 Channel-FP8 模型 trace 确认了社区 CT W8A8-FP8 →
dynamic token quant → fallback scaled-mm 路径。`quant-linear-chain`
补充 `ct-fp8` 参数：使用真实 scheme、channel scale、加载后处理和默认
padding，在 1/17/33 token 下独立 FP32 对拍并做三轮变输入 graph replay，
没有强制 CUTLASS 或替换平台路由。该 case 下现在收集 9 个参数化节点。
validation-host-a GPU5 的量化家族严格门 **8/8 case PASS**，包括这 9 个节点，
各执行后社区源码完整性通过；报告
`external-evidence:iteration21-quantization-family.json`。
这不是重跑全部五家族，也不替代真实量化模型的独立数值验收。

新增 `moe-int8-expert-chain`：真实官方
`W8A8Int8MoEMethod` 的权重创建、加载后处理、Triton runner、
top1/top2 路由及两段 GEMM/SiLU/combine，在 1/7/33 token 下验证。
量化整数先以独立 FP64 契约校验，仅在 FP32 reciprocal 的舍入误差界内
允许半整数两侧取整；尺度误差也单独断言，再以已验证的整数/尺度执行
独立 FP32 GEMM 参考。最终输出沿用原数值阈值和 cosine ≥ 0.9999，
并进行三轮变输入 graph replay。validation-host-b GPU3 的组件回归 **3 passed**，
见 `external-evidence:iteration30-int8-moe-chain.log`；
这不是全部 29 case 的整轮证据。

严格 MoE 家族增量验证 **7/7 case PASS**（零 skip/xfail，执行后社区源码
完整性通过），见 `iteration30-moe-family.json`。显式 W8A8 Linear 另增加
`quant-w8a8-linear`，矩阵现在为 30 case；公开权重创建、加载后转置、
per-token 量化与 scaled GEMM 覆盖 1/17/33 token、bias 两态及三轮变输入
graph，共 6 个参数化节点。输入设为可精确量化的二进制格点，隔离前述
半整数不确定性；独立 FP32 GEMM/尺度/bias 对拍保持原输出阈值。
validation-host-b GPU4 的原始 INT8 GEMM 与该新链路合计 **20 passed**，
见 `iteration31-int8-linear-gpu.log`；完整 30 case 尚待整轮验证。

2026-11-04 validation-host-a GPU4 的量化家族严格门 **9/9 case PASS**，包含新增
W8A8 Linear 的 6 个节点及 CT-FP8 等 9 个 quant-linear 节点，零 skip/xfail，
执行后社区源码完整性通过；报告 `iteration34-quantization-family.json`。
这是量化家族整轮证据，与 MoE 7/7 的独立报告分别保留。

新目录组合在 validation-host-c 单卡 GPU 0 的严格矩阵实测为 **27/27 PASS**，
原始报告 `external-evidence:phase2-all.json`。其中 25 个 GPU case，
另 2 个是显式 CPU 契约。社区路由入口接入后，计算链与 GDN 桥回归
另跑 **13 passed**。这次补链实际抓到并修复 GDN bridge 的多余状态转置；
内核实际按 `[slot, head, V, K]` 访问，保留原 pool view 才能同时保持数值
和 envelope slot stride。仍保留上述 FP8/INT8 专家量化等缺口。

第三步再新增非零 RoPE MLA case，当前清单共 28 case。新增 case 的
`256+32`、`512+64` KV 布局各测 prefill/decode/真实 graph，共 6 个
参数化节点在同节点全部通过；graph 连续三轮变输入独立对拍。
原 27 case 的整轮报告与新增 case 的增量报告分别保留，没有将两次执行
描述成一次完整 28 case 运行。模型级 Mamba graph 也已通过，但不替代
上述组件级 Mamba 独立变输入数值对拍缺口。

2026-11-04 新增 `mamba-fresh-input-graph`，固定清单增至 31 case。
用社区 fixture 构造真实 `MambaMixer2`，执行官方投影、causal Conv1D、
SSM、gated RMSNorm 和输出投影。CPU FP32 参考独立计算上述数学，
仅在公开的激活/状态存储边界转 BF16，不调用生产算子或其参考 forward。
batch1/page1/group1 与 batch3/page16/group2 各捕获一次真实 CUDAGraph，
三轮改变输入及非连续状态槽位；独立参考状态连续递推，
检查输出、卷积与 SSM 状态、eager/graph 一致性及未使用槽位不被改写。
这不替代 scheduler 图回放、纯 Mamba2 模型或 speculative 验收。

validation-host-a GPU4 的整个计算链文件 **25 passed**；严格 Mamba 家族
**7/7 case PASS**（包含 2 个新参数节点，零 skip/xfail），矩阵工具
回归 **32 passed**，执行后社区源码完整性通过。证据：
`external-evidence:iteration39-chains.log`、
`iteration39-mamba-family.json`、`iteration39-matrix-cpu.log`、
`iteration39-mamba-runtime.json`。该报告仅覆盖 Mamba 家族，
不是完整 31 case 的一次整轮运行。

2026-11-04 新增 `fp8-grouped-expert-contract`，清单增至 32 case。
masked 与 contiguous 两种 DeepEP 布局各覆盖 K/N=128 和 256，
128×128 权重 block scale、per-token-group 激活 scale、有符号 FP8，
以及非连续 scale view。masked 覆盖空专家、1/17/32 个有效 token
及 padding 禁写；contiguous 按 64-row expert alignment，覆盖 129 行
的尾 tile。每项独立 FP32 反量化对拍，保持 cosine ≥0.9999、
maxrel ≤0.05，真实 graph 三轮改变输入、权重、尺度和设备端
counts/expert IDs。这些是 adapter Triton 实现的直接合同，
不宣称已经覆盖分布式 dispatch/combine 或默认 DeepEP 选型。

validation-host-a GPU5 整个 FP8 GEMM 文件 **17 passed**（含新增 4 个参数节点），
执行后社区源码完整性通过，见
`external-evidence:iteration40-fp8-grouped.log` 和
`iteration40-fp8-grouped-runtime.json`。
接入后的严格量化家族 **10/10 case PASS**，零 skip/xfail，矩阵工具
**32 passed**，执行后社区源码完整性通过；分别见
`iteration41-quantization-family.json`、`iteration41-matrix-cpu.log` 和
`iteration41-quantization-runtime.json`。L0 **GREEN（31 warn）**、纯
CPU 算子 **38 passed**，见 `iteration41-l0.log`、`iteration41-ops-cpu.log`。
Mamba 7/7 与量化 10/10 是分别执行的家族报告，不拼接为整轮 32 case。

## DoD

矩阵完成的必要条件：

- 五个 family 均至少有一个构造/路由用例和一个数值或状态语义用例；
- 结果 artifact 逐 case 记录 argv、returncode、timeout、stdout/stderr；
- 任一固定 case 失败，runner 返回非零；
- 缺失上游文件 loud fail，不静默跳到其他源码树；
- 后续 HCU 实测结果按 exact case 更新，不以总通过数掩盖替换失败。

输入校验要求非空 `manifest_id`、正整数 `defaults.timeout`，以及恰好五个
不重复的家族；`families`/`cases` 每项必须为对象。布尔值不能冒充整数，
坏输入通过 `MatrixError` 返回可读错误。artifact 原子写入失败时清理临时文件，
并保留原有结果文件，避免磁盘错误产生残留或覆盖上一次有效证据。
