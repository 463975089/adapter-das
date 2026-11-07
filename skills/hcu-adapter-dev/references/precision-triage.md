# 精度问题快速定位流程

`run_triage.py` 的 pytest 组只有实际 passed 且无 skip/xfail/xpass 才算
PASS；全部跳过标 SKIP，混合结果或没有可识别通过计数标 INCOMPLETE。
总退出码 0=全部所选组已验证，1=存在失败，2=无失败但验证不完整；
缺少 layers/e2e 资产和仅环境信息均不能输出 ALL GREEN。定位报告中的
跳过与预期失败须单独列出，不能算作设备或模型通过。

默认 `ops/rope-parity` 执行 `test_original_rope.py` 的 8 项社区原始
JIT 数值/变输入 Graph 契约，不依赖模型目录。只有显式设置
`HCU_CKPT` 才额外执行旧配置探针 `ops/rope-checkpoint-parity`；
显式路径或配置错误仍报告失败。该组件测试不代表 checkpoint
模型精度通过。入口回归修复前 2 FAIL，修复后连同 CPU 算子
55 PASS（隐藏 GPU、CUDA 未初始化）；真实 HCU 的诊断子组
8 PASS、源码前后审计通过，L0 GREEN（31 条既有警告）。
这里只验证变动的 RoPE 子组，FA/AR 跳过及整轮 ops 状态不作追认。
证据 `external-evidence:triage271-complete-raw.json.gz`，
SHA256 `099aaba347fee83e29af326ad8a43a2ed60ac01609af5f67888d229c8bc6fc86`。

入口：`python tests/tools/run_triage.py [--stage env|ops|layers|e2e] [--full]`
（容器内、adapter 根目录）。四级漏斗，每级 FAIL 都会打印疑似算子 + 对应
快关 + 复现测试。

## 症状 → 怀疑面速查（先按症状收窄再跑工具）

| 症状 | 首要怀疑 | 快速验证 |
|---|---|---|
| 第 2 个 decode token 起分叉，token1/prefill 对 | 图内算子（AR 类） | `HCU_DISABLE_OPS=aiter_ar` 复跑；`--full` AR 压测 |
| 首个生成 token 就垃圾 | prefill 链：rope / 量化解包 / GEMM | `_rope_probe.py`、`_w4a8_probe.py` |
| 越生成越差 | KV cache / attention / 位置编码累计 | FA 场景矩阵；长 prompt e2e |
| eager 对、graph 开就坏 | 图内烘焙（H2D/host 状态） | `_graph_safety_probe.py` |
| top1 抖动但文本通顺 | bf16 噪声带（非 bug） | 层对齐看 cos 而非逐 token |
| 一次性全垃圾 | 权重解包/scale 约定（÷16 等） | `_w4a8_probe.py` 分解段对拍 |

## 四级流程

1. **env**：arch（gfx938 有 fp8）、wheels、快关状态、HCU_* env——先确认
   配置如你所想（gate 是否生效、emu 是否开着）。
2. **ops**：rope/rmsnorm/gemm/graph-safety 探针（数值谓词判定）+
   `tests/kernels/` + FA 场景矩阵 + AR gate（`--full` 加 2 卡 AR 压测）。
3. **layers**：12 层截层快启（`launch_w4a8_12l.sh`，`HCU_DUMP_LAYERS=1`）
   + 纯 torch 参考（`_ref_forward_h.py`）→ `_bisect_cmp.py` 找第一个
   DIVERGE 层。**TP8 坑**：残差流延迟归约，module hook 抓到的是 rank0
   部分和（norm≈1/8），唯一可靠捕获点 = 各层 `input_layernorm` 的
   forward_pre_hook（归约后）。
4. **e2e**：短 prompt（语义锚 "Paris"）+ 2700-token 长 prompt
   （>index_topk=2048，覆盖 sparse decode 变体）。判定用分布级锚点
   （见 run_triage.py 实际实现与 docs/models.md 验证口径；无 golden JSON
   基线文件，跑分记录以 models.md 行为准）。

## 方法论要点（历史沉淀）

- **绑定 checkpoint 内的模型代码**：`tools.checkpoint_files` 除权重与配置，
  还递归绑定本地 `.py` 字节，覆盖 `trust_remote_code` 使用的配置、
  tokenizer、模型及相对导入 helper；指纹过程不执行这些代码。
  编辑、新增或删除源码均使旧绑定失效，不能将历史遗漏源码的清单
  倒推为完整绑定。该修复的 158 项 CPU 回归与 L0（34 条既有警告）
  通过，不构成新增模型精度结论。依赖包和运行环境另行留存于仓库外。
- **一次只变一个变量**：换镜像/换栈/换旗标都会移动基线，先同栈复现再隔离
  （--disable-cuda-graph 首测被镜像变量骗过一次）。
- **稳定性与独立精度分别判定**：真实 30B MoE 的 softmax 并列选择修复后，
  同 Engine 重复及 Eager/Graph 的完整返回概率均逐位一致，原 CPU FP32 V1
  仍失败，详见 `docs/l15-readiness-plan.md`。不能继续用重复分叉解释
  已稳定结果，也不能把 CPU BF16 诊断或 greedy 一致当作原分布门禁通过。
  只保存聚合误差的历史报告不能独立重算完整分布，应明确证据边界。
- **先确认计算开关实际改变输出**：真实 Qwen3 短双语 HF GPU BF16
  诊断中，原始流及两种 fused residual 分子约定分别切换
  `allow_bf16_reduced_precision_reduction`，三组完整概率张量均逐位一致。
  同一诊断的 GPU FP32 对照与 CPU FP32 top-20 最大 logprob 差为
  `9.728e-5`。该证据排除了此开关作为这些输入的修复，不外推其他
  GEMM 形状、模型或实现，也不豁免 SGLang 原 V1 精度失败。
- **观察原地函数需同时绑定调用边界与控制输出**：模块 hooks 可能漏掉
  直接融合调用；以原函数 code object 观察时应在未启用采样的 warmup
  初始化，拒绝覆盖其他 profiler，先克隆写回前输入、再保存返回后输出。
  覆盖计数须匹配实际层数/调用数，观察后 token 及完整报告概率须与
  同节点未观察对照一致，最后绑定源码与权重审计。CPU synthetic
  预检不等于真实设备采样，具体 Qwen 普通及融合 QK Norm 证据见
  `test-suite.md`。
- **图问题用探针不用猜**：capture→replay 换输入，输出跟随=安全；不跟随=
  烘焙。三个桥（rope/rmsnorm/topk）全靠这招排除。
- **前缀共享可替换物理 KV 槽位**：prefill 后的 radix cache 可能将另一
  请求的相同前缀映射到已缓存槽位。不能只按首次分配的槽位列表断言
  decode 缓存串用；须核对实际 KV 索引、完整 token 前缀及逐位 K/V
  等价，再按语义历史做独立 attention 参考。真实短双语 Qwen 诊断
  已逐项验证这种替换，见 `test-suite.md`；这不是允许任意槽位重排。
- **解码对照先匹配完整历史**：`tools.l15_decode_consistency` 比较
  `input_ids + output_ids[:position]` 完全相同的位置，支持 eager/graph
  与冻结前缀的单 token prefill 对照。输出分叉后不得继续按数组位置比较
  不同条件下的概率；ID 全相同也需查看共同 top token 的概率偏差。
  未报告 token 的概率属于缺失证据，不能当作零误差。工具只做概率诊断，
  不验证源码/权重绑定、随机种子或独立参考，不能作为模型精度通过。
  若 eager 自身的增量解码也与完整前缀 prefill 不同，排查范围应包含
  KV cache、量化及 prefill/decode 算法与舍入边界，不能只归因 Graph。
  这些路径本身可能采用不同数值算法，完整 prefill 也不是独立精度金标。
- **attention 舍入须分段归因**：从实际低层调用取得有效 mask、选中并
  反量化的 key、Q、sink 和 scale；缓存地址或后端的占位 `k/v` 不能代替
  实际读取值。key 按保留重复项的多重集合对齐，排除未消费的 padding
  head。独立高精度参考须包含零值虚拟 sink，并直接正确舍入到 BF16，
  避免先转 FP64 再转 BF16 在中点附近产生二次舍入。分别检查打分、
  softmax 概率和 P@V；若对齐后的概率相同，仍需检查累加顺序。先用
  原形状重放复现已采样输出，再只交换概率与 value 的对应排列验证
  因果。原始 FP32 累加器未保留时须明确缺失，不能从 BF16 输出倒推。
  局部舍入归因不替代模型分布验收，也不能据此排序生产 key、提高默认
  dtype 或放宽原精度门，来强制跨模式逐位一致。
- **金标在 fp32/torch 侧**：NCCL 也在被测链路里，不能当参考；AR 黄金
  参考 = all_gather 后 fp32 精确和。
- **存储 dtype 不等于 GEMM 计算 dtype**：真实 Qwen3 MoE 的显式 FP32
  短请求中，首层 prefill/decode 的 FP32 hidden states 与专家权重
  实际传入社区 Triton GEMM 时，四次 `compute_type` 均为 FP16。
  须绑定实际调用及未观察控制输出，再按原内核的输入转换、累加、
  路由权重和输出舍入顺序做独立参考；不能只凭 server dtype 判断
  全计算链 FP32，也不能修改社区默认 dtype 来代替精度定位。
  该首层观察不外推全层或历史请求，详见 `docs/l15-readiness-plan.md`。
- **计算类型参数须按内核阶段解释**：上述非量化 MoE 分支的
  `compute_type=FP16` 作用于输出转换，A/B 仍直接按 FP32 加载；
  原形状重放的实际编译产物显示 TF32 dot、FP32 累加及 FP16
  输出舍入。须分别检查输入转换、累加、路由相乘和存储边界，
  不能将参数名等同于乘法精度。TF32 向零截断参考能匹配大部分
  真实输出，仍不等于指令级模拟；只读缓存候选也不能冒充实际
  调用的内核绑定。详见 `docs/l15-readiness-plan.md` 的分段核对。
- **量化参考保留激活 QDQ**：CT channel-FP8 的 CPU 参考必须同时包含
  动态 token-FP8 输入量化；只反量化权重会改变模型。显式参考入口逐个核对
  绑定权重和真实逐层 QDQ 调用，默认拒绝其他量化格式。原分布门失败仍须
  保留；社区 CT 只接受 FP16/BF16，不能扩大 dtype 准入来获取 FP32 对照。
- **逐层对比先对齐计算约定**：HF Qwen3 RMSNorm 在乘权重前转回 BF16，
  社区 SGLang 默认在乘权重后转回；同输入下的首层差异不能直接判为
  adapter 错误。fused-add norm 必须分别保存相加前输入和残差，社区 JIT
  默认以未舍入 FP32 和计算方差、以激活 dtype 舍入后的和作分子。
  对照实际选用的路径，不能拿另一条路径的公式替代其契约。
- **观察器须有对照**：社区公开 `--forward-hooks` 可供 eager 采样；
  核对实际 server info、注册日志和样本覆盖，参数被接受不等于 hooks 已匹配。
  当前社区在 graph 捕获之后注册 Python hooks，不能据此声称观察到图重放。
  custom weight loader 属于在线权重更新流程，不是启动后的通用采样入口；
  不得绕过其更新生命周期守卫。复现原增量请求并核对全部报告的 ID/logprob，
  不能用单请求全量 prefill 推断原问题。CPU clone 含主机同步，仅供 eager
  诊断，需保留未加观察器的对照。
- **逐层采样先核对行覆盖与专家对应关系**：MoE 最后一行相同不能证明
  整个 prefill 相同；其他 token 的差异可能经过下一层 attention 影响
  最后一行。定位首次差异应补齐相关层的全部有效行，并分别保存残差、
  attention 与 MLP 边界。top-k 数组顺序不同须先按专家 ID 对齐集合和
  对应权重，不能直接判为路由错误；相同集合也不证明不同累加顺序逐位
  等价。实际 token/位置、形状及物理缓存映射一致只排除已采字段的差异，
  不证明 KV 内容或具体内核正确。真实串行 MoE 观察证据见
  `docs/l15-readiness-plan.md`。
- **实查 MoE router 的运行时消费者与并列规则**：模块源码中存在
  `_is_cuda` 分支不证明真实请求进入它。真实 BF16 MoE 的完整行观察与
  独立算子重放发现，当前进程通过社区 `topk_softmax` 调用影子算子；
  旧 `torch.topk(sorted=False)` 在完全相同 logits 的并列边界选择不同
  专家。社区 power-of-two AOT CUDA 和 Triton softmax router 明确使用
  降序分数、并列时较小专家 ID。影子 softmax 使用稳定降序排序恢复
  该规则，不给 logits 加 epsilon、不改 gate dtype。按来源记录实际
  Python 调用链并在原形状重放，不能只由静态源码推断选型。
- **层名和 token 行必须实查**：DSV4 的 `_next_layer` 模块引用会影响
  `named_modules()` 去重后的规范名称，固定的 `model.layers.N.mlp`
  可能匹配不到实际存在的层。可用覆盖嵌套路径的模式发现模块，再按实际
  `layer_id` 标记，并检查每层、每 rank、每请求的覆盖；匹配失败不是缺层证明。
  TP 的尾部 padding 与 `tensor_split` 会改变同一 token 所在 rank 和行号。
  先用实际进程与 rank 映射、原始 token 历史和各分片形状定位有效行，
  不比较任意两个 row 0，也不把 padding 当作有效激活。post-hook 的输入
  可能已被原地修改，归因前还需与计算前的独立 clone 核对。
- **混合向量一致不等于残差流一致**：DSV4 的 MLP 输入只是 mHC 多条残差流
  混合后的向量；前层 MLP 的输入和输出相同，不能排除其原始残差、
  Engram 或后续 mHC 状态不同。若最早差异已在某层 MLP 输入，先观察该层
  attention 及前后的原始状态，不能直接归因为该层专家或压缩比变化。
  实际模型可直接调用 `.forward` 或层方法，绕过普通 module hooks；
  attention 又可用关键字传参，默认 post-hook 的 `args` 因而为空。
  还须区分完整 attention 输出与 `AttnOutput.partial` 的待归约局部和，
  以及 `HcState` 的物化残差与待执行 post。入口状态可能被 `release`
  清空，须在消费前独立保存；观察器不能通过主动物化状态改变执行路径。
- **分布级断言**：bf16 噪声会重排 top1，判定用 cos/top-k 集合/文本锚点。
- **attention 参考需保留指数权重转换位置**：grouped decode 的社区
  内核可在 P·V 前将 `exp(score-max)` 转为 V dtype，同时以未转换的
  FP32 权重求分母；直接用理想 softmax 或将归一化概率转 BF16 都不等价。
  先核对实际 dispatch、平台分支、KV 历史、split 元数据与有效 tile。
  实际双行 BF16 首层样本中，保留该转换位置的数学参考重现全部
  8192 个 context 元素，而理想参考转换后仅匹配 6296 个。
  这只是该层、该步、实际缓存的局部证据，不证明历史缓存生成、
  硬件指令序列或整模型精度通过；详见 `docs/l15-readiness-plan.md`。
- **模型 BF16 不代表 RoPE 缓存 BF16**：独立生成历史 KV 时实查
  cos/sin 的存储精度和乘加边界。实际首层样本的缓存为 FP32，
  将其误设为 BF16 的数学变体只匹配英文 12527/14336、
  中文 12054/13824 个历史 K 元素；保留 CPU FP32 三角函数后分别
  匹配 14334/14336、13824/13824。CPU 与设备三角函数仍可能有
  差异，不把这一变体当成逐指令重放或整模型验收。
- **短序列 DSA 注意**：kv_len ≤ index_topk 走 k_only 快速路径，MQA emu
  根本不被调、logits 全零是设计内——别在这里白排查。
- **搬内核先对拍"入参语义"再对拍数值**（2026-10-16 vendoring sglang-das
  fla/hcu）：fork chunk_kda 期望已激活（sigmoid 过）的 beta，喂 raw beta
  得 cos 0.37 且层层查内核都"对"——入参约定（beta_is_raw/gate ndim）先
  核实，别急着怀疑 kernel。
- **standalone 门 ≠ serving 门**：fork 内核 standalone 对拍全绿（含服务
  形状、非连续状态视图）但 serving 中 Berlin 锚点确定性翻转；emu/fork
  启动二分 + 干净重启复现可实锤"内核在服务上下文特有"。任何内核替换默认
  不启用，e2e 锚点门过了才转默认。
- **必须比较真实 decode 边界**：完整 prompt 的 CPU 参考与 HCU decode 输入不能直接跨执行模式比较。观察真实 prefill 加四步 decode 后，四步 normalized K 和 RoPE 后 K 均逐位匹配 staged FP32-RoPE 参考；目标缓存的中英文 K/V 分别为 14334/14336、14332/14336 和 13824/13824、13820/13824。先核对 token/position、物理 KV 索引、尾行写入和实际 QKV 边界；少量历史元素残差不足以证明 KV cache 错位，也不足以证明整模型精度通过。
- **先验证 prefill/cache 链接再归因**：同一真实批次中，prefill 产生的 47 行 K/V 与目标 decode 读取的历史缓存前 47 行可以逐位对齐；若 K/V 均 24064/24064 匹配，少量历史差异不能归因于 cache write/read 索引错位，应继续比较算术边界。
- **按坐标追踪历史残差**：若实际 prefill K/V 与目标 cache 前缀逐位匹配，再将 staged 参考差异坐标映射到 prefill 行，才能区分历史算术舍入与 cache 路径问题。当前样本的英文 2 个 K、英文 4 个 V、中文 3 个 V 均来自 prefill；中文另 1 个 V 来自第二个 decode，四步 decode 的 K 全部匹配。
- **先固定实际投影输入再分析归约**：历史 prefill/decode 样本的投影输入为 112639/112640 逐位匹配，一个输入差异足以改变对应 V 坐标。以实际输入计算 FP64、FP32 顺序及二叉树累加，只能按坐标分别重现 6/10 个残差；不能将选择不同变体的局部匹配说成一个算法或硬件指令重放。两处 K 归一化坐标用实际 raw K 可重现时，应继续追踪投影边界，不能直接判为 norm 错误。
- **局部替换需验证下游影响**：先用实际 Q/K/V 的数学参考重现已采集 context 和输出，再分别替换 Q、K、V 及全部操作数。当前首层目标位置全部替换仅改变 43/4096 个输出元素，最大变化约 `6.10e-5`，相对 CPU FP32 的最大差仍约 `4.87e-4`；这只证明该位置的局部敏感性，没有运行后续层或解释整模型 V1。FP32 RMSNorm 变体重现一个输入行，也不能唯一反推硬件归约顺序。
- **残差合同需同时采计算前后**：在 eager 原请求中保存 norm 的输入、原残差、权重、输出及写回残差，再核对 attention/MLP 和前后层连接，不通过物化待执行贡献改变路径。当前双行 48 层样本中，96 次 norm 的 393216 个输出与 FP32 数学变体全部匹配，95 次写回的 389120 个元素及 47 处层间连接全部匹配。未舍入和作方差、激活 dtype 存储后的残差作分子是社区默认融合合同，不能换成另一公式来判错；实际合同匹配也不替代整模型 V1。
- **路由敏感性要分开固定输入、分数与集合**：用真实专家权重在同一实际行上重放官方 experts，先核对原 CPU 路由端点，再分别替换输入、路由分数和专家集合。当前两处首次集合变化的局部输出，固定 HCU 分数而换集合产生的 RMS 变化约为 0.01242/0.01740，方向与实际总误差接近；这是局部放大证据，不能把 RMS 分量直接相加、判定 top-k 实现错误或替代整模型路由替换及原精度验收。
- **专家链按实际中间值分段对照**：先核对所选权重与 checkpoint，再分别从实际第一段输出重算激活、从已乘路由权重的第二段输出重算求和，避免上游误差混入下游判断。当前四层样本的 49152 个激活和 16384 个归约元素全部匹配，但两段 GEMM 相对 FP64→FP32→BF16 仍有 29/229376 个差异；独立重算确认统计后也应保留这些残差，不将局部链路证据替代整模型精度。
- **局部路由对照还需验证最终输出**：私有干预仅替换目标专家集合、保留实际输入与分数，必须有无干预观察、同集合重算、请求隔离和移除干预后的恢复控制。真实目标位置中，恢复 CPU 集合使英文最大 top-20 差由 0.606 降为 0.419，中文却由 3.972 升为 4.427，两者仍失败；不能把局部改善或强制路由作为生产修复。API 返回值与 JSON 归档比较前，还需统一 tuple/list 容器表示并保留所有浮点值，避免将结构差异误判为数值变化。
