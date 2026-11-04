# 精度问题快速定位流程

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

- **一次只变一个变量**：换镜像/换栈/换旗标都会移动基线，先同栈复现再隔离
  （--disable-cuda-graph 首测被镜像变量骗过一次）。
- **图问题用探针不用猜**：capture→replay 换输入，输出跟随=安全；不跟随=
  烘焙。三个桥（rope/rmsnorm/topk）全靠这招排除。
- **金标在 fp32/torch 侧**：NCCL 也在被测链路里，不能当参考；AR 黄金
  参考 = all_gather 后 fp32 精确和。
- **量化参考保留激活 QDQ**：CT channel-FP8 的 CPU 参考必须同时包含
  动态 token-FP8 输入量化；只反量化权重会改变模型。显式参考入口逐个核对
  绑定权重和真实逐层 QDQ 调用，默认拒绝其他量化格式。原分布门失败仍须
  保留；社区 CT 只接受 FP16/BF16，不能扩大 dtype 准入来获取 FP32 对照。
- **逐层对比先对齐计算约定**：HF Qwen3 RMSNorm 在乘权重前转回 BF16，
  社区 SGLang 默认在乘权重后转回；同输入下的首层差异不能直接判为
  adapter 错误。fused-add norm 必须分别保存相加前输入和残差，社区 JIT
  默认以未舍入 FP32 和计算方差、以激活 dtype 舍入后的和作分子。
  对照实际选用的路径，不能拿另一条路径的公式替代其契约。
- **观察器须有对照**：custom weight loader 安装只读 hooks 时检查公开
  begin/update/end 的成功返回和采样文件完整性；未成功安装不能算采样成功。
  复现原双请求增量解码并核对输出 ID/logprob，不能用单请求全量 prefill
  推断原问题。CPU clone hooks 含主机同步，仅供 eager 诊断，不能进入 graph。
- **分布级断言**：bf16 噪声会重排 top1，判定用 cos/top-k 集合/文本锚点。
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
