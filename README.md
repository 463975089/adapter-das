# adapter-das（HCU 适配层）

> 团队仓库：adapter-das（内网私仓；克隆地址与凭据走团队流程）。
> 从镜像到跑通的手工部署看 [docs/bootstrap.md](docs/bootstrap.md)。

**许可证与第三方声明**：本仓库以 Apache-2.0 发布（[LICENSE](LICENSE)）；
引入的外部开源源码（flash-linear-attention、vLLM、causal-conv1d）与内部
仓库移植清单见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

让宿主框架（**sglang / vLLM**）官方代码的 **python 部分零修改**、直接跑在
海光 HCU（HCU/DTK，HIP 路径）上的适配层（adapter）。

当前 SGLang 迭代以**社区源码文件原样、功能与精度正确、多真实模型验证**
为目标，性能不设门槛，每日追新最后接入。历史 JIT/text 磁盘补丁正在迁移，
尚未达到全链路源码零写回；严格组件/模型入口现检查运行前后源码完整性，
拒绝将已改写源码上的成功执行记为该目标通过。范围与证据见
[迭代目标](docs/l15-readiness-plan.md#当前迭代目标2026-11-03)。

**平台语义走 is_cuda 分支**（governance §0.5）：上游平台判定由 adapter
统一回答 CUDA——新模型/新功能上游先落 CUDA 分支，这是卡商追新的关键路径；
HCU 没有的 NVIDIA 库（flashinfer/trtllm 等）按**四级路由**找归宿：
① lightop → ② aiter → ③ **triton 自研先行**（先跑通，triton 优先保证正确性）
→ ④ 响亮 stub（fp8 fnuz 等语义护栏白名单锁定，见 governance.md §0.5）。

核心机制是**影子库替换**：对宿主直接引用的算子库（sglang 的 `sgl_kernel`、
`aiter`、`flashinfer`、`flash_attn`，vLLM 的 `_custom_ops` 等）不做调用点
改造，而是由本仓库提供同名/同 ABI 的影子实现并自动接管 import，官方代码
拉下来就能跑。算子实现不绑定特定后端：**先用 triton 实现保证正确性**，
**再做手写/vendor 优化**（lightop / aiter / flash_attn / flash_mla，
triton / vendor / torch 三档策略，见"影子库与适配桥"）。

与 **plugin** 的分工（[docs/governance.md](docs/governance.md) §0）：adapter
只做**行为还原**——宿主（sglang / vLLM）在其参照平台（AMD ROCm /
NVIDIA CUDA）上有的行为，HCU 上补齐到一致（影子库/
算子路由/质量门禁），并在一致约束下持续优化性能（三档档内演进、vendor 竞速）；
**上游没有的能力**归**独立 plugin 仓**——plugin 的目的同样是**性能优化**
（算子融合、专项内核、新调度路径等官方代码没有的功能），只是以"新增能力"
形态承载：经上游 seam 或注册机制接入、可直调 lightop/aiter 等底层库，
其代码不进本仓。判别问句："上游在参照平台上有没有这个行为？"——性能
不决定选仓，这个问句才决定。

影子替换能严丝合缝的原因：sglang 的统一算子层 `sglang.kernels`
（`BaseFusedOp` 多后端模型）在 HIP 平台上把 **AOT 后端**
（`forward_aot` → `import sgl_kernel`）作为默认内核路径（JIT triton 后端
只注册给 CUDA）。因此只要提供一个同名的 `sgl_kernel` 包，上游 sglang
零修改即可在 HCU 上解析到我们的内核。

**ai/新人上手请先读 [docs/reading-guide.md](docs/reading-guide.md)**（30 秒
任务路由 + 文档地图 + 不变量速查）。分层架构与组件协调关系总览见
[docs/architecture.md](docs/architecture.md)（配图
[architecture.html](architecture.html)，浏览器直开）；算子治理（实现选位
裁决链、active/ref 角色、运行时注册表与 dump、接入时序/共建边界/
维护闭环）见 [docs/governance.md](docs/governance.md)；测试体系从精度到功能、性能、
运行态路由统计与代码退役的目标路线见
[docs/test-route-governance.md](docs/test-route-governance.md)，L1.5 逐文件候选与分诊账见
[docs/l15-test-inventory.md](docs/l15-test-inventory.md)，对应 MR 拆分、DoD 与回滚见
[docs/test-route-implementation-plan.md](docs/test-route-implementation-plan.md)。

## 安装

双 dist（governance §7.4）：`sgl-kernel` 纯影子内核 + `hcu-adapter` 适配层
（sitecustomize + 桥 + vllm entry-points），版本线各自独立、`==` 互锁
（构建期由 `DAS_BUILD_STAMP=<时间>.g<commit>` 渲染出 `+hcu.<时间>.g<commit>`
同时写入两包版本，adapter 精确依赖同戳内核，无需上传后处理），
`sgl_kernel/__init__` 哨兵防半装（只装内核不装适配层 → import 期响亮报错）。
构建戳、互锁、`--no-stamp`、安装组合与排障见 [docs/build.md](docs/build.md)。

```bash
# wheel 态（部署）：packaging/build.sh 一次出两张 wheel 到 dist/
# 不带时间/commit 戳构建：packaging/build.sh --no-stamp
pip install --no-deps dist/sgl_kernel-*.whl dist/hcu_adapter-*.whl   # 两张都装，顺序无关

# 源码态（开发）
pip install -e .                       # sgl-kernel（影子内核）
pip install -e packaging/hcu-adapter   # hcu-adapter（适配层）
# 之后正常安装/运行 sglang（不要安装官方 sgl-kernel wheel，避免同名冲突）
```

> **升级提示**（entry-points 迁移动作，vllm plugin 注册归属变化）：vllm
> entry-points 已从 sgl-kernel 迁至 hcu-adapter。已部署环境升级到双 dist 版
> **必须先卸载旧版 sgl-kernel**（`pip uninstall sgl-kernel`）再安装两张新
> wheel——旧包残留既与影子同名冲突，又会让 vLLM plugin 注册双份/消失。

## 覆盖范围

算子/桥资产清单**以治理目录与战报为准，不在此维护静态表**：
- 后端与档位（emu/triton/vendor 演进、production/validated/candidate/
  broken 状态机）：仓库 `BACKEND_CATALOG`（机器可读）+
  [docs/config.md](docs/config.md) 状态语义；
- 功能面与量化路径：`hcu_adapter/`（JIT 补丁 / vendor 路由 /
  ops 重绑 / 模型级补丁三十余座桥）；
- 模型×并行实跑战报：[docs/models.md](docs/models.md)。

初版核心算子面（elementwise / layernorm / rope / moe gate / moe align /
kv cache io 的 triton·torch 实现，API 逐字对齐官方 sgl-kernel）见
`sgl_kernel/` 包本体，不赘列。

设计原则：
- API 签名与官方 `sgl-kernel/python/sgl_kernel` 逐字对齐（含 `out=`、
  `enable_pdl` 占位参数、destination-passing 语义）。
- triton kernel 数学均在 fp32 累加；bf16/fp16/fp32 输入。
- import 永不失败：未覆盖的算子在使用时抛带提示的 `NotImplementedError`。
- GPU 路径禁用一切主机同步（`.item()` / `nonzero` / 布尔掩码索引），
  保证 decode CUDA-graph capture 链路可用。

## 影子库与适配桥（sglang/vllm python 零改动）

| 组件 | 作用 |
|---|---|
| `sitecustomize.py` | PYTHONPATH 自动加载的三行入口；只调用 `hcu_adapter.bootstrap.install()`，不承载探测、桥实现或数值逻辑 |
| `hcu_adapter/bootstrap.py` | 宿主分发器：按 `HCU_ADAPTER_HOST`、启动参数、已加载模块和可用包确定 `sglang` / `vllm` / `none`，先装 common，再只装所选宿主桥 |
| `hcu_adapter/bridges/` | 宿主专属 glue：`sglang.py` 保持既有桥顺序，`vllm.py` 集中 vLLM `_custom_ops` ABI 绑定，`common.py` 只放跨框架启动守卫；官方已有的行为不在这里重复实现 |
| `sgl_kernel/` | 同名影子包 = `sglang.kernels` 的 AOT 默认内核路径：elementwise / layernorm / rope / moe / kv cache io / fp8·int8 GEMM / flash_mla emu，API 逐字对齐官方 sgl-kernel |
| `hcu_adapter/` | 元路径影子库：`aiter` 真包（DTK vendor wheel）优先 + 缺口填空（`fused_clamp_act_mul` 等）、`flashinfer` / `flash_attn` 桩——import 成功、未实现调用响亮报错；另含 `hcu_env` 环境注册表、`kill_switch` 快关、slimquant / paged-MQA logits（emu/triton 双档）/ dsa_topk / quant·ar-norm 等 vendor 桥模型级补丁 |
| `sgl_kernel/kvcacheio.py` | PD 分离/分层 KV transfer（transfer_kv_* 契约，上游 mooncake 等 KV 传输路径的调用面）的完整 torch 移植 |

## 运行要点（HCU 实测）

**推荐形态 = 零环境变量起服**（"正确部署零环境变量"是本项目的回归治理
原则）：平台语义、JIT 预热关闭、模型感知默认（DSA 融合逃生门等）均由
adapter 自动落位，启动日志前 20 行可见每个生效值及来源
（`HCU adapter env:` 摘要行）。手敲 env 只用于调试/隔离。

```bash
export PYTHONPATH=/path/to/hcu-sglang-adapter:/path/to/sglang/python
python -m sglang.launch_server --model-path <model> --port 30000 \
    # 默认路由已按平台自动选型；后端/量化显式旗标只在需要固定形态时加
```

- 典型生产已验证：GLM-5.3-Flash（KDA+DSA 混合，306G TP8，长文解码
  fork 内核转正后 +80%）；MoE 线 Qwen3.6/3.5 系、Qwen3-30B-A3B；dense
  小模型；CT int8 / W4A8 桥；EP（DeepEP LL）与 PD 分离；完整模型×并行
  战报见 [docs/models.md](docs/models.md)。
- **正确性/性能卡位分工**：BW1102（gfx938，validation-host-26 级）= 生产卡型；BW1101
  类仅用于离线正确性对拍，性能结论与 gfx938 特有判别必须生产卡复核。
- thinking 模型建议给足 `max_tokens` 或用
  `chat_template_kwargs: {"enable_thinking": false}`，否则思考文本会占满
  输出预算（评测误判的常见来源）。
- 后端配置/启动命令生成/每卡显存评估可用浏览器直开
  [tools/hcu_config_ui.html](tools/hcu_config_ui.html)（详见
  [docs/config.md](docs/config.md)）。

## 测试

分层门禁体系（**以 [docs/ci.md](docs/ci.md) 为准**，本处为速览）：

```bash
pytest tests/test_ops_cpu.py -v          # 纯 CPU，torch 兜底路径（任何机器）
pytest tests/kernels tests/adapter -v    # 自有算子/桥（HCU 节点，含图 capture 回归）
pytest tests/vendor -v                   # vendor 高危：FA 场景矩阵 + aiter AR 压测/快关 gate
bash ci/l1_gpu_gate.sh                   # 一键全门：L0 + cpu + kernels + 上游单测 + 门禁 + vendor
python tests/tools/run_triage.py         # 精度问题四级快速定位（容器内）
```

- **上游官方单测常驻收集**（L1.5）：容器内上游树的
  `test/registered/attention/unittests`（dsa/kda/dense/mla 等目录）以
  上游代码为入口直接 pytest（零拷贝——收集列表/过滤器/环境注入在
  `ci/l1_gpu_gate.sh`，测试代码本体不进本仓，随上游版本自动跟进）。
- 自有算子三段式单测（参照/形状矩阵/容差，容差表见
  `skills/hcu-adapter-dev/references/test-suite.md`）是质量金标准；
  单测绿 ≠ serving 通，模型级验证以 models.md 战报为准。
- 高危算子均配**快关**：`HCU_DISABLE_OPS=<op>` 一翻开关即绕开疑似算子
  复跑；快关注册流程见
  [`skills/hcu-adapter-dev/references/kill-switches.md`](skills/hcu-adapter-dev/references/kill-switches.md)。
- 端到端精度锚点（Berlin/Tokyo 事实检索 + 马拉松 + 长文门）作为 serving
  级守卡，有效性实证：零 env 四门首跑曾连抓两个栈级 bug（详见
  models.md 2026-10-19 行）。

## 开发规约（skill）与协作

改本仓库前先读 [`skills/hcu-adapter-dev/SKILL.md`](skills/hcu-adapter-dev/SKILL.md)：
影子库矩阵与三档实现策略（triton/vendor/torch）、测试四质量维度、
快关注册流程、精度定位流程、cookbook（NVIDIA 参数）降级矩阵。
多人协作红线（分仓边界/裁决链/接入时序/共建边界）见
[docs/governance.md](docs/governance.md)（§0/§3/§7）；`tests/tools/` 物料
命名语法见 [tests/tools/README.md](tests/tools/README.md)。

## 已知未覆盖 / 路线图

（已验证特性与模型清单以 [docs/models.md](docs/models.md) 为准——零 env
部署、EP/LP 拓扑、PD 分离、投机解码 MTP、hicache、量化桥等已通；此处只列
未覆盖项。）

- 量化路径：compressed-tensors float-quantized（FP8）格式的 serving 层
  （已知病灶，未覆盖）；W4AFP8 混合面（int4-channel 权重 × fp8-dynamic
  激活，上游 scheme 有缺口）；FP8 DeepEP normal（a_scale 语义 ≠
  128-block）。
- aiter attention 的 AMD 原生符号（`paged_attention_ragged` / `mla_*`）：
  按需 triton 化（混合适配方案）。
- hicache 的 GDN 混合模型 MAMBA 状态宿主迁移输出漂移待复核（dense 全通，
  等 GDN rig 配 hicache 窗口）。
- DTK 工具链已知缺口（内核级，待上游/轮子跟进）：triton codegen 布局
  敏感误编译疑点（conv1d parity 有 xfail-strict 桩常驻门禁自动跟踪）、
  tilelang 0.1.13+ ThreadSync 回归（已钉 0.1.12）、fp8 内核族符号病理
  （详见 models.md 判例）。
- plugin 仓：上游没有的新能力（算子融合/专项内核/新调度路径）归独立
  plugin 仓——sglang 侧已立仓 `sglang-plugin-das`（挂上游 v0.5.22 官方
  entry_points + HookRegistry）；分工与判别问句见文首，索引与共建边界见
  docs/governance.md §0/§7。

## CI

MR 快速自动测试分层（L0 结构门禁云端秒级 / L1 GPU 数值门禁容器侧 / L2 e2e 夜间）：见 [docs/ci.md](docs/ci.md)。

## DeepInfer 接入

固定上游接口的 HCU 适配候选见[DeepInfer 手册](docs/frameworks/deepinfer.md)与[首版范围](docs/frameworks/deepinfer-scope.md)。模型扩展保留在独立 plugin；新组合尚待节点验收。

vLLM 的必需扩展/capability 配置、唯一 owner 与失败报告见[启动要求](docs/frameworks/vllm-required-extensions.md)。
