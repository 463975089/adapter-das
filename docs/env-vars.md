# HCU Adapter 环境变量手册（调试覆盖层）

> **2026-10-12 起定位调整**：后端选择的正式机制是**配置文件** `hcu_adapter.yaml`
>（手册见 [config.md](config.md)，交互生成 `tools/hcu_config_wizard.py`）。环境
> 变量降级为**调试覆盖**：显式设置仍最高优先（一键临时回退不用改文件），但正式
> 部署零环境变量。新功能只加配置项，不加环境变量。

**三原则**（客户交付口径）：

1. **一个手册**：所有变量都在本文件，没有口口相传的隐藏开关。
2. **自动适应**：变量由平台探测（arch/wheel/参数）自动裁决，**正确部署零环境变量**。
3. **显式兼顾调试**：显式设置永远优先于自动规则；启动日志打印每个生效值及来源
   （`explicit` / `config` / `auto` / `default`），一行看清当前激活状态：
   ```
   HCU adapter env: HCU_AITER_AR=refused(old-aiter graph bug)(auto) HCU_MQA_EMU=1(explicit)
   ```

裁决顺序：**显式 env > 配置文件（hcu_adapter.yaml）> 自动规则 > 默认值**
（hcu_env.resolve 实现；配置文件机制见 [config.md](config.md)）。注册表在
`sgl_kernel/_adapters/hcu_env.py`（新增变量必须同时登记本手册）。算子后端的
正式选择走配置文件；运行时每算子实际谁在跑，看启动日志 `HCU adapter runtime:`
行或 `HCU_ADAPTER_DUMP`（规约见 [governance.md](governance.md) §4）。

## 全量变量表

| 变量 | 默认 | 自动规则 | 说明 |
|---|---|---|---|
| `HCU_DISABLE_OPS` | 空 | —（快关只应显式） | 快关总闸，逗号分隔：`aiter_ar`（拒用→NCCL 回退）、`aiter_fa`（调用时报错）。精度排查用，见 skills/…/kill-switches.md |
| `HCU_PLATFORM_SEMANTICS` | cuda | —（§0.5 拍板路线，platform_seam.py 桥） | sglang 平台判定语义：cuda=上游 `is_cuda/is_hip/is_cuda_alike/platforms` 统一回 CUDA（新模型天然走 CUDA 分支），外加 is_flashinfer_available=False、external-event 关、CUDA 版本特性门 False、FP8_E4M3_MAX 重算 fn(448)；硬件层（torch.version/triton/vllm/kernels 选核）保持 HIP 真值。hip=legacy 适配路线（降级调试 / A-B 对照） |
| `HCU_ADAPTER_DUMP` | 空 | —（排查只应显式） | 启动时落 runtime dump：每个 catalog 算子的活跃实现、安装来源（seam/rebind/text-patch）、env 生效值+来源。`=路径` 写 JSON 文件（归因脚本与配置级二分消费），`=1` 打进启动日志。规约见 docs/governance.md |
| `HCU_AITER_AR` | 0 | gfx93\* 且检测到**旧版** aiter（`aiter.ops.triton.custom_all_reduce`）→ 自动拒用（图重放损坏，1eb038d）；新版 `aiter.dist` GraphSafe 类已过全套压测 → 自动允许 | =1 显式夺回（压测先行：`pytest tests/vendor/test_allreduce.py`）。**custom AR 在位时本变量不参与**（sglang 直接路由到 `sgl_kernel.allreduce`，aiter AR 不会再被选中） |
| `HCU_TRITON_AR` | 1 | —（custom AR 的 sgl_kernel.allreduce 模块随包装载） | 自制 custom AR（graph 安全 TP all-reduce）：=1 路由 sglang CustomAllreduce 至自家实现（`sgl_kernel/_adapters/triton_ar.py` + `libhcu_ar.so`，信令+payload 走 host-shm）；=0 回退 aiter AR 自动规则（多卡 graph 会坏，须 `--disable-cuda-graph`）。门禁 `tests/tools/_test_hcu_ar2.py` |
| `HCU_TRITON_AR_SLOTS` | 空（按 WS 取 8/4/2） | — | custom AR 轮转槽数 S（槽复用由 DONE 标志守卫，槽越多容忍的乱序越深） |
| `HCU_TRITON_AR_BUDGET` | 100000000 | — | custom AR 每次自旋预算（PCIe 轮询次数，≈1.4µs/次；超出降级继续并计 diag 计数器——`TritonARState.read_sync_words()` 可读） |
| `HCU_TRITON_AR_MAX_KB` | 512 | —（实测交叉点：4/64/256KB 快 NCCL 2.6/2.1/1.5×，1MB 起 NCCL 快） | custom AR 尺寸上限：≤此值走 custom AR，更大 sglang 自动走 NCCL；0=不设限 |
| `HCU_MQA_EMU` | 0 | gfx93\* 自动开（aiter MQA 核硬断言 gfx942/950，gfx936 无 fp8、gfx938 暂仍被门挡）；DTK 放开门槛后 gfx938 分支转自动关 | paged/ragged MQA logits 的 torch 模拟；显式 0/1 覆盖 |
| `HCU_FA3_EMU` | 1 | — | fa3 attention backend 的 torch 模拟（`sgl_kernel.flash_attn`）：开=设 `SGLANG_USE_SGL_FA3_KERNEL=1` 并放行 `_is_fa3_supported`（HIP 上原样必 raise/落 fa2 坏 wheel）；=0 回上游行为。fp8 KV（descale）与 fp8-q 路径已含 |
| `HCU_FA_IMPL` | auto | — | fa3 注意力面档位：auto=宽窗 emu（契约全、graph-safe）|**triton=decode L=1 走自研 triton 分页核**（`sgl_kernel/paged_attn_triton.py`，fp32 累加，门禁 5/5 双卡型 ALL-PASS，decode 4.7× emu；宽窗仍 emu）|vendor=显式 opt-in 走 DTK flash_attn wheel（gfx936 wheel paged 内核真实 decode 词汤，2026-10-12 bw7，缺口清单待 DTK 修）|emu=恒模拟 |
| `HCU_DSA_PRESHUFFLE` | 1 | — | DSA aiter-preshuffle 路径；=0 回 legacy page_size=1 |
| `HCU_DSA_PAGE_SIZE` | 空 | CLI `--page-size` 等效优先 | DSA page_size 覆盖，须为 index_kpool 倍数 |
| `HCU_SLIMQUANT` | 1 | —（方法按 checkpoint 的 quantization_config 自选，非该格式则不激活） | slimquant_w4a8 注册；=0 隔离排查用 |
| `HCU_RMS_NORM_IMPL` | triton | —（triton 生产档；aiter 档已过 GPU parity 门禁可显式切换） | rms_norm 后端档位：triton\|aiter(rmsnorm_forward 桥)\|torch；fused_add_rmsnorm/gemma 变体恒走 triton(torch) |
| `HCU_SILU_MUL_IMPL` | auto | —（auto=triton；lightop 档过契约+图重放压测后 auto 转正） | 及格即转正不让客户改默认值；=lightop 显式启用 vendor 档（A/B：`tests/tools/_lightop_silu_ab.py`） |
| `HCU_DEEPEP_BRIDGE` | 1 | —（签名探测：仅当 deep_ep 是 DTK fork 形 quant_type 且缺 use_fp8 时才打补丁） | deep_ep API 桥：sglang v0.5.21 `low_latency_dispatch(use_fp8=)` 调用形 → DTK fork（`topk_weight` 位置参 + quant_type 0=bf16/2=E4M3FNUZ/3=UE8M0 映射，权重从调用帧兜取）；=0 关闭 |
| `HCU_DEEPGEMM_BRIDGE` | 0 | —（EP 启动显式 =1） | DTK `deepgemm`（连写包名）别名成 `deep_gemm` + wrapper `ENABLE_JIT_DEEPGEMM` 翻转（HIP 平台门原样恒 False），解锁 deepep_ll/normal MoE runner——bf16 grouped GEMM（masked/contiguous）上游同名已验；fp8 的 scale 布局转换在 shim 里响亮 NotImplementedError（未验不瞎算）。EP 服务启动脚本必带 `=1` + `SGLANG_DEEPEP_BF16_DISPATCH=1` |
| `HCU_TRITON_CONTIG` | 1 | —（随 DEEPGEMM 桥生效） | DeepEP normal/auto prefill 的 contiguous grouped GEMM 用 adapter triton 内核（`deep_gemm_contig_triton.py`，原生 [E,N,K] 权重 + mk-alignment get/set API），替代坏掉的 DTK w16a16 contiguous 内核（2D VMFault/3D 数字错）；=0 退回 DTK 原核仅作对照（会崩）。门禁 `tests/tools/_test_contig_triton.py` |
| `HCU_FUSED_MOE_IMPL` | auto | auto / aiter / triton | fused MoE（standard dispatch，triton runner 入口 `fused_experts`）先试 DTK aiter `aiter_moe`：块量化(block_shape=[128,128])走 ASM 调优表 `tuned_fmoe_asm_w8a8_group.csv`，通道量化(block_shape=None)走 moe_c 配置；形状不在调优表 → 自动回 stock triton 内核（每形状 log 一次）。`aiter`=回退时告警；`triton`=关闭。仅 w8a8 + bf16 激活 + 无 bias/no_combine；deepep runner 不受影响。门禁 `tests/tools/_test_aiter_moe_gate.py`（nmz18 gfx938 rel 0.02-0.04 PASS） |
| `HCU_PREFILL_CP` | 0 | —（显式 =1 开启） | prefill 上下文并行（`--enable-prefill-cp --attn-cp-size N --cp-strategy zigzag`）HIP 平台门 + 两处语义补丁：①rope 融合守卫——CP 分块 extend 时本地 token 数 < 全序列 slot 数，HIP 专属融合 rope+KV 写核断言 t==t_slot，slot 数≠token 数即自动降级纯 rope（decode 不受影响）；②`enable_fused_set_kv_buffer` 判定词 HIP 臂补 `not is_cp_active`（上游漏了，CUDA 臂有）——否则模型以为融合 rope 会写 KV 而传 save_kv_cache=False，结果没人写池（输出乱码）。nmz26 Qwen3-30B-A3B attn-cp=2 zigzag 已验（输出与 TP2 基线对齐） |
| `HCU_PREFILL_CP_DEBUG` | 空 | 1 | CP 排查插桩：zigzag 策略各阶段（shard/gather/materialize）校验和日志，定位首个数值坏环节 |
| `HCU_FA_EMU_DEBUG` | 空 | 1 | fa3 emu 每次调用打 q 形状/cu_seqlens/cache_seqlens/校验和（graph capture 期间自动静默） |
| `HCU_AITER_MOE_DEBUG` | 空 | 1 | aiter fused-MoE 桥诊断：=1 时每个早退守卫/命中/回退决策打 warning 日志（定位某模型为何没走 aiter） |
| `HCU_DUMP_LAYERS` | 0 | — | 调试：层 residual 流 dump（`_PATH`/`_T`/`_TOTAL` 配套） |
| `HCU_DUMP_INNER` | 0 | — | 调试：L0/L1 xa/xb 张量 |
| `HCU_DUMP_TOPK` | 0 | — | 调试：DSA topk 样本 |
| `HCU_DUMP_MQA` | 0 | — | 调试：MQA emu 调用日志 |

## 客户部署建议（一句话）

什么都不设。需要调试/隔离时按上表显式设置，并看启动日志的
`HCU adapter env:` 行确认生效来源。

## 给维护者

- 新增变量：`hcu_env.py::REGISTRY` 登记（env/default/auto/about 四项，
  auto 规则写成无副作用的探测函数）→ 本手册加行 → 若是高危算子快关，
  同时走 `kill_switch.py` 注册 + gate 测试。
- 自动规则必须**可离线判定**（arch/wheel 探测，不发网络请求），且失败时
  静默回退默认值（resolve 已兜底）。
- `resolve(name)` 返回 `(value, source)`；行为接线处禁止再裸读
  `os.environ`，统一走注册表，保证日志汇总不漏。
