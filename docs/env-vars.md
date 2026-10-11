# HCU Adapter 环境变量手册

**三原则**（客户交付口径）：

1. **一个手册**：所有变量都在本文件，没有口口相传的隐藏开关。
2. **自动适应**：变量由平台探测（arch/wheel/参数）自动裁决，**正确部署零环境变量**。
3. **显式兼顾调试**：显式设置永远优先于自动规则；启动日志打印每个生效值及来源
   （`explicit` / `auto` / `default`），一行看清当前激活状态：
   ```
   HCU adapter env: HCU_AITER_AR=refused(old-aiter graph bug)(auto) HCU_MQA_EMU=1(explicit)
   ```

裁决顺序：**显式 env > 自动规则 > 默认值**。注册表在
`sgl_kernel/_adapters/hcu_env.py`（新增变量必须同时登记本手册）。

## 全量变量表

| 变量 | 默认 | 自动规则 | 说明 |
|---|---|---|---|
| `HCU_DISABLE_OPS` | 空 | —（快关只应显式） | 快关总闸，逗号分隔：`aiter_ar`（拒用→NCCL 回退）、`aiter_fa`（调用时报错）。精度排查用，见 skills/…/kill-switches.md |
| `HCU_AITER_AR` | 0 | gfx93\* 且检测到**旧版** aiter（`aiter.ops.triton.custom_all_reduce`）→ 自动拒用（图重放损坏，1eb038d）；新版 `aiter.dist` GraphSafe 类已过全套压测 → 自动允许 | =1 显式夺回（压测先行：`pytest tests/vendor/test_allreduce.py`）。**custom AR 在位时本变量不参与**（sglang 直接路由到 `sgl_kernel.allreduce`，aiter AR 不会再被选中） |
| `HCU_TRITON_AR` | 1 | —（custom AR 的 sgl_kernel.allreduce 模块随包装载） | 自制 custom AR（graph 安全 TP all-reduce）：=1 路由 sglang CustomAllreduce 至自家实现（`sgl_kernel/_adapters/triton_ar.py` + `libhcu_ar.so`，信令+payload 走 host-shm）；=0 回退 aiter AR 自动规则（多卡 graph 会坏，须 `--disable-cuda-graph`）。门禁 `tests/tools/_test_hcu_ar2.py` |
| `HCU_TRITON_AR_SLOTS` | 空（按 WS 取 8/4/2） | — | custom AR 轮转槽数 S（槽复用由 DONE 标志守卫，槽越多容忍的乱序越深） |
| `HCU_TRITON_AR_BUDGET` | 100000000 | — | custom AR 每次自旋预算（PCIe 轮询次数，≈1.4µs/次；超出降级继续并计 diag 计数器——`TritonARState.read_sync_words()` 可读） |
| `HCU_TRITON_AR_MAX_KB` | 512 | —（实测交叉点：4/64/256KB 快 NCCL 2.6/2.1/1.5×，1MB 起 NCCL 快） | custom AR 尺寸上限：≤此值走 custom AR，更大 sglang 自动走 NCCL；0=不设限 |
| `HCU_MQA_EMU` | 0 | gfx93\* 自动开（aiter MQA 核硬断言 gfx942/950，gfx936 无 fp8、gfx938 暂仍被门挡）；DTK 放开门槛后 gfx938 分支转自动关 | paged/ragged MQA logits 的 torch 模拟；显式 0/1 覆盖 |
| `HCU_FA3_EMU` | 1 | — | fa3 attention backend 的 torch 模拟（`sgl_kernel.flash_attn`）：开=设 `SGLANG_USE_SGL_FA3_KERNEL=1` 并放行 `_is_fa3_supported`（HIP 上原样必 raise/落 fa2 坏 wheel）；=0 回上游行为。fp8 KV（descale）与 fp8-q 路径已含 |
| `HCU_DSA_PRESHUFFLE` | 1 | — | DSA aiter-preshuffle 路径；=0 回 legacy page_size=1 |
| `HCU_DSA_PAGE_SIZE` | 空 | CLI `--page-size` 等效优先 | DSA page_size 覆盖，须为 index_kpool 倍数 |
| `HCU_SLIMQUANT` | 1 | —（方法按 checkpoint 的 quantization_config 自选，非该格式则不激活） | slimquant_w4a8 注册；=0 隔离排查用 |
| `HCU_SILU_MUL_IMPL` | auto | —（auto=triton；lightop 档过契约+图重放压测后 auto 转正） | 及格即转正不让客户改默认值；=lightop 显式启用 vendor 档（A/B：`tests/tools/_lightop_silu_ab.py`） |
| `HCU_DEEPEP_BRIDGE` | 1 | —（签名探测：仅当 deep_ep 是 DTK fork 形 quant_type 且缺 use_fp8 时才打补丁） | deep_ep API 桥：sglang v0.5.21 `low_latency_dispatch(use_fp8=)` 调用形 → DTK fork（`topk_weight` 位置参 + quant_type 0=bf16/2=E4M3FNUZ/3=UE8M0 映射，权重从调用帧兜取）；=0 关闭 |
| `HCU_DEEPGEMM_BRIDGE` | 0 | —（EP 启动显式 =1） | DTK `deepgemm`（连写包名）别名成 `deep_gemm` + wrapper `ENABLE_JIT_DEEPGEMM` 翻转（HIP 平台门原样恒 False），解锁 deepep_ll/normal MoE runner——bf16 grouped GEMM（masked/contiguous）上游同名已验；fp8 的 scale 布局转换在 shim 里响亮 NotImplementedError（未验不瞎算）。EP 服务启动脚本必带 `=1` + `SGLANG_DEEPEP_BF16_DISPATCH=1` |
| `HCU_TRITON_CONTIG` | 1 | —（随 DEEPGEMM 桥生效） | DeepEP normal/auto prefill 的 contiguous grouped GEMM 用 adapter triton 内核（`deep_gemm_contig_triton.py`，原生 [E,N,K] 权重 + mk-alignment get/set API），替代坏掉的 DTK w16a16 contiguous 内核（2D VMFault/3D 数字错）；=0 退回 DTK 原核仅作对照（会崩）。门禁 `tests/tools/_test_contig_triton.py` |
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
