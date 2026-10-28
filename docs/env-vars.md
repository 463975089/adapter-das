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
`hcu_adapter/hcu_env.py`（新增变量必须同时登记本手册）。算子后端的
正式选择走配置文件；运行时每算子实际谁在跑，看启动日志 `HCU adapter runtime:`
行或 `HCU_ADAPTER_DUMP`（规约见 [governance.md](governance.md) §4）。

## 全量变量表

| 变量 | 默认 | 自动规则 | 说明 |
|---|---|---|---|
| `HCU_ADAPTER_HOST` | auto | 启动参数 / 已加载模块 / 单一可用宿主；双包无启动线索时保持 sglang 兼容 | 自动分发宿主桥；`sglang`/`vllm` 可用于诊断时显式固定，`none` 禁用自动桥。父进程的自动结果经 adapter 私有键传给 spawn 子进程，正常部署无需设置。 |
| `HCU_DISABLE_OPS` | 空 | —（快关只应显式） | 快关总闸，逗号分隔：`aiter_ar`（拒用→NCCL 回退）、`aiter_fa`（调用时报错）。精度排查用，见 skills/…/kill-switches.md |
| `HCU_PLATFORM_SEMANTICS` | cuda | —（§0.5 拍板路线，platform_seam.py 桥） | sglang 平台判定语义：cuda=上游 `is_cuda/is_hip/is_cuda_alike/platforms` 统一回 CUDA（新模型天然走 CUDA 分支），外加 is_flashinfer_available=False、external-event 关、CUDA 版本特性门 False、FP8_E4M3_MAX 重算 fn(448)；硬件层（torch.version/triton/vllm/kernels 选核）保持 HIP 真值。hip=legacy 适配路线（降级调试 / A-B 对照） |
| `HCU_ADAPTER_DUMP` | 空 | —（排查只应显式） | 启动时落 runtime dump：每个 catalog 算子的活跃实现、安装来源（seam/rebind/text-patch）、env 生效值+来源。`=路径` 写 JSON 文件（归因脚本与配置级二分消费），`=1` 打进启动日志。规约见 docs/governance.md |
| `HCU_JIT_PROGRESS_LOG` | `/tmp/hcu_jit_progress.log` | l1_gpu_gate.sh 每轮自动设 `/tmp/hcu_jit_progress.<pid>.log`（并发 gate 不互串） | JIT 逐颗编译进度期刊（`=""` 关闭）：`hcu_adapter/jit_build_log.py` 包上游 `ninja.build`，每颗**真实**编译（缓存命中不进此路径）向 stderr+journal 各出 `compiling/built/FAILED` 一行（FAILED 后调用侧重试会再记一对）；gate 心跳每 60s tail 此文件——journal 持续推进=在编译、冻结=楔死，日志可辨（2026-10-20） |
| `HCU_NCCL_TOPO_GUARD` | **1（默认）** | 失配才动（box16 2026-10-25） | NCCL_TOPO_FILE 卡数失配守卫（`nccl_topo_guard.py`）：镜像预设全卡拓扑 XML 而可见卡为子集（`HIP_VISIBLE_DEVICES=0-3` 等）时，NCCL 首个集合通信即致命 XGMI 错（could not find GPU 5000，box16 4-proc 实锤）。守卫数 XML `<gpu>` 节点 vs torch 可见卡数：**失配→清空该 env**（NCCL 回退 sysfs 自侦，任意子集皆正确）+一行日志；匹配→保留；判不了→保留。=0 关闭 |
| `HCU_JIT_WATCHDOG` | **1（默认）** | 冷缓存才动（box16 2026-10-25） | 冷 JIT watchdog 地板（`watchdog_cold.py`）：冷容器首捕获 8 rank 全量 JIT 编译风暴超 sglang 默认 watchdog 300s→SIGQUIT 杀树（box16 冷启 16 并发首请求实锤）。TRITON_CACHE_DIR/TORCH_EXTENSIONS_DIR 文件数均 <200 判冷；冷且用户未显式给 `--watchdog-timeout`（值=300 默认）时抬地板到 **1500s**（box16 实证装得下编译风暴）；**显式值恒赢**；种子 warm 盒不受影响。=0 关闭 |
| `HCU_TVMFFI_WARMUP` | **1（默认）** | 冷缓存才动（nmz26 2026-10-26） | tvm-ffi dlpack 首编自递归 breaker（`tvmffi_warmup.py`）：冷缓存下 tvm_ffi 首次 import 派构建子进程，子进程继承 PYTHONPATH 重进 sitecustomize → 桥 eager import 链（dsa_topk_v2→xgrammar→tvm_ffi）再派子进程 → 自递归堆积（bootstrap §3 的 122 进程/50 分钟案；nmz26 dsv41f 冷容器 2026-10-26 复现，签名=每 ~30s 孵一个新 python3、各烧 ~30s import 后阻塞）。sitecustomize 前部（先于 dsa_topk_v2 等重桥）探温 `~/.cache/tvm-ffi/**.so`，冷则 flock 排他（TP spawn 风暴单构建者）+ 剥 PYTHONPATH 亲跑 tvm_ffi 自带构建脚本（bootstrap §3 自动化）；加载器仍要自派子进程时窗口内 Popen 包装注入干净 env 物理断路。=0 关闭（调试层） |
| `HCU_W4A8_MOE` | **w4a16（默认）** | w4a8 检查点才动（nmz26 2026-10-26） | CT W4A8 MoE scheme 路由（`ct_w4a8_moe.py`）：DAS 转换的 w4a8 检查点（如 DeepSeek-V4.1-Flash-Channel-INT4-w4a8）config_groups 全 regex targets，`target_scheme_map` 无 "Linear" 桶，上游 `CompressedTensorsW4AFP8MoE.__init__` 构造期 KeyError('Linear')（v0522 w4a8_fp8_moe.py:87，模型建不过，权重未加载）。**w4a16（默认）**=wrap config `__init__` 把 int4 组 `input_activations` 置 None → 上游自然走 `_is_hip` 的 WNA16TritonMoE 臂（box16 已验证路径，10.6 channel 改写收口），激活 int8 量化被忽略（MoE 激活跑 bf16，数值更保真；上游 nvfp4 w4a4→w4a16 同款降级模式）；dense int8/int8-act 组不动。**heal**=只作用域内补 "Linear" 桶（取该层 regex 匹配到的 scheme_dict，用毕即删、不污染 dense 组），保留 stock W4AFP8 语义——当前会响亮撞 CUTLASS W4A8 缺核，为将来 W4A8 HIP 内核（triton int8 dot / vendor 档）备的路。**0**=还原 stock KeyError |
| `HCU_DSV4_WQKV_SCALE` | **1（默认）** | dsv4 融合 wqkv_a 且 CT int8 scale 才动（nmz16 2026-10-26） | DSV4 wqkv_a CT-int8 scale 融合针脚（`dsv4_wqkv_scale_patch.py`）：v0522 load_weights 融合分支抓 `.weight`/`.weight_scale_inv`/`.qweight` 但漏 CT INT8 per-channel `.weight_scale`——DAS w4a8 检查点的 wq_a/wkv int8 scale 被 "not found in params_dict" 丢弃，`wqkv_a.weight_scale`（W8A8Int8 create_weights 所建）拿不到数据 = 注意力投影带未初始化 scale 反量化（**boot 正常、数值垃圾**；nmz16 2026-10-26 shard 加载期 warning 定位）。针脚幂等补两个 endswith（q/kv scale 走同款 dim-0 融合 helper）；**关融合不是绕法**（WeightsMapper 后的组 targets 只认融合名 wqkv_a，未融合名无 scheme=int8 权重 cast bf16 更坏）。上游各平台同病，PR 前由本针脚顶住。=0 跳过 |
| `HCU_WNA16_CHANNEL_MOE` | **1（默认）** | channel-int4 才动（box16 2026-10-25） | channel-wise int4 MoE 改写（`wna16_channel_moe.py`）：compressed-tensors pack-quantized `strategy=channel`（group=-1，如 DeepSeek-V4.1-Flash-INT4）经 WNA16TritonMoE 给出 `block_shape=[0,-1]`，树上 triton invoke 的 gptq 反量化分支只在 `block_shape[1]>0` 走，u8 packed 权重直落普通 kernel 的 `tl.dot(bf16,u8)` 编译期炸（首中=decode 图捕获，服务起不来）。channel≡整行一组的退化 group-wise：改写 `block_shape=[0,max(K_up,K_down)]`（两 GEMM cdiv==1，scheme 转置后的 [E,N,1] scale 直配）→ 数值等价、零权重转换、capture 安全。仅命中 `use_int4_w4a16` 且 group≤0 的调用面；fp8/int8/分组 int4/bf16 不动；=0 还原 stock |
| `HCU_AITER_AR` | auto | gfx93\* 且检测到**旧版** aiter（`aiter.ops.triton.custom_all_reduce`）→ 自动拒用（图重放损坏，1eb038d）；新版 `aiter.dist` GraphSafe 类已过全套压测 → 自动允许 | TP all-reduce 的 aiter 档开关（2026-10-18 路由更正后语义）：auto（默认）随检测；=1 显式夺回（压测先行：`pytest tests/vendor/test_allreduce.py`）；=0 显式拒用 → **纯 NCCL fallback**。（自家 custom AR 已于 2026-10-19 除役删除：R-01 定罪 + R-13 #1 改判废弃，aiter 压测全绿后无需修复，`HCU_TRITON_AR*` 四键随之撤销） |
| `HCU_SPEC_LOC_SAFE` | **1（默认）** | 化 | spec-decode KV 位置安全路由：把 sglang `mem_cache/allocation.get_last_loc` 重绑纯 torch 变体——stock 在 is_cuda 平台语义下走 `get_last_loc_triton`（树内注释自证 HIP 混合位宽误编译，EAGLE+page_size>1 产生越界 last_loc）。2026-10-17 实测：GLM-5.3 MTP 下能让 Berlin 转逐次确定，但**不治本**（accept 仍 0.02、输出仍错——结构性缺口在 das 的 srt spec 补丁族，见 plugins.md P1）。=0 还原 stock 分发 |
| `HCU_MQA_LOGITS_IMPL` | **triton（默认，2026-10-18 R-13 #4 晋级）** | 化 | DSA indexer paged-MQA 实现档：**triton**=自研 triton 档（`mqa_logits_triton.py`）——**production 默认**：臂随 `HCU_DSA_PRESHUFFLE` 同一开关（默认交错=生产布局）；fp8-e4m3fn 位级解码+fp32 累加+页程序早退（64K 窗口下 ctx 外页两次标量载即返）；对金标准**精确**（rel=0.0、topk 全对，SetKAndS 交错/行主序/fused_store 三 writer 臂）、解码形状 N=8/窗口 64K **13.2× 快于 emu-vec**（0.19 vs 2.52ms）；R-02 validated → serving A/B（四门绿+马拉松 0/20+长文对，解码 +9% 短/+4% 长）→ 2026-10-18 R-13 #4 晋级翻默认；哨兵 tests/kernels/test_mqa_triton_gfx938.py（含修复后 emu 交叉验证）；**vec（=显式 opt-out）**=全窗口单遍双 bmm（gather→`sT=kg@qᵀ`→`row=sT@wᵀ`→×sc→mask；H 收缩在 GEMM fp32 累加器内完成，**大操作数永不转置、(N,H,T) 永不 fp32 物化**——两者在 DTK 图 replay 下各有 ~124ms/核崖（eager/standalone 小图无恙；loop 按 2048 tile 走小张量快路径故无恙）；N≤`HCU_MQA_VEC_MAXROWS`(32) 才走，prefill 大 N 自动回落 loop）；**loop**=(64,2048) 双层 tile 循环参考实现（曾占 decode 图节点 23.9k 的 72%，=loop 可回归对照）。战果（2026-10-17）：图节点 -84%、hipGraphLaunch API p50 540→21.6ms、单流 2.3→3.50 tok/s。**2026-10-18 R-02：vec/loop 池解包修正为页级+交错感知**——writer 普查（writer census）：GLM-5.3 走 IndexerKPool（kpool=4，独立类、不走 base store/fused_store），池由 kpool 写族（`kpool_softmax_rotate_write_cache` 等，stock triton gfx938 div-4 bug→`kpool_write_emu`）+SetKAndS 在 adapter 强制的 preshuffle 门下写入=**16-tile 交错 k+页尾 fp32 scale**；prefill gather（`_kpool_cache_k_offsets`）同序读**一致无辜**；旧行级 132 字节分裂使 k 每 slot 错位+scale 读到 k 字节——**生产解码 DSA top-k 一直确定性错误**（真实 writer 池对拍 rel=nan；锚点对 topk 不敏感未暴露）；`HCU_DSA_PRESHUFFLE=0` 切行主序臂（=kpool=1/fused_store 世界；PS=1 字节不变））；**vendor**（R-02 终裁取证档）=deepgemm 轮 `paged_mqa_logits` 直调（mqa_vendor.py 桥）：布局=行主序 k+页尾 64×fp32 scale=**kpool=1 世界同构**（kpool 模型需池重排）；**轮子 bug 定罪——q/k 负值 fp8 字节被钳为 0**（logits=Σw·max(q,0)·max(k,0)·s 四位小数拟合，真实数据 rel~3 垃圾），待 DTK 修复（deepgemm 2.1.0+dtk2604）；catalog 档 broken 选配即拒；双事实哨兵 tests/kernels/test_mqa_vendor_gfx938.py（正数据精确+带符号失配两钉） |
| `HCU_MQA_EMU` | 0 | gfx93\* 自动开（aiter MQA 核硬断言 gfx942/950，gfx936 无 fp8、gfx938 暂仍被门挡）；DTK 放开门槛后 gfx938 分支转自动关 | paged/ragged MQA logits 的 torch 模拟；显式 0/1 覆盖 |
| `HCU_FA3_EMU` | 1 | — | fa3 attention backend 的 torch 模拟（`sgl_kernel.flash_attn`）：开=设 `SGLANG_USE_SGL_FA3_KERNEL=1` 并放行 `_is_fa3_supported`（HIP 上原样必 raise/落 fa2 坏 wheel）；=0 回上游行为。fp8 KV（descale）与 fp8-q 路径已含 |
| `HCU_FA_IMPL` | auto | — | fa3 注意力面档位：auto=宽窗 emu（契约全、graph-safe）|**triton=分页 kvcache 面（decode L=1 + 宽窗 L>1 bottom-right causal）走自研 triton 核**（`sgl_kernel/paged_attn_triton.py`，fp32 累加、页号 int64，门禁 10/10 双卡型 ALL-PASS，decode 4.7×/prefill ~7× emu）**+ varlen 预填充面（ragged dense K/V：extend/target-verify/kvcache 门面 varlen 形态）走自研 triton 核**（`sgl_kernel/fa_varlen_triton.py`，fp32 累加 + ieee dot、token 号 int64、swa window + fp8-KV descale、静态 grid=max_seqlen_q 图安全；Dv≠D/fp8-q/softcap/qv/sinks 面仍 emu；哨兵 tests/kernels/test_fa_varlen_triton.py 三段式+capture-replay+零长序列 pit；BW1101 对拍，性能数字待 BW1102 复核）|vendor=显式 opt-in 走 DTK flash_attn wheel（gfx936 wheel paged 内核真实 decode 词汤，2026-10-12 bw7，缺口清单待 DTK 修）|emu=恒模拟。**2026-10-19 serving A/B 双轮**：第一轮 ≤8k 平价、揪出 ops 重绑丢 q_v 包参 boot 杀手（2759711）、11k 红牌（2/2 毙 vs auto 3/3 过）；红牌根因=emu (maxL×C) 板无 q 行分块，**c59135b q 行子分块全臂击穿**（peak 22→6.8GiB、一钉 test_extend_face_peak_bounded、fp64 金标 0.0046、47 门绿）；修复后 triton 臂终验 11k×3 绿（11.3s=auto）+全电池绿+锚点与 auto 臂逐字一致+LONGCTX-6k **28.4→3.1s（~9×）**+解码不变；档位守 candidate（glm53f q_v 面恒 emu，翻默认无感，待行使 triton 核的模型过 A/B） |
| `HCU_LINEAR_ATTN_GDN` | auto | — | GDN/gated-delta-rule 线性注意力档位：auto|triton=上游 triton fla（现状）；**aiter=aiter CK 核路由**（fwd_h 状态递推 blockdim64 + packed decode 同契约直换绑，K=V=128；门禁 6/6 双卡型：gfx936 逐位、gfx938 深积累 fp32 次序差带宽内；Qwen3.5-27B TP4 e2e 四锚点全对；fwd_h 稳态 1.3-1.9×，decode 非捕获态 ~1.0×）。注：SM90 上游会把 GDN prefill 默认翻到 flashinfer——adapter 已 deny `flashinfer.gdn_prefill/gdn_decode/gdn_kernels` 让 default 回 triton。**GDN e2e 配方**：`--disable-radix-cache`（bw7 cargo 1.75 缺口）+ **`--cuda-graph-bs-prefill 4 8 32 128 512 1024 2048 4096 8192`**（prefill graph 捕获 OOM 根因=默认 58 桶表 ×GDN 状态池预算不够，显式稀疏桶列即解，勿用 --disable-prefill-cuda-graph——9 桶全 graph 配置实测起服+锚点对；mem 0.45 是混合池下限附近，0.42 会配不出 mamba 池） |
| `HCU_DSA_PRESHUFFLE` | 1 | — | DSA aiter-preshuffle 路径；=0 回 legacy page_size=1 |
| `HCU_DSA_PAGE_SIZE` | 空 | CLI `--page-size` 等效优先 | DSA page_size 覆盖，须为 index_kpool 倍数 |
| `HCU_SLIMQUANT` | 1 | —（方法按 checkpoint 的 quantization_config 自选，非该格式则不激活） | slimquant_w4a8 注册；=0 隔离排查用 |
| `HCU_RMS_NORM_IMPL` | triton | —（triton 生产档；aiter 档已过 GPU parity 门禁可显式切换） | rms_norm 后端档位：triton\|aiter(rmsnorm_forward 桥)\|torch；fused_add_rmsnorm/gemma 变体恒走 triton(torch) |
| `HCU_SILU_MUL_IMPL` | auto | —（auto=triton；lightop 档过契约+图重放压测后 auto 转正） | 及格即转正不让客户改默认值；=lightop 显式启用 vendor 档（A/B：`tests/tools/_lightop_silu_ab.py`） |
| `HCU_DEEPEP_BRIDGE` | 1 | —（签名探测：仅当 deep_ep 是 DTK fork 形 quant_type 且缺 use_fp8 时才打补丁） | deep_ep API 桥：sglang v0.5.21 `low_latency_dispatch(use_fp8=)` 调用形 → DTK fork（`topk_weight` 位置参 + quant_type 0=bf16/2=E4M3FNUZ/3=UE8M0 映射，权重从调用帧兜取）；=0 关闭 |
| `HCU_DEEPGEMM_BRIDGE` | 0 | —（EP 启动显式 =1） | DTK `deepgemm`（连写包名）别名成 `deep_gemm` + wrapper `ENABLE_JIT_DEEPGEMM` 翻转（HIP 平台门原样恒 False），解锁 deepep_ll/normal MoE runner——bf16 grouped GEMM（masked/contiguous）上游同名已验；fp8 的 scale 布局转换在 shim 里响亮 NotImplementedError（未验不瞎算）。EP 服务启动脚本必带 `=1` + `SGLANG_DEEPEP_BF16_DISPATCH=1` |
| `HCU_TRITON_CONTIG` | 1 | —（随 DEEPGEMM 桥生效） | DeepEP normal/auto prefill 的 contiguous grouped GEMM 用 adapter triton 内核（`deep_gemm_contig_triton.py`，原生 [E,N,K] 权重 + mk-alignment get/set API），替代坏掉的 DTK w16a16 contiguous 内核（2D VMFault/3D 数字错）；=0 退回 DTK 原核仅作对照（会崩）。门禁 `tests/tools/_test_contig_triton.py` |
| `HCU_FUSED_MOE_IMPL` | auto | auto / aiter / triton | fused MoE（standard dispatch，triton runner 入口 `fused_experts`）先试 DTK aiter `aiter_moe`：块量化(block_shape=[128,128])走 ASM 调优表 `tuned_fmoe_asm_w8a8_group.csv`，通道量化(block_shape=None)走 moe_c 配置；形状不在调优表 → 自动回 stock triton 内核（每形状 log 一次）。`aiter`=回退时告警；`triton`=关闭。仅 w8a8 + bf16 激活 + 无 bias/no_combine；deepep runner 不受影响。门禁 `tests/tools/_test_aiter_moe_gate.py`（nmz18 gfx938 rel 0.02-0.04 PASS） |
| `HCU_W4A16_MOE_IMPL` | auto | auto / triton / moe_c / deepgemm | W4A16 MoE（moe_wna16，AWQ/GPTQ-int4 线）专家 GEMM 档：auto=现答 triton（stock MoeWNA16 臂不变）；**moe_c**=aiter 轮 moe_c w4a16 marlin 厂商 GEMM 桥（`hcu_adapter/moe_c_w4a16.py`，box16 dsv41f-INT4 2026-10-26：aiter 对齐→厂商 GEMM1→dsv4 精确 clamp-silu（非 aiter 的 situ softcap）→厂商 GEMM2→sum；探针 parity med 0.0055、M16 全层 1161µs vs triton 8100µs（7×，40 层 ≈ −280ms/步）、capture 安全（无宿主同步、prep 落在 warmup）；守卫=g32 网格（channel 按值复制展开）、gemm1 (K/32)%16==0、N%256、I≥256；轮 gemm2 有 K 律 (K/32)%16==0——I 非 512 倍数时 K 补零 nibble 8 到 512 倍数（2304→2560，数学等价））；**deepgemm**=R-05 桥（`hcu_adapter/w4a16_moe.py`，DTK 轮 masked w4a16 grouped C++ gemm 作专家 GEMM，自管 dispatch/silu/combine 小管线；eager-only v1=捕获/inplace 恒回 triton；接手守卫=无零点对称臂 (q−8)·s、g∈{32,64,128}（载入期一次性扩 32 格）、K%128/I%128/2I%16、qweight=轮 b 格式零转换（marlin repack+scale 展开缓存于权重张量））——链在同一 fused_experts 钩面（moe_c 最外 10.7 → channel 改写 10.6 → deepgemm 10.5 → aiter 10 → stock）；deepgemm 探针 recon rel=0.0022 |
| `HCU_PREFILL_CP` | 0 | —（显式 =1 开启） | prefill 上下文并行（`--enable-prefill-cp --attn-cp-size N --cp-strategy zigzag`）HIP 平台门 + 两处语义补丁：①rope 融合守卫——CP 分块 extend 时本地 token 数 < 全序列 slot 数，HIP 专属融合 rope+KV 写核断言 t==t_slot，slot 数≠token 数即自动降级纯 rope（decode 不受影响）；②`enable_fused_set_kv_buffer` 判定词 HIP 臂补 `not is_cp_active`（上游漏了，CUDA 臂有）——否则模型以为融合 rope 会写 KV 而传 save_kv_cache=False，结果没人写池（输出乱码）。nmz26 Qwen3-30B-A3B attn-cp=2 zigzag 已验（输出与 TP2 基线对齐） |
| `HCU_PREFILL_CP_DEBUG` | 空 | 1 | CP 排查插桩：zigzag 策略各阶段（shard/gather/materialize）校验和日志，定位首个数值坏环节 |
| `HCU_FA_EMU_DEBUG` | 空 | 1 | fa3 emu 每次调用打 q 形状/cu_seqlens/cache_seqlens/校验和（graph capture 期间自动静默） |
| `HCU_AITER_MOE_DEBUG` | 空 | 1 | aiter fused-MoE 桥诊断：=1 时每个早退守卫/命中/回退决策打 warning 日志（定位某模型为何没走 aiter） |
| `HCU_DUMP_LAYERS` | 0 | — | 调试：层 residual 流 dump（`HCU_DUMP_LAYERS_PATH`/`HCU_DUMP_LAYERS_T`/`HCU_DUMP_LAYERS_TOTAL` 落盘与窗口，`HCU_DUMP_LAYERS_ONCE`/`HCU_DUMP_LAYERS_FIRES`/`HCU_DUMP_LAYERS_SAVE` 触发与轮次配置） |
| `HCU_DUMP_INNER` | 0 | — | 调试：L0/L1 xa/xb 张量 |
| `HCU_DUMP_TOPK` | 0 | — | 调试：DSA topk 样本 |
| `HCU_DUMP_MQA` | 0 | — | 调试：MQA emu 调用日志 |
| `HCU_DUMP_INT8` | 0 | — | 调试：serve 态 W8A8-Int8 apply 审计——=1 包 `CompressedTensorsW8A8Int8.apply_weights`，前 N 次调用用纯 torch（per-token 反量化 × channel 反量化权重）重算并记 cos/maxerr 到 /tmp/int8_apply_dump.log（独立 GEMM 探针全过而 serve 端乱码时抓真实进程内调用链） |
| `HCU_DUMP_MHC` | 0 | — | 调试：GLM-5.3-Flash mHC 边界 dump（2026-10-16 定罪役仪器）——=1 包装 mhc_post 融合调用落盘 /tmp/hcu_mhcpost_NNN.pt（`HCU_DUMP_MHC_MAX` 限次）；与 `_mhc_post_torch` 参考对拍定位 z 偏移 |
| `HCU_LAYER_AUDIT` | 0 | — | 调试：逐层前向审计——=1 按类挂 forward 计时/形状日志到 `HCU_LAYER_AUDIT_PATH`（默认 /root/logs/layer_audit.log.<pid>）；`HCU_LAYER_AUDIT_CLASSES` 逗号类名过滤（默认 DecoderLayer,LinearAttention,GatedDeltaNet,SparseAttention,MoeBlock,MLP,Glm5NextModel） |
| `HCU_SWIGLU_CLAMP` | 1 | — | 行为：DSV4 swiglu-clamp MoE 激活路由到 adapter triton（silu_and_mul_clamp，上游无 HIP 臂）——默认开；=0 退回上游路径 |
| `SGLANG_USE_SGL_FA3_KERNEL` | 1（`HCU_FA3_EMU` 生效时 setdefault） | — | 上游原生 env（fa3 可用性旗标）：fa3-emu 桥生效时 setdefault=1 并把 `_is_fa3_supported` 短路为 True，让上游按 fa3 语义调度、实际执行走 emu |
| `VLLM_PLUGINS` | none（sitecustomize setdefault） | — | 上游原生 env：vllm 平台插件 allowlist。unset=None=load-all 且空串值有 import 期竞态（见 sitecustomize 门内注释），hip 门内 setdefault 非空哨兵 none（过滤一切插件）；vllm_hcu 平台插件在本栈会挂死进程（nmz26 dtk26041 实测），sglang 平台经 platform_seam 解析；操作员显式导出永远赢 |
| `HCU_QUANT_IMPL` | auto | 化 | per-token fp8 量化族 vendor 档（R-09③，quant_aiter 桥）：auto=hip+aiter 轮可导入则 aiter——**rowwise 包装 `sglang_per_token_quant_fp8` 在 HIP 上游本就 NameError（`sgl_per_token_quant_fp8` 不存在，group 入口整行快捷分支同踩）=行为还原**；group plain 臂（row-major fp32 尺度）15.1×（4096×8192，尺度与金标准逐位一致、q RTN 命中 99.8%）；scaled_fp8_quant 两 dynamic natives=上游 SGLANG_USE_AITER 同款调用（**static 恒原生**：aiter 轮 kernel 无 clamp，|x/s|>448 溢出 NaN 实证 21%，与原生 clamp 语义不符即裁）；整行 fused silu+quant 臂走 aiter fused（R-09②，7.5×）。异形臂（column_major/tma 对齐/ue8m0/masked_m/非 e4m3/3D 批）恒走上游；aiter 失败 log-once 回退上游。=stock 退回上游原生；production 晋级待 fp8 模型 serving A/B（门禁 tests/kernels/test_quant_aiter_gfx938.py 常驻 L1） |
| `HCU_AR_NORM_IMPL` | auto | 化 | 融合 AR+RMSNorm vendor 档（R-09①，ar_norm_aiter 桥 v2 双符号）：auto=hip+aiter GraphSafe 类可导入且 HCU_AITER_AR auto 规则未拒（旧布局图重放损坏史连带拒）则 aiter——**上游候选门 `apply_aiter(SGLANG_USE_AITER 全局门)/apply_flashinfer(sm90/100)` 在 HIP 恒关**，桥 arm `apply_flashinfer_allreduce_fusion` 三副本（add_norm 定义点+exit.py+fusions/allreduce.py from-import）重绑以武装候选与生产端推迟，dispatch=flashinfer 入口→`GroupCoordinator.fused_allreduce_rmsnorm`→aiter `custom_fused_ar_rms`；8rank 探针（nmz26 BW1102，hidden=4096）：**fp64 真值 norm 1ulp/residual 0ulp（优于 rccl 3-8ulp）**、vs customAR+norm **1.38-2.11×**、图回放精确（R-06 顾虑清除）；**tp8 serving 实证介入**（GLM-5.3 纯 TP 无 dp：attn+FFN 双边界、图捕获期起服、216 次融合 0 失败、输出正常）；dp-attention 在 arm 层即拒（dp==tp 时 attn 边界本无 AR 可融）；仅全 TP 组目标融合否则逐调用退 stock；=stock 全退上游 |
| `HCU_STATE_SNAPSHOT` | 空 | 1 | R-06 取证仪器（11c7f99）：挂 Glm5NextModel.forward（importlib 惰性包装），fresh prefill 入口 JSONL 记录 kpool 尾缓冲/mamba temporal+conv 的 abs 和与 nz 层数——状态载体排除法取证 |
| `HCU_STATE_ZERO` | 空 | 1 | 上者清零臂：=tail\|mamba\|all\|pages（pages 按 out_cache_loc 推页号，k/v buffer list-aware）——A/B 验证"清掉候选载体后还翻不翻" |
| `HCU_SIDE_PROBE` | 空 | 1 | 图内侧缓冲探针（dd0ec61）：=replay\|eager——捕获期把 fa3-emu 每调用张量 copy_ 进预分配侧 slab（拷贝入图，replay 重执行），读图 replay 真值（forward hook 不可 hook replay） |
| `HCU_KDA_PREFILL_IMPL` | **fork（hip 默认，2026-10-18 转正）** | 化 | GLM-5.x KDA prefill（chunk_kda，42 层）内核选择：**fork**=vendored sglang-das `fla/hcu` triton（对拍 emu cos 0.99999/终态 0.999995，快 473-1009×；beta 约定=v0.5.21 后端已激活直通，GLM 4 维 gate 即已激活；10-16 serving 翻转已归因 **custom AR** store 序 bug（160 臂 A/B：custom 60/100 vs 非 custom 1/60），10-18 宿主复位后干净复验转正：双 fork 锚点 4/4+马拉松 0/20+长文 4972 tok 对）；emu=顺序 torch 步进（=emu 退路，T=2048 要 414ms）；stock=官方 v0.5.21 内核（gfx938 Aqk 全错，红门 tests/kernels/test_kda_chunk_gfx938.py 常驻）。旧键 `HCU_KDA_PREFILL_EMU=1` ≡ 本键 =emu |
| `HCU_CONV1D_IMPL` | **fork（hip 默认，2026-10-18 转正）** | 化 | KDA causal_conv1d_fn（prefill varlen）+ causal_conv1d_update（decode 每步）内核选择：**fork**=vendored sglang-das hcu triton（对拍 emu cos 0.999996+/conv 状态位精确相等；fn 16.4×、update 4.3×，spec-decode 旗标臂原生支持；污染态 9-10/20 翻转观测作废（KFD 坏态数据），10-18 干净复验 conv-only 臂 0/20+Tokyo+长文对）；emu=torch 步进（=emu 退路）；stock=官方内核（update cos 0.26 / fn 初态臂死，红门常驻）。旧键 `HCU_CONV1D_{UPDATE,FN}_EMU=1` ≡ 本键 =emu |
| `HCU_KPOOL_WRITE_EMU` | 0 | — | DSA kpool 指数 K 缓存三写内核（softmax_rotate/decode_update/scatter）的 torch emu。内核复核在 (pages,8448) 2D buf 上健康（此前 ÷4 误判是 probe 侧自造 (pages,64,1,384) buf 使 shape[1]=64 被当页字节数的陷阱），默认直通 triton；留作回归 watch |
| `HCU_KPOOL_READ_EMU` | **1（GLM-5.x DSA 默认）** | 化 | index-K 缓存 gather 读（GetKAndS.aiter）走 torch emu：aiter `cp_gather_indexer_k_quant_cache`（preshuffle 臂）在 gfx938 非法地址访问/输出零——第 5 坏内核，不修则 prefill topk 的 K 读全零。emu 与写侧同 16-tile 映射（roundtrip 全等，tests/kernels/test_kpool_write_gfx938.py）。=0 退回 aiter 内核 |
| `HCU_MHC_UNFUSE` | **1（GLM-5.x mHC 默认）** | 化 | GLM-5.x mHC 层边界融合（attn→FFN 段 fused `mhc_post_pre`）禁用：`is_cross_layer_mhc_fusion_enabled()` 只查 `SGLANG_OPT_FUSE_MHC_POST_PRE` / USE_TILELANG_MHC_{PRE,POST}（默认全 True）**不查 is_hip** → gfx938 上每层跑 aiter triton fused mhc_post_pre，其 FFN 段 h_post 与 `_mhc_post_torch` 参考同输入对拍 z 偏移 +0.5~+5（hc_pre/hc_post runtime dump 实锤）→ 逐层污染 mHC 残差流 = GLM-5.3-Flash 词汤主因（第 6 坏内核）。hip 下默认强制 `SGLANG_OPT_FUSE_MHC_POST_PRE`=false 走 hc_post+hc_ffn_pre 纯 torch 链；=0 恢复融合 |
| `HCU_MHC_EMU` | 0 | — | **无效操作（保留防呆）**：原想 emu hyperconnection.py 的 fused_hc_mix/hc_combine，但 glm5_next 实际消费 `kernels/ops/layernorm/mhc.py`，这两个包夹无调用方；mHC 真正的坏点是融合门控不查 is_hip（见 `HCU_MHC_UNFUSE`）。设 1 无任何效果 |
| `SGLANG_DISABLE_DSA_INDEXER_FUSION` | 0 | **GLM-5.x DSA 启动配方显式 =1**（见下） | 上游原生 env。GLM-5.x indexer Q 走 V3.2 `[rope|nope]` 布局融合核（fused_q_indexer_rope_first_quant），上游仅 CUDA 实现（docstring 写明 CUDA only，HIP 无臂，`_is_hip=True` 救不了）→ graph 捕获期 JIT 炸。=1 走 unfused（分离 rope+act_quant+分存，HIP 专家原路径） |
| `SGLANG_MOE_CONFIG_DIR` | 空（走包内 configs/） | — | fused-MoE triton 调优文件目录（上游 env 正门）。无 HCU 命名 tune 文件时 adapter 的 moe_config_guard 把超 64KB LDS 的 tile 就地 clamp（BN=BK=64/stages≤2；gfx93x 硬限 64KB，NV 兄弟默认 tile 72KB 直接 OutOfResources）；此目录供后续正式 tuning 产物落位 |

## 平台 runtime env（非 adapter 键，不在 hcu_env 注册表）

cookbook（`inference-cookbook-das`，团队同级仓库）glm-5.3 配方的 sglang
export 块里有一组 DTK/HIP runtime env。graph 发射相关项与我们的 decode
瓶颈直接对口，2026-10-16 在 nmz26 TP4 完成三轮 A/B，**结论：机制坐实、
官方文档背书的 env 两个变量在本栈（dtk26041）全部失败，不要启用**——

| env | 文档默认 | A/B 结果（2026-10-16，nmz26 TP4） |
|---|---|---|
| `ROC_AQL_QUEUE_SIZE` | 16384 | =131072：吞吐零变化（2.30 vs 2.34 tok/s）+ **贪心非确定**（Berlin 两轮不一致；对照组逐字复现）——两次独立重启复现 |
| `HIP_USE_GRAPH_QUEUE_POOL` | false | =1：吞吐零变化（430ms/步）+ **贪心非确定**（Tokyo 三轮两绿一红数字垃圾；Berlin 恰好基线签名） |
| `HIP_GRAPH_ACCUMULATE_DISPATCH` | **true** | cookbook 设 =1 实为**无操作**（默认已开）——此前"三件套"A/B 的唯一有效变量就是队列扩容 |
| `HIP_GRAPH_USE_CMD_CACHE` | **1** | 同上，默认已开，无操作 |

**关键修正**（据 DTK 官方文档
`dtk.sourcefind.cn/docs/runtime/zh/latest/runtime-api/env_variables.html`）：
三件套里两个 graph 开关默认即开，故三轮 A/B 实际只动过队列容量与队列池
两个变量——各自独立引入非确定性，判定 dtk26041 的队列扩容/graph 队列池
路径数值均不安全。

**机制证据链（用户 trace 分析 + 4×chrome trace 复核）**：
① `hipGraphLaunch` API 单次 p50 ≈ 540ms、11 次占满 5.3s 采样窗（每图
≈24k kernel；注意 graph 并行默认 4 流 × ~6k 包 < 默认队列 16384，"队列
撑爆"的朴素解释不成立，API 自旋的真正瓶颈在 runtime 提交路径）；
② rank0 短板：TP1-3 nccl 各烧 2.5s/窗（p95 17.8ms）纯等 TP0（TP0 nccl
仅 0.17s）。候选账（2026-10-21/22 nmz26 V4.1-Flash dp8ep8 实测推进）：
`HIP_GRAPH_LARGE_LAUNCH=1` **已测=无操作**（bs16 dspark 1090ms/步 vs 基线
1085ms，monster 拍逐拍同形、宿主 hipGraphLaunch 36.7ms 不变——疑该 DTK
build 未读此键；recipe 已摘除）；`DEBUG_HIP_FORCE_GRAPH_QUEUES`（graph
并行流数，文档注明与队列池配合）、`GPU_MAX_HW_QUEUES`（默认 27/进程/设备，
非我们瓶颈）未测；`HSA_KERNARG_POOL_SIZE=8388608` **排队中**（nmz26 实验
两遭 co-tenant 抢卡，脚本已备）。
trace 存证：容器 `/root/logs/prof_torch/1792113313*.trace.json.gz`
（Perfetto 可开）。后续走 docs/plugins.md P9 的 DTK 提报路径。cookbook
同块其余（`HIP_KERNEL_BATCH_CEILING=100`
等）未测；块里的 `SGLANG_*` 旗标属 sglang-das 0.5.18 fork 专属功能
（P1/P2 范畴），我们官方 v0.5.21 栈不设。

**拍（beat）定性推进（2026-10-21/22 夜，V4.1-Flash dp8ep8+dspark bs16 20+10 步
8-rank trace + 消融链）**：步内瓶颈不是提交路径（宿主 36.7ms/图、设备空转仅
15%），而是**图内大同步的确定性等待拍**——每步 5-6 拍、每拍 ~116ms、拍距
均匀 ~182ms、逐拍落点跨步稳定（±3ms）；拍内"被拉伸核"名字随机（markov/
copy/quant/gemm/nccl 都中过）= 谁先消费未到的跨 rank 缓冲谁吃等待；
**跨 rank 对齐（TP0/1/4 绝对时间）拍窗一致到 0.1ms=全 rank 同时停**，非某
rank 慢（skew 会抖、此拍不抖）；dspark 拍数 6≈块大小 5+验证 1（每阶段一次
大同步），无 dspark 2 拍（81+55ms）同机制。markov 采样核本体 µs 级小 grid
（markov_sample.py triton，rows×~16 parts），118.1ms 全是等跨 rank partial
gather。旁证：图内 nccl 8KB allreduce 微基准 33µs（快）、逐层 allgather_vec
均值 ~550µs（快）——拍是**每步少数几次 logits/lm-head 级大同步**固有的。
吞吐账（未 profiled 日志）：裸 decode 7.33 tok/s/rank（273ms/步）与 profiled
268ms 一致=拍非 profiler 伪影；dspark ~7.3 tok/s/rank 与裸 decode 每 token
持平（步 3×慢但每步 ~3 token）——**拍杀死前 dspark 无净收益**。trace 本地
`prof_dsv41f_dec20/*.trace.json.gz`（ll_*=LARGE_LAUNCH 轮）。附：profiler
`num_steps` 自动停 + trace 导出后服务器必崩（kill_process_tree，两次同签名，
22:21/23:49）——迭代判决改用免 profiler 日志吞吐。

## R-08 回仓盘点表（2026-10-18 落地）

PM 预分类（2026-10-17 扫 cookbook + `_launch_`/`_test_` 脚本，18 变量）→ 本表为逐行去向账本。
出现次数为盘点时点值。

### A 类：platform seam setdefault（已落地）

| 变量 | 次数 | 去向 | 理由 |
|---|---|---|---|
| SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python | 31 | seam（`platform_seam.platform_env_defaults`） | rust TreeCore 无 HCU 预编译扩展，python core 是平台事实 |
| HCU_DEEPGEMM_BRIDGE=1 | 13 | seam | DTK deepgemm 桥，gfx936 EP-LL + gfx938 GLM 线双验 |
| SGLANG_RUST_BUILD_MODE=never | 4 | seam | 与上条同源（构建机制同一开关族） |
| SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false | 4 | seam | 本栈非对称 TP 布局必误报 |
| SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 | 2 | seam（R-04 阶段一，35037be） | AOT 轮子纯启动税；boot 6:02→3:57 |
| **HCU_AITER_AR=0** | 12 | **改判不落** | hcu_env auto 规则已是正确默认（旧布局拒/新 GraphSafe 放行）；手敲多系 auto 规则就绪前的历史产物 |

语义钉：`tests/adapter/test_platform_env.py`（填充/显式覆盖赢/幂等三断言）。

### B 类：模型感知 auto（已落地 2026-10-18，`hcu_adapter/model_env.py`）

| 变量 | 去向 | 理由 |
|---|---|---|
| SGLANG_DSA_FUSE_TOPK=0 | model_env（hf config 含 index_kpool 等 → setdefault） | fused topk 无 HIP 臂 |
| SGLANG_DISABLE_DSA_INDEXER_FUSION=1 | model_env（同上） | rope_first/indexer 融合无 HIP 臂（上游逃生门） |
| SGLANG_MAMBA_CONV_DTYPE=float16 | model_env（fp16 + GDN 键 → setdefault；bf16 显式排除） | 上游 conv 池硬编码 bf16，fp16 首 forward 必崩 |
| HCU_TRITON_CONTIG | **改判不落**（默认已 =1） | hcu_env 默认即目标值 |
| HCU_FA_IMPL / HCU_LINEAR_ATTN_GDN | **改判不落**（默认 auto 已自适应） | 手敲多系 BW200 早期配方 |
| SGLANG_DEEPEP_BF16_DISPATCH | **改判留配方**（工作流层参数） | CLI 等价 --deepep-dispatcher-output-dtype bf16；R-04 fp8 解锁后退役 |

开关 `HCU_MODEL_ENV`（hcu_env 注册，default auto；off 退回手敲）；registry 登记
op=model_env；BACKEND_CATALOG/UI 快照/yaml 三处同 commit。**生效机制（53ebad0
终态）**：sitecustomize 期对 `--model-path`→config.json 递归段扫描做 **argv
嗅探**（首版 importlib 包装 ModelRunner 方案拦不到 launch 路径静态 import，
真实 serving 从不触发——零 env 四门首跑即抓出）；boot 日志实证行
`model-env default ... (model-derived, argv sniff <model-path>)`。规则钉：
`tests/adapter/test_model_env.py` test_apply_model_env（DSA 双设/无关不动/
fp16 GDN 设/bf16 GDN 不设/显式赢）+ test_apply_from_argv（嗅探臂）。

### C 类：留调试（手册在册即可）

`HCU_PREFILL_CP_DEBUG`、`HCU_FA_EMU_DEBUG`、`HCU_FP8_TRACE`；
`HCU_PREFILL_CP=1` 特性 opt-in 保留为配置项。

### 防复发

cookbook §0 已注明 seam 默认清单 + "新配方行内每个 env 必须标注去向"；PM 验收检查单
新增"新增 env 是否回仓"一项。

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

**拍围猎推进（2026-10-22 晨，续 R-08/拍定性）**：
① `HSA_KERNARG_POOL_SIZE=8388608` 实测=**无操作**（nodspark 免 profiler 日志
吞吐 7.33 tok/s/rank 与基线逐位一致）——env 梯子两连败（LARGE_LAUNCH 同），
与"拍=大同步固定成本"定性互证；
② **拍非 dp-attention 专属**：纯 TP8EP8（无 dp 汇聚）同负载 301ms/步
（53.1 tok/s 总）vs dp8 273ms——拍在两拓扑都在，dp8 反而更快（61 次/层 TP
AR 便宜得很）；共享嫌疑面收窄到每步骨架的 lm-head/logits/采样相位
（nods 拍核=Cijk_S_PostGSU lm-head GEMM 55.4ms + 中步 copy 81ms）；
③ **aiter AG 无罪**：nods trace `aiter::allgather_vec` 410 次均值 24µs——
拍不是汇聚核慢，是下游消费核等的数据迟到（拍核=先读汇聚缓冲者）；
④ **nods 流 97.5% busy**（idle 仅 68ms/2738ms）：拍不是空闲等待，是核内
拉伸（真跑但慢 50×），时长逐字复现（80.9/55.4ms）= 确定性机制非热/频抖动；
⑤ **profiler 毒化 TP8**：TP8 拓扑下 profiled 8.3s/步 vs 未 profiled 301ms
（28×，逐核 ~1.5ms 额外失速、507 核 >100ms、含 nccl AR 88ms）——R-08 提交
路径病在 TP8+profiler 组合下全面爆发；dp8 下 profiler 却近似无感（268≈273）。
**TP8 拓扑取证一律免 profiler（日志吞吐法），trace 仅 dp8 可信**；
⑥ profiler teardown 杀服务器第 3 次（TP8，05:52）——跨拓扑 100% 复现，
num_steps 自动停+导出后必崩，系统性问题确认；
⑦ MHC A1 双针（use_fused_stats 放行 HIP + mhc.py _is_hip 硬判）四门全绿
图捕获零错但吞吐中性（7.28 vs 7.33）——**trace 家族总额是拍拉伸转嫁**的
教训入账（真实功量 µs 级）；补丁保留（每层 ~10 核→2 核，图规模 -10%），
diff 存 tmp_fetch/MHC_PATCHES.md 待 needle 化。
下一步：拍机制取证需换工具（DTK rocprof/带内计时器/图内 hipEvent），
trace 族在拍存在时不可用于功量归因。
