# HCU sglang 启动 Cookbook（已验证配方）

所有配方均在 **nmz26 验证容器（容器名仅作环境记录）+ 官方 sglang v0.5.21 源码（/root/sglang-v0521-src）+ 本 adapter（主线）** 实测通过（GLM-5-W4A8 在 nmz18）。四判定详情（启动/生成/语义/长文）见 [models.md](models.md)；环境变量语义见 [env-vars.md](env-vars.md)。物料脚本在 `tests/tools/_launch_*.sh`。

## 0. 公共环境（每个配方都带）

```bash
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python   # adapter 优先 + 官方源码
export CUDA_VISIBLE_DEVICES=6                                   # 按卡改；多卡写 0,1,2,3
# 以下已由 adapter platform seam 默认化（R-08 回仓，显式 export 永远可覆盖）：
#   SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python（rust TreeCore 无预编译扩展必炸）
#   SGLANG_RUST_BUILD_MODE=never / SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false
#   HCU_DEEPGEMM_BRIDGE=1 / SGLANG_JIT_DEEPGEMM_PRECOMPILE=0（R-04 启动提速）
#   HCU_AITER_AR 走 hcu_env auto 规则（旧布局拒用、新 GraphSafe 放行），无需手敲=0
```

注意：fp16 底模的 GDN 模型（如 Qwen3.5-122B-AWQ）必须加 `export SGLANG_MAMBA_CONV_DTYPE=float16`（上游 conv 池硬编码 bf16 不看模型 dtype，首 forward 必崩）。

代码同步按 [bootstrap.md](bootstrap.md) §1 执行：**git 为主**——`git pull` 后 `git status --short` 必须为空、`git log -1` 与本地 HEAD 一致（同步好没好，diff 一眼可判）；tar 仅隧道断连应急，且必须落进 git 树解包以便 diff 对账（不可对账的同步不算完成）。

## 1. 基础模型（bf16 / FP8 / INT8，TP1）

```bash
# Qwen3-0.6B / Qwen3-4B (bf16)——TP1 基线；0.6B 可省 mem-fraction
python -m sglang.launch_server --model-path <模型路径> \
  --attention-backend triton --host 0.0.0.0 --port 30000
# Qwen3-4B 曾加：--mem-fraction-static 0.85 --page-size 64（GSM8K-100 68%）

# Qwen3-30B-A3B (MoE 57G) —— 64G 单卡
python -m sglang.launch_server --model-path <Qwen3-30B-A3B> \
  --attention-backend triton --mem-fraction-static 0.85 --host 0.0.0.0 --port 30000

# Qwen3.8-27B-FP8（fp8 全链：fp8_scaled_mm triton 后端，量化自动识别，无需旗标）
python -m sglang.launch_server --model-path <Qwen3.8-27B-FP8> \
  --attention-backend triton --mem-fraction-static 0.85 --host 0.0.0.0 --port 30000

# Qwen3.6-27B-Channel-INT8-w8a8（compressed-tensors int8，自动识别）
python -m sglang.launch_server --model-path <Qwen3.6-27B-Channel-INT8> \
  --attention-backend triton --mem-fraction-static 0.85 --host 0.0.0.0 --port 30000
# ⚠ 仅限 dense 线性层路径：MoE 版 CT-INT8（如 Qwen3-30B-A3B-Channel-INT8、
#   GLM-5.3-Channel-INT8）在 v0.5.21 上游 get_moe_scheme 对动态 int8 MoE 直接
#   NotImplementedError（仅 NPU，compressed_tensors.py:896，CUDA 同拒）——
#   归因见 models.md 2026-10-18 R-10 行，勿排期实跑。
```

## 1.5 nmz26 现役栈速查（2026-10-18 起：镜像 sf_b075-20261018-0048）

```bash
# 重建后必做两步（详见 bootstrap.md §0/§3/§5）：
#   ① tvm-ffi 预热（或从健康容器直拷 .so）
#   ② tilelang 钉 0.1.12（0.1.15 编译回归，保 mHC 融合路径）：
tar xzf /root/g53/vendor_tilelang_0112.tgz -C /usr/local/lib/python3.10/dist-packages/
# （先删 0.1.15 的 tilelang/ 目录与 dist-info；验证 import tilelang == 0.1.12+dtk2604.torch2110.2610100923）

# glm53f TP4 零 env 生产配方（四门绿+单流 7 tok/s，2026-10-18 新镜像复验）：
export PYTHONPATH=/root/hcu_t2:/sglang-v0521/python
export LD_LIBRARY_PATH=/opt/dtk/dcc/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/hip/lib:/opt/dtk/lib
export HIP_VISIBLE_DEVICES=4,5,6,7   # 平台卫生 env 见 tests/tools/_launch_g53f_tp4.sh
python3 -m sglang.launch_server --model-path /models/glm53f --served-model-name glm53-flash \
  --trust-remote-code --host 0.0.0.0 --port 30197 --tp-size 4 \
  --page-size 64 --mem-fraction-static 0.62 --chunked-prefill-size 8192 \
  --max-running-requests 16 --cuda-graph-max-bs-decode 16 \
  --disable-radix-cache --disable-prefill-cuda-graph \
  --dsa-topk-backend torch --dsa-prefill-backend fa3 --dsa-decode-backend flashmla_sparse \
  --nccl-port 30397
```

## 1.6 nmz18 × v0.5.22 × glm53f 配方（2026-10-20 四门全绿，R-17 三绑定首演）

容器 `g53f-v0522`（sf_b075-20261018-0048 + privileged + hyhal + 模型只读 +
`$HOME/sglang-v0522` 公共盘树 + g53_tp4 工具盘）；adapter 在容器内
`/root/hcu_v0522`（主仓 rel/v0.5.22 git 克隆，起服前 `git pull` + `git log -1`
对账）。冷容器必做：三板斧种子导入 + tilelang 0.1.12 钉版（§1.5 同款）。

```bash
export PYTHONPATH=/root/hcu_v0522:/sglang-v0522/python
export LD_LIBRARY_PATH=/opt/dtk/dcc/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/hip/lib:/opt/dtk/lib
export HIP_VISIBLE_DEVICES=0,1,2,3
# 卫生 env 两行必带：镜像烤了 508 机器的 NCCL_TOPO_FILE，BW1101 上
# ncclCommInitRank 直接 internal error（box16 g53f 脚本同款）
unset NCCL_TOPO_FILE
unset ROCR_VISIBLE_DEVICES
python3 -m sglang.launch_server --model-path /models/glm53f --served-model-name glm53-flash \
  --trust-remote-code --host 0.0.0.0 --port 30197 --tp-size 4 \
  --page-size 64 --mem-fraction-static 0.62 --chunked-prefill-size 8192 \
  --max-running-requests 16 --cuda-graph-max-bs-decode 16 \
  --disable-radix-cache --disable-prefill-cuda-graph \
  --dsa-topk-backend torch --dsa-prefill-backend fa3 --dsa-decode-backend flashmla_sparse \
  --nccl-port 30397
```

注意：v0.5.22 model_hook 会把 page-size 自动改到 256（resolve 后 server_args
实际 page_size=256），非 bug；权重 NFS 首载 ~49min（页缓存热后 ~13min）。
锚定：Berlin/Tokyo 双绿、单流 10.0 tok/s（128tok×3 确定性）、c8 ~40 tok/s、
~5k tok 双事实检索全对。

## 2. Qwen3.5 家族（VLM + GDN 线性注意力 + MoE）

### 2.1 Qwen3.5-122B-A10B-AWQ TP4（用户点名验证，四判定全过）

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3
export SGLANG_MAMBA_CONV_DTYPE=float16        # fp16 底模必须
python -m sglang.launch_server \
  --model-path /public/opendas/DL_DATA/llm-models/qwen3.5/Qwen3.5-122B-A10B-AWQ \
  --tp-size 4 --quantization moe_wna16 --attention-backend triton \
  --trust-remote-code --disable-cuda-graph --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 30078
# ⚠ --quantization moe_wna16 是唯一正确入口：默认无旗标会被 capability=(9,3)≥80
#   骗去自动转 awq_marlin，而 HIP 分支 awq_marlin_repack=None 必炸。
#   多卡 graph 已解禁（911eeeb+ adapter custom AR，见 §4 平台约束）——本例
#   AWQ 验证时为 eager，graph 版可去掉 --disable-cuda-graph 重验。
```

### 2.2 Qwen3.5-4B + MTP 投机解码（EAGLE，8/8 锚点，accept ~3.0/4）

```bash
M=/public/opendas/DL_DATA/llm-models/qwen3.5/Qwen3.5-4B
# 推荐：cuda graph ON（吞吐 4.6–7.6 tok/s，为 eager 版 2.6×；capture target+draft 各 23 档）
python -m sglang.launch_server \
  --model-path $M --speculative-draft-model-path $M \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --attention-backend fa3 --mem-fraction-static 0.80 \
  --host 0.0.0.0 --port 30084
# eager 对照版：加 --disable-cuda-graph（1.8 tok/s，仅作正确性对照）
# 要点：draft 与主模型同路径零重打包（自动改写 Qwen3_5ForCausalLMMTP）；
#      NEXTN 旗标也可用（v0.5.21 里它是 EAGLE 的保留别名）；
#      GDN 线性注意力层走 linear_attn_backend=triton（上游唯一实现，与全注意力 backend 无关）。
```

## 3. 功能开关速查（均在 Qwen3-0.6B/4B 上四判定通过）

| 功能 | 参数 | 状态 |
|---|---|---|
| fa3 attention backend | `--attention-backend fa3`（adapter emu，graph 安全） | ✅ |
| fp8 KV cache | `--kv-cache-dtype fp8_e4m3`（triton 后端零 adapter 改动） | ✅ |
| fa3 + fp8 KV + TP2 | 两者组合 + `--tp-size 2 --disable-cuda-graph` | ✅ |
| dp-attention | `--tp-size 2 --dp-size 2 --enable-dp-attention`（graph ON，零 adapter 改动） | ✅ |
| radix cache | 默认开（恒带 `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`） | ✅ 命中门禁过：重发 cached 256 输出逐字一致、部分前缀命中 |
| prefill CUDA graph | 默认 58 桶（weak_ref vendor 直通修复后**可全捕**，27B 捕完仍余 16G）；**GDN 混合模型建议 `--disable-prefill-cuda-graph`**：判决矩阵三场景全负（命中增量 1.5-1.8 vs eager 0.77s、COLD 11.6 vs 6.1s、16 并发 13.6 vs 4.2s——breakable 每层 eager break 桥接+桶 pad 的 host 面在 GDN/Mamba 混合上比纯 eager 更贵，2026-10-14 双 rig 判决） | ⚠ graph ON 不再 OOM，但 GDN 混合当前仅"能开"非"该开"；dense 模型的高频小桶场景待复测 |
| MTP 投机解码 | 见 §2.2 | ✅ 精度过；吞吐见 §2.2 注（emu 栈暂负收益，真 FA3 内核预期翻正） |
| hicache 分层 KV | `--enable-hierarchical-cache`（需 adapter 的 get_device_accessible_ptr 桥，已内置） | ⚠ dense 模型全通（逐出-回迁保真）；**GDN 混合模型 MAMBA 状态宿主迁移输出漂移待修，生产 GDN 模型暂不开** |
| AWQ 量化 | `--quantization moe_wna16`（见 §2.1） | ✅ |
| slimquant W4A8 | 零显式旗标（checkpoint 自动识别，nmz18 TP8 graph ON 验证） | ✅ GLM-5-W4A8_V2_6 |
| PD 分离（节点内） | TCP 版命令块见下方（mooncake_tcp，无需 IB）；RDMA 版见同节（mooncake RoCE，需容器注入三步） | ✅ 双传输面均验：TCP 8 路并发零串扰；RDMA 同配方 5/6+确定性+8 路全过；⚠ `--disaggregation-enable-kv-checksum` 并发误报勿开 |
| EP 低延迟（DeepEP） | 命令块见下方（LL 是 bf16 MoE 唯一可用 deepep 模式） | ✅ Qwen3-30B-A3B TP2+EP2（锚点+并发全对） |
| prefill CP（上下文并行） | `HCU_PREFILL_CP=1` + `--enable-prefill-cp --cp-strategy zigzag --attn-cp-size 2`（与 `--tp-size 2` 成对，fa3） | ✅ Qwen3-30B-A3B nmz26 双卡：输出与 TP2 基线对齐（capital/long OK、arith 同因思考超长 MISS、determinism 基线亦 DIFF 非 CP 回归）。上游 HIP 门+两处语义补丁见 [env-vars.md](env-vars.md) `HCU_PREFILL_CP`；排查插桩 `HCU_PREFILL_CP_DEBUG=1`；物料 `tests/tools/_launch_cp2.sh` |
| EP 高吞吐 / auto / normal | `--deepep-mode auto/normal`（bf16，同 LL 配方改 mode 即可） | ✅ contiguous grouped GEMM 已用 triton 重写替代坏掉的 DTK w16a16 内核（`HCU_TRITON_CONTIG=1` 默认开）：auto/纯 normal 均锚点+并发+确定性全过；**FP8 线仍锁**（a_scale row/tensor-wise ≠ sglang 128-block，非桥可解） |

```bash
# hicache 示例（dense 模型；小池可强制逐出验证回迁）
python -m sglang.launch_server --model-path /root/models/Qwen3-0.6B \
  --attention-backend fa3 --enable-hierarchical-cache \
  --host 0.0.0.0 --port 30087

# PD 分离示例（节点内两卡 + CPU mini-LB；Qwen3-0.6B 实测）
# 1) prefill（GPU6）——bootstrap-port 两端一致；nccl-port 各自独立防撞
CUDA_VISIBLE_DEVICES=6 MC_TCP_ENABLE_CONNECTION_POOL=true python -m sglang.launch_server \
  --model-path /root/models/Qwen3-0.6B --trust-remote-code --attention-backend triton \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-bootstrap-port 30590 --nccl-port 30390 --tp 1 --page-size 64 \
  --host 0.0.0.0 --port 30190
# 2) decode（GPU7）
CUDA_VISIBLE_DEVICES=7 MC_TCP_ENABLE_CONNECTION_POOL=true python -m sglang.launch_server \
  --model-path /root/models/Qwen3-0.6B --trust-remote-code --attention-backend triton \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-bootstrap-port 30590 --nccl-port 30391 --tp 1 --page-size 64 \
  --host 0.0.0.0 --port 30191
# 3) mini-LB（CPU，客户端打 30090）
python -m sglang_router.launch_router --pd-disaggregation --mini-lb \
  --prefill http://127.0.0.1:30190 --decode http://127.0.0.1:30191 \
  --host 0.0.0.0 --port 30090
# ⚠ 勿加 --disaggregation-enable-kv-checksum：并发≥2 必误报（数据面本身正确）；
#   mooncake_tcp = mooncake 强制 TCP 传输（MC_FORCE_TCP），节点内无需 IB/RDMA。

# EP 低延迟示例（DeepEP，Qwen3-30B-A3B TP2+EP2 于 GPU6/7 实测）
#   关键旗标一个不能少：moe-runner-backend deep_gemm 是 deepep_ll 格式唯一注册的 runner；
#   SGLANG_DEEPEP_BF16_DISPATCH=1 让 dispatch 输出 bf16（工作流层参数，留配方；CLI 等价 --deepep-dispatcher-output-dtype bf16；R-04 fp8 contiguous 解锁后退役）；
#   mem 0.60：HIPC 权重打包会在运行时双份缓存专家权重，留出余量；
#   --disable-radix-cache：LL buffer 固定槽位与 radix cache 不兼容（上游要求）。
CUDA_VISIBLE_DEVICES=6,7 SGLANG_DEEPEP_BF16_DISPATCH=1 \
python -m sglang.launch_server \
  --model-path /public/opdas/DL_DATA/llm-models/qwen3/Qwen3-30B-A3B \
  --trust-remote-code --attention-backend triton \
  --tp-size 2 --ep-size 2 --moe-a2a-backend deepep --deepep-mode low_latency \
  --moe-runner-backend deep_gemm --disable-radix-cache --disable-cuda-graph \
  --mem-fraction-static 0.60 --nccl-port 30392 \
  --host 0.0.0.0 --port 30193
# ✅ EP 高吞吐同一配方改 mode 即可（2026-10-11 双验）：--deepep-mode auto
#   （prefill normal + decode LL）或 normal（decode 也 contiguous）。
#   contiguous grouped GEMM 已由 adapter triton 内核接管（HCU_TRITON_CONTIG=1
#   默认开；DTK w16a16 contiguous 原核 2D VMFault/3D 数字坏，勿关开关做生产）。
#   单测门槛 tests/tools/_test_contig_triton.py（rel≈3e-3，62.6 TFLOP/s bf16）。
#   仍锁：FP8 DeepEP normal（DTK a_scale row/tensor-wise ≠ sglang 128-block）。

# PD 分离 RDMA 版（mooncake RoCE 直传；Qwen3-0.6B 实测 30195/30196/30091）
#   与 TCP 版唯一差异：--disaggregation-transfer-backend mooncake（默认即 RDMA）。
#   ⚠ 必须用 `docker exec --privileged` 启动两端服务（memlock 见下方注入配方）。
#
# —— 非特权容器接入 RDMA 三步（一次性，容器重启需重做第 1/2 步）——
# 1) 设备节点：容器默认无 /dev/infiniband，CAP_MKNOD 在，直接建：
#    for i in $(seq 0 9); do mknod /dev/infiniband/uverbs$i c 231 $((192+i));
#      mknod /dev/infiniband/issm$i c 231 $((64+i)); done; chmod 666 /dev/infiniband/*
# 2) cgroup devices 白名单（否则 open 报 EPERM；需 docker 组权限，只动自己的容器）：
#    docker run --rm --privileged mtv-img:latest sh -c \
#      'mkdir -p /tmp/cg && mount -t cgroup -o devices none /tmp/cg && \
#       echo "c 231:* rwm" > /tmp/cg/docker/<容器长ID>/devices.allow'
#    （daemon 禁挂 cgroup 路径时改用 --pid host + nsenter -t 1 -m 进宿主 mount ns 执行）
# 3) memlock：容器默认 64MB，RDMA CQ 创建报 ENOMEM [12]——
#    `docker exec --privileged` 启动服务即解（CAP_IPC_LOCK 豁免 pinned 记账；
#    runc exec 每次按容器 spec 重置 rlimits，运行时改 PID1 不传播，硬限 64 也提不动）。
#   传输面验证：两端日志 `Using RDMA transport (RoCE/iWARP)` + `Found N HCAs`
#   （TCP 版对应行是 TcpTransport listen）；decode 侧 #transfer-req 为在途传输。
# prefill 端（GPU6）：
CUDA_VISIBLE_DEVICES=6 python -m sglang.launch_server \
  --model-path /root/models/Qwen3-0.6B --trust-remote-code --attention-backend triton \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 30591 --nccl-port 30395 --tp 1 --page-size 64 \
  --host 0.0.0.0 --port 30195
# decode 端同参数改 --disaggregation-mode decode、--port 30196、--nccl-port 30396；
# mooncake DTK 0.3.13 的 initialize 恰为 v0.5.21 的 4 参形态，无需 adapter 桥。
```

## 3.5 CI（MR 自动测试）

L0 结构门禁（零依赖秒级，托管平台 CI）+ L1 GPU 数值门禁（容器内自动挑空闲卡，结果回帖 MR）+ L2 e2e（夜间/发版）。启用与归层规则见 [ci.md](ci.md)；入口 `ci/l0_gate.py` / `ci/l1_gpu_gate.sh`。

## 4. 平台约束（必读）

1. **多卡 TP>1 + cuda graph 已解禁（911eeeb 起，默认生效）**：历史乱码根因不是 NCCL，是 aiter AR 图重放损坏 + `sgl_kernel.allreduce` 缺失（sglang 图内 AR 只能落 custom AR）双因叠加；adapter 自制 custom AR（信令+payload 走 host-shm，graph 安全，小消息快 NCCL ~2×，>512KB 自动回 NCCL）修复，TP2 graph e2e 已过（见 models.md 2026-10-11 行）。旧栈或显式 `HCU_TRITON_AR=0` 时仍需 `--disable-cuda-graph`。
2. **AWQ 永远显式 `--quantization moe_wna16`**（理由见 §2.1）。
3. **fp16 底模 GDN 模型加 `SGLANG_MAMBA_CONV_DTYPE=float16`**。
4. thinking 模型（Qwen3-30B-A3B 等）：锚点判定给足 `max_new_tokens`，长文看 `</think>` 后文本。
5. Qwen3.5 全系是 VLM（视觉塔自动加载，纯文本请求正常工作）。
6. `--page-size` 默认 1 可用；批次 suite 用过 64（R1/flashmla 等场景需要）。
7. DeepSeek-R1-L4 + flashmla（`--decode-attention-backend flashmla --prefill-attention-backend triton --page-size 64`）为**验证 rig**（截层模型语义本坏，只测链路非精度），物料 `tests/tools/_launch_r1l4_*.sh`。
8. **任何启动恒显式 `--attention-backend triton/fa3`**：不指定时默认解析到 aiter，dense prefill 的 `mha_batch_prefill_func` 是模块不可调用，首 forward 必崩（PD prefill 首测实测）。
9. **DeepEP EP 并行三模式全开**（bf16 MoE）：`low_latency`/`auto`/`normal` 均已验证（normal/auto 的 contiguous grouped GEMM 走 adapter triton 内核，`HCU_TRITON_CONTIG=1` 默认开）；必须 `--moe-runner-backend deep_gemm` + `HCU_DEEPGEMM_BRIDGE=1` + bf16 dispatch env（见 §3 命令块）。FP8 DeepEP（normal 线）仍锁：DTK a_scale 语义与 sglang 128-block 不匹配。

## 5. trace 耗时异常排查配方（驱动空泡吸收家族，2026-10-26 box16 实战提炼）

**背景（历史口供，用户 2026-10-26）**：DTK 栈 prof 早年 trace 里 kernel 之间常见空泡（提前下发/graphlaunch 也在），runtime 一度归驱动、一度归编译器；后来驱动侧"解了"——**空泡从 trace 消失、kernel 耗时变长=空泡时间被折算进 kernel 窗**。nmz18/nmz26 的拍（GPU 不执行已入队活）与本问题同源。上层可绕：das 用整 MoE 单核 vendor kernel（aiter moe_c/lightop marlin）后很少复现——**核少而肥=泡触发面小**。

**症状指纹**（占其一即入本流程）：
1. 某核 trace 耗时远超其工作量应有的量级（如 elementwise 级工作记出 ms 级）；
2. 每颗 X 核的**后继核**耗时异常，而该后继核在其他位置正常（全局 p50 对比）；
3. 嫌疑核自身记录**低得离谱**（如 GEMM 记 7µs）+主流"覆盖"接近满载但墙钟慢；
4. 同核 standalone 快、in-server 慢，且步时**稳态慢**（不是间歇塌陷——稳慢=每步都在交泡税，box16 教训：稳慢比间歇更隐蔽）。

**排查主线**（顺序执行，全部 tracer 无关取证优先）：
1. **别信 trace 绝对值，先看结构**：correlation→谁发射（hipGraphLaunch=图内）、前后继关系、同流 gap；
2. **后继吸收检验**：对每颗嫌疑核取后继核耗时 vs 该后继核全局 p50——"后继异常"=前核真值被位移/吸收（实证：gptq 记 7µs，后继 act 记 5056µs≈UP 真值）；
3. **真值三板斧**（cuda events，不经 roctracer）：
   a. standalone 探针：同核同形状热编译后事件计时（注意**短探针才准**——长扫描撞降频，同配置可差 3x：1.2ms vs 3.9ms 实测）；
   b. 图内差分：capture 含/不含嫌疑核的 graph 重放墙钟差（图内外一致=真慢；probe_graph_pair.py 模板）；
   c. 工作量缩放：专家数/M 扫描——**耗时平坦=固定税（泡），按比例=真算力病**（int4 GEMM 实证：M=1 与 M=6 同耗 1182/1186µs=token 无关，专家数 1→48 才 730→3785µs）；
4. **通信排除**：8 卡集合微基准（allreduce 81µs/a2a 112µs 级=健康，模板 bench16coll.py）；
5. 结论落地：内核级病→换 vendor 核（照 das 接法），配置级病→调优器**必须套 smem 守卫预算**否则运行期被钳（tuner 欠账教训）。

**采集协议坑**：/start_profile+stop 后 ~90s 调度器段错误（roctracer teardown）——**一次采集一重启**；trace 用 num_steps 自动停采集窗。

**战例**：REDUCTION_PATCHES W14/W14 补（box16 int4 GEMM 定罪全过程）；trace 存档 prof_dsv41f_box16_decode/。
