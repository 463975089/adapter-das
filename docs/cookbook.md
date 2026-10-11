# HCU sglang 启动 Cookbook（已验证配方）

所有配方均在 **nmz26 容器 lzg_pr418_test + 官方 sglang v0.5.21 源码（/root/sglang-v0521-src）+ 本 adapter（Gitee master）** 实测通过（GLM-5-W4A8 在 nmz18）。四判定详情（启动/生成/语义/长文）见 [models.md](models.md)；环境变量语义见 [env-vars.md](env-vars.md)。物料脚本在 `tests/tools/_launch_*.sh`。

## 0. 公共环境（每个配方都带）

```bash
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python   # adapter 优先 + 官方源码
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python            # 恒带：rust TreeCore 在 HCU 无预编译扩展必炸
export HCU_AITER_AR=0                                           # aiter 自定义 AR 在本机拒用（自动回退 NCCL）
export CUDA_VISIBLE_DEVICES=6                                   # 按卡改；多卡写 0,1,2,3
```

注意：fp16 底模的 GDN 模型（如 Qwen3.5-122B-AWQ）必须加 `export SGLANG_MAMBA_CONV_DTYPE=float16`（上游 conv 池硬编码 bf16 不看模型 dtype，首 forward 必崩）。

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
```

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
#   多卡必须 --disable-cuda-graph（见 §4 平台约束）。
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
| radix cache | 默认开（恒带 `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`） | ✅ 命中门禁过：重发 cached 256 输出逐字一致、部分前缀命中 |
| MTP 投机解码 | 见 §2.2 | ✅ 精度过；吞吐见 §2.2 注（emu 栈暂负收益，真 FA3 内核预期翻正） |
| hicache 分层 KV | `--enable-hierarchical-cache`（需 adapter 的 get_device_accessible_ptr 桥，已内置） | ⚠ dense 模型全通（逐出-回迁保真）；**GDN 混合模型 MAMBA 状态宿主迁移输出漂移待修，生产 GDN 模型暂不开** |
| AWQ 量化 | `--quantization moe_wna16`（见 §2.1） | ✅ |
| slimquant W4A8 | 零显式旗标（checkpoint 自动识别，nmz18 TP8 graph ON 验证） | ✅ GLM-5-W4A8_V2_6 |
| PD 分离（节点内） | 三件套命令块见下方（mooncake_tcp，无需 IB） | ✅ 数据面（8 路并发正确零串扰）；⚠ `--disaggregation-enable-kv-checksum` 并发误报勿开 |

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
```

## 4. 平台约束（必读）

1. **nmz26 多卡必须 `--disable-cuda-graph`**：TP>1 + cuda graph 输出乱码（NCCL/aiter AR 均坏、triton/fa3 均坏；TP1 graph 正常、TP2 eager 正常；nmz18 TP8 graph 曾通过——两机栈差异待深挖）。
2. **AWQ 永远显式 `--quantization moe_wna16`**（理由见 §2.1）。
3. **fp16 底模 GDN 模型加 `SGLANG_MAMBA_CONV_DTYPE=float16`**。
4. thinking 模型（Qwen3-30B-A3B 等）：锚点判定给足 `max_new_tokens`，长文看 `</think>` 后文本。
5. Qwen3.5 全系是 VLM（视觉塔自动加载，纯文本请求正常工作）。
6. `--page-size` 默认 1 可用；批次 suite 用过 64（R1/flashmla 等场景需要）。
7. DeepSeek-R1-L4 + flashmla（`--decode-attention-backend flashmla --prefill-attention-backend triton --page-size 64`）为**验证 rig**（截层模型语义本坏，只测链路非精度），物料 `tests/tools/_launch_r1l4_*.sh`。
8. **任何启动恒显式 `--attention-backend triton/fa3`**：不指定时默认解析到 aiter，dense prefill 的 `mha_batch_prefill_func` 是模块不可调用，首 forward 必崩（PD prefill 首测实测）。
