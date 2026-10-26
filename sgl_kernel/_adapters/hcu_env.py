"""HCU adapter environment-variable registry: auto-adapt + one manual.

Design contract (customer-facing):
1. ONE manual lists every variable (docs/env-vars.md) — no tribal knowledge.
2. Variables AUTO-ADAPT: platform detection (arch, wheel presence) and
   server args decide the value; a correct deployment needs ZERO env vars.
3. Explicit env always wins over auto rules, for debugging; every resolved
   value is logged once at startup with its source (explicit/auto/default),
   so "what is active right now" is always answerable from the log.

Resolution order: explicit env  >  auto rule (arch/wheel/args probe)  >
default. Never hard-fail on an unknown value: log and fall back.
"""

import logging
import os

LOG = logging.getLogger("hcu.adapter")


def arch_name():
    """gcnArchName of device 0 ('gfx938:sramecc+:xnack-' -> 'gfx938')."""
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        return (
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        )
    except Exception:
        return ""


def has_fp8(arch=None):
    """gfx938+ BW-class DCUs carry the fp8 matrix unit; gfx936 does not."""
    a = arch or arch_name()
    try:
        major = int(a[3:]) if a.startswith("gfx") else 0
    except ValueError:
        return False
    return major >= 938


# ---------------------------------------------------------------------------
# registry: name -> {default, auto(fn->value), env, about}
# A resolved value is a short string (for the startup summary + manual);
# consumers translate it into behavior. `None` value = feature off/skip.
# ---------------------------------------------------------------------------

def _sglang_uses_new_aiter_ca():
    """True when this sglang build's AR dispatch goes through the NEW
    aiter.dist GraphSafe path (source marker check; no aiter import)."""
    try:
        import sglang.srt.distributed.device_communicators.\
            custom_all_reduce as car

        with open(car.__file__, encoding="utf-8", errors="ignore") as f:
            return "aiter.dist.device_communicators" in f.read()
    except Exception:
        return False


def _auto_aiter_ar():
    # gfx93x + OLD aiter layout (sglang dispatch NOT on aiter.dist):
    # CUDA-graph replay corrupts data (1eb038d) -> refuse, NCCL fallback.
    # NEW aiter.dist GraphSafeAiterCustomAllreduce passed the full stress
    # suite incl. graph-replay on gfx938 (2026-10-10, nmz26). Unknown
    # layouts stay refused (conservative); HCU_AITER_AR=1 opts in.
    if arch_name().startswith("gfx93"):
        if _sglang_uses_new_aiter_ca():
            return "allowed(new-aiter stress-passed)"
        return "refused(old-aiter graph bug)"
    return "allowed"


def _auto_quant_impl():
    # hip + importable aiter wheel -> aiter tier (rowwise NameError fix +
    # 15.1x group quant, probe_r09_quant.py 2026-10-19); else upstream.
    try:
        import torch

        if getattr(torch.version, "hip", None) is not None:
            import aiter  # noqa: F401

            return "aiter"
    except Exception:
        pass
    return "stock"


def _auto_ar_norm_impl():
    # fused AR+RMSNorm rides the SAME aiter GraphSafe communicator the
    # plain AR auto-rule already vets: refuse wherever HCU_AITER_AR
    # refuses (old-aiter graph-corrupting layouts), else aiter on hip.
    try:
        import torch

        if getattr(torch.version, "hip", None) is None:
            return "stock"
        from aiter.dist.device_communicators.custom_all_reduce import (  # noqa: F401
            CustomAllreduce,
        )
    except Exception:
        return "stock"
    if _auto_aiter_ar().startswith("refused"):
        return "stock"
    return "aiter"


def _auto_mqa_emu():
    # torch emulation is needed wherever the DTK aiter MQA-logits kernels
    # are unusable: they hard-gate on gfx942/950 (gfx936 lacks fp8 AND
    # gfx938 has fp8 but is still gated by the current wheels). Flip the
    # gfx938 branch to None when the DTK wheel lifts the gate.
    return "on" if arch_name().startswith("gfx93") else None


REGISTRY = {
    # kill switches (triage; see kill_switch.py) — explicit-only by nature
    "HCU_DISABLE_OPS": {
        "env": "HCU_DISABLE_OPS", "default": "",
        "about": "快关总闸：逗号分隔禁用高危算子（aiter_ar,aiter_fa）",
    },
    # runtime dump — path for JSON (<summary>/1 = log a dict, debug tier)
    "HCU_ADAPTER_DUMP": {
        "env": "HCU_ADAPTER_DUMP", "default": "",
        "about": "启动时落 runtime dump JSON（每算子活跃实现+安装方式；1=打进日志）",
    },
    # platform semantics — governance §0.5 is_cuda 路线（platform_seam.py）
    "HCU_PLATFORM_SEMANTICS": {
        "env": "HCU_PLATFORM_SEMANTICS", "default": "cuda",
        "about": "sglang 平台判定语义：cuda=上游判定统一回 CUDA（新模型天然"
                 "走 CUDA 分支，§0.5）；hip=legacy 适配路线（调试/A-B 对照）",
    },
    # model-aware env defaults (R-08 B-tier; model_env.py)
    "HCU_MODEL_ENV": {
        "env": "HCU_MODEL_ENV", "default": "auto",
        "about": "模型感知 env 自动化：ModelRunner init 按 model_config "
                 "setdefault——DSA 模型（hf config 含 index_kpool 等）关 "
                 "SGLANG_DSA_FUSE_TOPK/SGLANG_DISABLE_DSA_INDEXER_FUSION"
                 "（两融合无 HIP 臂）、fp16 GDN 底模设 "
                 "SGLANG_MAMBA_CONV_DTYPE=float16（上游 conv 池硬编码 bf16，"
                 "fp16 首 forward 必崩）；显式 export 永远赢；off=退回手敲",
    },
    # bring-up env guards (box16 campaign 2026-10-25; each acts only on a
    # detectable failure precondition and logs one line when it acts)
    "HCU_NCCL_TOPO_GUARD": {
        "env": "HCU_NCCL_TOPO_GUARD", "default": "1",
        "about": "NCCL_TOPO_FILE 卡数失配守卫（nccl_topo_guard.py）：镜像预"
                 "设全卡拓扑 XML 而可见卡为子集（HIP_VISIBLE_DEVICES=0-3 等）"
                 "时，NCCL 首个集合通信即致命 XGMI 错（could not find GPU "
                 "5000，box16 4-proc 2026-10-25）。守卫数 XML <gpu> 节点 vs "
                 "torch 可见卡数：失配→清空该 env（NCCL 回退 sysfs 自侦，任"
                 "意子集皆正确）+一行日志；匹配→保留不动；判不了→保留。"
                 "=0 关闭（调试层）",
    },
    "HCU_JIT_WATCHDOG": {
        "env": "HCU_JIT_WATCHDOG", "default": "1",
        "about": "冷 JIT watchdog 地板（watchdog_cold.py）：冷容器首捕获 8 "
                 "rank 全量 JIT 编译风暴超 sglang 默认 watchdog 300s→SIGQUIT "
                 "杀树（box16 冷启 16 并发首请求，2026-10-25 实锤）。"
                 "TRITON_CACHE_DIR/TORCH_EXTENSIONS_DIR 文件数均 <200 判冷；"
                 "冷且用户未显式给 --watchdog-timeout（值=300 默认）时抬地"
                 "板到 1500s（box16 实证装得下编译风暴）；显式值恒赢；种子"
                 "warm 盒不受影响；=0 关闭（调试层）",
    },
    "HCU_WNA16_CHANNEL_MOE": {
        "env": "HCU_WNA16_CHANNEL_MOE", "default": "1",
        "about": "channel-wise int4 MoE 改写（wna16_channel_moe.py）："
                 "compressed-tensors pack-quantized strategy=channel（group="
                 "-1，如 DeepSeek-V4.1-Flash-INT4）经 WNA16TritonMoE 给出 "
                 "block_shape=[0,-1]，树上 triton invoke 的 gptq 反量化分支"
                 "只在 block_shape[1]>0 走，u8 packed 权重直落普通 kernel 的 "
                 "tl.dot(bf16,u8) 编译期炸（首中=decode 图捕获，服务起不来，"
                 "box16 2026-10-25）。channel≡整行一组的退化 group-wise：改写"
                 "block_shape=[0,max(K_up,K_down)]（两 GEMM cdiv==1，scheme 转"
                 "置后的 [E,N,1] scale 直配），数值等价、零权重转换、capture "
                 "安全。仅命中 use_int4_w4a16 且 group≤0 的调用面；fp8/int8/"
                 "分组 int4/bf16 不动；=0 还原 stock",
    },
    # custom all-reduce
    "HCU_AITER_AR": {
        "env": "HCU_AITER_AR", "default": "0",
        "auto": _auto_aiter_ar,
        "about": "gfx93* 旧 aiter AR 自动拒用（图重放损坏）；=1 显式夺回",
    },
    "HCU_QUANT_IMPL": {
        "env": "HCU_QUANT_IMPL", "default": "auto",
        "auto": _auto_quant_impl,
        "about": "per-token fp8 量化族 vendor 档（R-09③，quant_aiter 桥）："
                 "auto=hip+aiter 轮可导入则 aiter（sglang_per_token_quant_"
                 "fp8 在 HIP 本就 NameError=行为还原；group plain 臂尺度与"
                 "金标准逐位一致+q RTN 命中 99.8%，15.1×）；aiter=显式；"
                 "stock=退回上游原生（triton raw group + torch scaled "
                 "natives）。异形臂（column_major/tma/ue8m0/masked/非 "
                 "e4m3/3D）恒走上游；aiter 失败 log-once 回退",
    },
    "HCU_AR_NORM_IMPL": {
        "env": "HCU_AR_NORM_IMPL", "default": "auto",
        "auto": _auto_ar_norm_impl,
        "about": "融合 AR+RMSNorm vendor 档（R-09①，ar_norm_aiter 桥）："
                 "auto=hip+aiter GraphSafe 类可导入且 HCU_AITER_AR auto 规"
                 "则未拒（旧布局图重放损坏史连带拒融合）则 aiter——对 "
                 "fp64 精确 AR 真值 norm 1ulp/residual 0ulp（优于 rccl "
                 "3-8ulp），vs customAR+norm 1.38-2.11×（hidden=4096 "
                 "t=1..4096），图回放新鲜数据精确；仅全 TP 组目标融合"
                 "（dp-attention/attn-tp 切分或 MoE 子组时逐调用退回 "
                 "stock flashinfer 入口）；=stock 全退上游",
    },
    # （HCU_TRITON_AR/HCU_TRITON_AR_SLOTS/BUDGET/MAX_KB 已随 custom AR 除役
    #   撤销——2026-10-19 R-13#1 改判：aiter GraphSafe 在新镜像压测全绿，
    #   homegrown custom AR 的 host-shm 汇合协议修复冻结不实施，代码删除，
    #   git 史 triton_ar.py/ar_ext.cu 可考古。）
    # DSA / attention
    "HCU_SPEC_LOC_SAFE": {
        "env": "HCU_SPEC_LOC_SAFE", "default": "1",
        "about": "spec-decode KV 位置安全路由：把 sglang mem_cache allocation "
                 "的 get_last_loc 重绑到纯 torch 变体——stock 在 is_cuda 平台"
                 "语义下走 get_last_loc_triton，其内核在 HIP 上有 Triton 混合"
                 "位宽误编译（树内注释自证），EAGLE+page_size>1 时产生越界 "
                 "last_loc → KV 读写错位（GLM-5.3 MTP 实测 accept len≈1.0 + "
                 "target 轨迹漂移 + 逐次不确定；二分定位 MTP 独立致病）。"
                 "sglang-das 用 hcu_get_last_loc 同向修复；=0 还原 stock 分发",
    },
    "HCU_MOE_ALIGN_FUSED": {
        "env": "HCU_MOE_ALIGN_FUSED", "default": "2048",
        "about": "moe_align_block_size 小形状融合档：topk lanes<=N 时用"
                 "单 CTA triton 核一发做完 hist+scan+fill+block_expert 及"
                 "两个 aten setup fill（fill_/zeros），每层 6 核→1 核"
                 "（decode 44 层=264→44 核/步，约 -220 核/步且削 5×图节点"
                 "/层）。桶映射/扫描数学/哨兵(-1→专属桶→expert_id=-1)/尾块"
                 "值与四件套 bit 级一致；段内 token 序为 FIFO（四件套是原子"
                 "到达序，runner 按行独立不感知）。大形状（prefill）仍走"
                 "四件套。=0 关闭（回归对照），=数字改阈值",
    },
    "HCU_MQA_LOGITS_IMPL": {
        "env": "HCU_MQA_LOGITS_IMPL", "default": "triton",
        "about": "DSA indexer paged-MQA 实现档：vec=全窗口单遍双 bmm"
                 "（大操作数免转置物化）——decode 图节点 23.9k→3.7k（-84%），"
                 "hipGraphLaunch API p50 540ms→21.6ms，单流 2.3→3.50 tok/s、"
                 "bs8 14.0→18.69；注意 (N,T,D) 转置物化/整窗 fp32 物化在 DTK "
                 "图 replay 下有 ~124ms/核崖（eager 无恙），故 bmm 两侧只允许"
                 "小张量做转置、H 收缩走第二 bmm；N>HCU_MQA_VEC_MAXROWS(32) "
                 "自动回落 loop（prefill 保内存）；loop=(64,2048) 双层 tile 循环"
                 "参考实现（=loop 可回归对照）。**2026-10-18 R-02：vec/loop 的"
                 "池解包修正为页级+交错感知**（writer 普查：GLM-5.3 "
                 "IndexerKPool/kpool=4 不走 base store/fused_store，池由 kpool "
                 "写族+SetKAndS 在强制 preshuffle 门下写=16-tile 交错 k+页尾 "
                 "fp32 scale，prefill gather 同序一致；旧行级 132 字节分裂"
                 "k 每 slot 错位+scale 读到 k 字节——**生产解码 DSA top-k 一直"
                 "确定性错误**，锚点对 topk 不敏感未暴露；`HCU_DSA_PRESHUFFLE="
                 "0` 时切行主序臂=kpool=1/fused_store 世界，PS=1 字节不变）**；"
                 "triton=自研 triton 档（mqa_triton.py，臂随 HCU_DSA_PRESHUFFLE"
                 " 同一开关：默认交错=生产布局、=0 行主序）；对金标准精确 "
                 "rel=0.0、解码形状 13.2× 快于 emu-vec（0.19 vs 2.52ms），"
                 "**production 默认档**（2026-10-18 R-02 validated → R-13 #4 "
                 "晋级翻默认：serving A/B 四门绿+马拉松 0/20+长文对，解码 "
                 "+9% 短/+4% 长对 emu-vec；=loop 可回归对照）；哨兵 "
                 "test_mqa_triton_gfx938.py（含修复后 emu 交叉验证）；"
                 "vendor=deepgemm 轮直调用取证档（mqa_vendor.py 桥；其布局="
                 "行主序+页尾=kpool=1 世界同构，kpool 模型还需池重排+轮子 bug"
                 " 定罪——q/k 负值 fp8 字节被钳为 0，真实数据垃圾 rel~3，"
                 "待 DTK 修复；catalog 档 broken 选配即拒；双事实哨兵 "
                 "test_mqa_vendor_gfx938.py 钉住）",
    },
    "HCU_MQA_EMU": {
        "env": "HCU_MQA_EMU", "default": "0",
        "auto": _auto_mqa_emu,
        "about": "MQA logits torch 模拟：gfx93* 自动开（aiter 核门挡 gfx942/950）",
    },
    "HCU_FA3_EMU": {
        "env": "HCU_FA3_EMU", "default": "1",
        "about": "fa3 attention backend torch 模拟（sgl_kernel.flash_attn emu）："
                 "开=设 SGLANG_USE_SGL_FA3_KERNEL=1 并放行 _is_fa3_supported；"
                 "=0 回上游行为（HIP 上 fa3 必 raise/落 fa2 坏 wheel）",
    },
    "HCU_FA_IMPL": {
        "env": "HCU_FA_IMPL", "default": "auto",
        "about": "fa3 注意力面档位：auto=emu（宽窗；缺 vendor 轮自动全 emu）"
                 "|triton=分页 decode/宽窗 + ragged varlen 预填充走自研 "
                 "triton 核（paged_attn_triton.py / fa_varlen_triton.py，"
                 "契约超界面回 emu）|vendor=decode/verify 短窗走 DTK "
                 "flash_attn wheel（gfx936 wheel paged 短窗存在词汤级数值"
                 "问题，2026-10-12 bw7 实测，待 DTK 修）|emu=恒 torch 模拟。"
                 "**2026-10-19 serving A/B（glm53f tp4）**：≤8k ctx 质量+"
                 "性能平价（锚点/解码 8.27 tok/s 完全一致/prefill ±10%）；"
                 "首 boot 揪出 ops 重绑丢 q_v 面包参（only_qv+qv→emu 断言"
                 "杀捕获，2759711 修）；长上下文红牌（11k 2/2 boot 毙服务）"
                 "已随 R-13 #2 修复全臂击穿——根因是 emu 的 (maxL×C) 板未给"
                 "q 行分块（22.02GiB@8192 面），q 行子分块（c59135b，peak "
                 "6.80GiB）修后 11k 三连绿+LONGCTX-6k 28.4→3.1s（~9×）+锚"
                 "点与 auto 臂逐字一致、解码 8.27 tok/s 不变；档位守 "
                 "candidate——glm53f 的 q_v paged 面恒落 emu，翻默认对本模型"
                 "无感，待行使 triton 核的模型过 serving A/B 再晋级",
    },
    "HCU_LINEAR_ATTN_GDN": {
        "env": "HCU_LINEAR_ATTN_GDN", "default": "auto",
        "about": "GDN/gated-delta-rule 线性注意力档位：auto|triton=上游 triton "
                 "fla（现状默认）；aiter=aiter CK 核路由（fwd_h 状态递推 + "
                 "packed decode；K=V=128，state pool gather/scatter）",
    },
    "HCU_DSA_PRESHUFFLE": {
        "env": "HCU_DSA_PRESHUFFLE", "default": "1",
        "about": "DSA aiter-preshuffle 路径（DTK wheel 内核可用时）；=0 回 legacy",
    },
    "HCU_DSA_PAGE_SIZE": {
        "env": "HCU_DSA_PAGE_SIZE", "default": "",
        "about": "DSA page_size 覆盖（CLI --page-size 等效，须为 index_kpool 倍数）",
    },
    # quantization
    "HCU_SLIMQUANT": {
        "env": "HCU_SLIMQUANT", "default": "1",
        "about": "slimquant_w4a8 方法注册（按 checkpoint 配置自选）；=0 关闭",
    },
    # vendor op backends (替换矩阵档位: auto|triton|lightop|torch)
    "HCU_SILU_MUL_IMPL": {
        "env": "HCU_SILU_MUL_IMPL", "default": "auto",
        "about": "silu_and_mul 后端档位：auto=triton（lightop 过验证后转正），"
                 "triton|lightop|torch",
    },
    "HCU_RMS_NORM_IMPL": {
        "env": "HCU_RMS_NORM_IMPL", "default": "triton",
        "about": "rms_norm 后端档位：triton|aiter(rmsnorm_forward 桥)|torch；"
                 "fused_add_rmsnorm/gemma 变体恒走 triton(torch)",
    },
    "HCU_FUSED_MOE_IMPL": {
        "env": "HCU_FUSED_MOE_IMPL", "default": "auto",
        "about": "fused MoE（triton runner 入口）后端档位：auto=先试 DTK "
                 "aiter(aiter_moe，块量化走 ASM 调优表/通道量化走 moe_c)，"
                 "无调优形状自动回 triton；aiter=同上但回退时告警；triton=关闭",
    },
    "HCU_W4A16_MOE_IMPL": {
        "env": "HCU_W4A16_MOE_IMPL", "default": "auto",
        "about": "W4A16 MoE（moe_wna16，AWQ/GPTQ-int4 线）专家 GEMM 档："
                 "auto=现答 triton（stock MoeWNA16 臂不变）；**moe_c**="
                 "aiter 轮 moe_c w4a16 marlin 厂商 GEMM 桥（moe_c_w4a16."
                 "py，box16 dsv41f-INT4 2026-10-26：align→厂商GEMM1→dsv4 "
                 "精确 clamp-silu→厂商GEMM2→sum，探针 7×于 triton 8.1ms/"
                 "层（M16 全层 1161us）、parity med 0.0055、capture 安全；"
                 "守卫=g32 网格（channel 网格按值复制展开）、gemm1 K/32"
                 "%16==0、N%256、I≥256 且 I 非 512 倍数时 K 补零到 512 倍"
                 "数（nibble 8 数学等价，2304→2560））；**deepgemm**="
                 "R-05 桥（DTK 轮 masked w4a16 grouped C++ gemm 作专家 GEMM"
                 "，自管 dispatch/silu/combine 小管线，eager-only v1：捕获"
                 "路径与 inplace 恒回 triton）——接手条件=无零点对称臂"
                 "（(q−8)·s 与轮同规）、g∈{32,64,128}（64/128 载入期一次性"
                 "扩到 32 格）、K%128/I%128/2I%16 对齐、qweight 定格即"
                 "轮 b 格式零转换（marlin repack+scale 展开缓存于权重张量"
                 "）；triton=关闭此桥",
    },
    "HCU_PREFILL_CP": {
        "env": "HCU_PREFILL_CP", "default": "0",
        "about": "prefill 上下文并行（--enable-prefill-cp --cp-strategy "
                 "zigzag|interleave --attn-cp-size N + fa3 系 attention）："
                 "=1 开 HIP 平台门 + 两处 CP 语义补丁（rope 融合守卫、"
                 "enable_fused_set_kv_buffer CP 守卫——否则 CP 下没人写 KV 池）；"
                 "=0 保持上游拒绝行为",
    },
    "HCU_PREFILL_CP_DEBUG": {
        "env": "HCU_PREFILL_CP_DEBUG", "default": "",
        "about": "CP 排查插桩：zigzag 策略各阶段（shard/gather/"
                 "materialize_full_kv）校验和日志（prefill_cp.py piece 4）",
    },
    "HCU_FA_EMU_DEBUG": {
        "env": "HCU_FA_EMU_DEBUG", "default": "",
        "about": "fa3 emu 调用日志（q 形状/cu_seqlens/cache_seqlens 校验和；"
                 "graph capture 期间自动静默防 D2H 崩溃）",
    },
    # disaggregation / EP transfer
    "HCU_DEEPEP_BRIDGE": {
        "env": "HCU_DEEPEP_BRIDGE", "default": "1",
        "about": "deep_ep API 桥：sglang v0.5.21 low_latency_dispatch(use_fp8)"
                 " -> DTK fork（quant_type/topk_weight 位置参）调用形翻译；=0 关闭",
    },
    "HCU_DEEPGEMM_BRIDGE": {
        "env": "HCU_DEEPGEMM_BRIDGE", "default": "0",
        "about": "DeepGEMM 桥（EP 显式开启）：DTK deepgemm 别名为 deep_gemm +"
                 " wrapper ENABLE_JIT_DEEPGEMM 翻转，解锁 deepep MoE runner"
                 "（bf16 grouped GEMM 已验；fp8 scale 布局未验会响亮报错）；"
                 "=1 开 =0 关",
    },
    "HCU_TRITON_CONTIG": {
        "env": "HCU_TRITON_CONTIG", "default": "1",
        "about": "DeepEP normal/auto prefill 的 contiguous grouped GEMM 用"
                 " triton 内核（DTK w16a16 contiguous 内核坏：2D VMFault/3D"
                 " 全错；同时补 mk-alignment get/set API）；=0 退回 DTK 原核"
                 "（会崩）作对照",
    },
    "HCU_KDA_PREFILL_IMPL": {
        "env": "HCU_KDA_PREFILL_IMPL", "default": "fork",
        "about": "GLM-5.x KDA prefill（chunk_kda，42 层）内核选择："
                 "fork=vendored sglang-das hcu triton（对拍 emu cos 0.99999，"
                 "快 473-1009×；beta 约定=v0.5.21 后端已激活直通；"
                 "**2026-10-18 转正为 hip 默认**——10-16 serving 翻转已归因 custom AR "
                 "store 序 bug（160 臂 A/B：custom 60/100 vs 非 custom 1/60），宿主复位后干净复验：双 fork 锚点 4/4+马拉松 0/20+长文 4972 tok 对）；"
                 "emu=顺序 torch 步进（=emu 退路，T=2048 要 414ms）；"
                 "stock=官方 v0.5.21 内核（gfx938 Aqk 全错，gate 测试红常驻）。"
                 "旧键 HCU_KDA_PREFILL_EMU=1 等价本键 =emu",
    },
    "HCU_CONV1D_IMPL": {
        "env": "HCU_CONV1D_IMPL", "default": "fork",
        "about": "KDA causal_conv1d_fn（prefill varlen）+ causal_conv1d_update"
                 "（decode 每步）内核选择：fork=vendored sglang-das hcu triton"
                 "（对拍 emu cos 0.99999+/conv 状态位精确相等；fn 16.4×、"
                 "update 4.3×；**2026-10-18 转正为 hip 默认**——污染态 9-10/20 翻转观测作废（KFD 坏态数据），干净复验 conv-only 臂 0/20+Tokyo+长文对）；emu=torch 步进（=emu 退路）；stock=官方内核"
                 "（update cos 0.26 / fn 初态臂死，gate 红常驻）。旧键 "
                 "HCU_CONV1D_{UPDATE,FN}_EMU=1 等价本键 =emu",
    },
    "HCU_KPOOL_WRITE_EMU": {
        "env": "HCU_KPOOL_WRITE_EMU", "default": "0",
        "about": "DSA kpool 指数 K 缓存三写内核的 torch emu。初判 kernel 坏后"
                 "复核为 probe 侧 buf 布局错误（kernel 传 (pages,8448) 2D，"
                 "shape[1]=8448 正确；误造 (pages,64,1,384) 四维 probe 使"
                 " shape[1]=64 当页字节数）——kernel 配合正确 loc 写到位，"
                 "默认=0 直通 triton；留作内核回归 watch",
    },
    "HCU_KPOOL_READ_EMU": {
        "env": "HCU_KPOOL_READ_EMU", "default": "1",
        "about": "index-K 缓存 gather 读（GetKAndS.aiter → aiter "
                 "cp_gather_indexer_k_quant_cache preshuffle 臂）走 torch emu："
                 "该 aiter 内核在 gfx938 非法地址访问/返回零（定罪 probe GPU4）"
                 "→ prefill topk score 全零 → 词汤。emu 用与写侧同一 16-tile "
                 "映射解排（CPU/GPU roundtrip 全等）。=0 退回 aiter 内核",
    },
    "HCU_MHC_EMU": {
        "env": "HCU_MHC_EMU", "default": "0",
        "about": "GLM-5.3-Flash MHC 超连接 mix/combine 走 torch emu：官方 "
                 "fused_hc_mix（persistent kernel + device-scope atomics）与 "
                 "hc_combine（.cuh tvm_ffi JIT）只按形状+is_cuda 门控，HIP 上"
                 "直接启用且 gfx938 正确性未验证；每 forward 45 层×2 次调用，"
                 "combine 微错=残差流渐变污染（rms 1.4→5.4、max→194、平分布"
                 "词汤签名）。emu 为 hyperconnection.py 同文件参考数学的 fp32 "
                 "转录。=0 走官方内核（注：glm5_next 实际不消费这两个内核，"
                 "该开关是无效操作，留作其他 mHC 模型用）",
    },
    "HCU_MHC_UNFUSE": {
        "env": "HCU_MHC_UNFUSE", "default": "1",
        "about": "强制 SGLANG_OPT_FUSE_MHC_POST_PRE=false（HIP 上）：跨层融合"
                 "门控 is_cross_layer_mhc_fusion_enabled 只查环境开关（全默认"
                 "True）不查 is_hip，gfx938 上每层 attn→ffn 边界都跑 aiter "
                 "triton fused mhc_post_pre，实测（HCU_DUMP_MHC dump 对拍）其"
                 "FFN 段 h_post 与 _mhc_pre_torch 参考在相同输入下 z 偏移"
                 "+0.5~+5——每层污染残差流，即 GLM-5.3-Flash 词汤主病灶。"
                 "关融合后走 hc_post+hc_ffn_pre 纯 torch 链（HCU 上 aiter 需 "
                 "gfx950、tilelang 被 not is_hip 挡）。=0 恢复融合快路",
    },
    # debug dumps (see skills/hcu-adapter-dev/references/precision-triage.md)
    "HCU_DUMP_LAYERS": {"env": "HCU_DUMP_LAYERS", "default": "0",
                        "about": "层 residual 流 dump（精度定位用）"},
    "HCU_DUMP_INNER": {"env": "HCU_DUMP_INNER", "default": "0",
                       "about": "L0/L1 xa/xb 内部张量 dump"},
    "HCU_DUMP_TOPK": {"env": "HCU_DUMP_TOPK", "default": "0",
                      "about": "DSA topk 输入输出样本 dump"},
    "HCU_DUMP_MQA": {"env": "HCU_DUMP_MQA", "default": "0",
                     "about": "MQA emu 调用日志"},
}


def resolve(name):
    """-> (value, source) where source in {'explicit','auto','default'}.

    The returned value is the EFFECTIVE setting as a string ('1'/'0'/''
    or a short descriptive value for auto rules); consumers map it to
    behavior. Auto rules only fire when the env var is UNSET.
    """
    spec = REGISTRY.get(name)
    if spec is None:
        return None, "unknown"
    env = spec["env"]
    if os.environ.get(env, "") != "":
        return os.environ[env], "explicit"
    # config file (deployment tier): below debug env, above auto rules.
    # Invalid files raise here on first touch -- loud and early.
    from . import hcu_config as _cfg

    cv = _cfg.lookup(env)
    if cv is not None:
        return str(cv), "config"
    if "auto" in spec:
        try:
            v = spec["auto"]()
            if v is not None:
                return str(v), "auto"
        except Exception:
            pass
    return str(spec["default"]), "default"


def enabled(name):
    """Boolean convenience: tri-state envs count '0'/'' as off."""
    v, _ = resolve(name)
    return v not in ("", "0", "None", "refused")


def log_summary():
    """Startup lines: config file in effect + every resolved var + source."""
    from . import hcu_config as _cfg

    cfg_line = _cfg.summary_line()
    if cfg_line:
        LOG.warning(cfg_line)
    parts = []
    for name in REGISTRY:
        v, src = resolve(name)
        if v in ("", "0"):
            continue  # keep the line short: only non-default-effective vars
        parts.append(f"{name}={v}({src})")
    if parts:
        LOG.warning("HCU adapter env: " + " ".join(parts))
    else:
        LOG.warning("HCU adapter env: all defaults (no explicit HCU_* vars)")
    from . import registry as _reg

    _reg.log_summary(LOG)
    out = os.environ.get("HCU_ADAPTER_DUMP", "")
    if out:
        d = _reg.dump(out if out != "1" else None)
        if out == "1":
            LOG.warning("HCU adapter dump: %s", d)
