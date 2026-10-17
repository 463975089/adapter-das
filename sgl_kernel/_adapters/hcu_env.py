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
    # custom all-reduce
    "HCU_AITER_AR": {
        "env": "HCU_AITER_AR", "default": "0",
        "auto": _auto_aiter_ar,
        "about": "gfx93* 旧 aiter AR 自动拒用（图重放损坏）；=1 显式夺回",
    },
    "HCU_TRITON_AR": {
        "env": "HCU_TRITON_AR", "default": "0",
        "about": "custom AR（sgl_kernel.allreduce，host-shm 信令 + HIP IPC "
                 "拉读）：R-01 判决（2026-10-15，nmz26 tp4 s7nccl 臂）其跨 "
                 "block 汇合计数器是 device 域原子，不传递系统域写序——"
                 "last block 的 PUB/DONE 发布可早于其他 block 的 host-shm "
                 "payload 写落 DRAM，对端间歇读到旧轮数据（Berlin 跳 think "
                 "60%→2.5%，NCCL 臂 1/40 vs custom 48/80），故默认停用走 "
                 "NCCL；=1 显式夺回（修复 ar_ext.cu 汇合协议前勿在生产开）",
    },
    "HCU_TRITON_AR_SLOTS": {
        "env": "HCU_TRITON_AR_SLOTS", "default": "",
        "about": "custom AR 轮转槽数（默认按 WS 取 8/4/2）",
    },
    "HCU_TRITON_AR_BUDGET": {
        "env": "HCU_TRITON_AR_BUDGET", "default": "100000000",
        "about": "custom AR 每次自旋预算（PCIe 轮询次数，超出降级继续并计 "
                 "diag；默认 1e8 ≈ 2.5 分钟）",
    },
    "HCU_TRITON_AR_MAX_KB": {
        "env": "HCU_TRITON_AR_MAX_KB", "default": "512",
        "about": "custom AR 尺寸上限（KB）：≤此值走 custom AR（host-shm，"
                 "小消息快于 NCCL ~2x），更大走 NCCL；0 = 不设限",
    },
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
    "HCU_MQA_LOGITS_IMPL": {
        "env": "HCU_MQA_LOGITS_IMPL", "default": "vec",
        "about": "DSA indexer paged-MQA torch emu 形态：vec=全窗口单遍双 bmm"
                 "（大操作数免转置物化）——decode 图节点 23.9k→3.7k（-84%），"
                 "hipGraphLaunch API p50 540ms→21.6ms，单流 2.3→3.50 tok/s、"
                 "bs8 14.0→18.69；注意 (N,T,D) 转置物化/整窗 fp32 物化在 DTK "
                 "图 replay 下有 ~124ms/核崖（eager 无恙），故 bmm 两侧只允许"
                 "小张量做转置、H 收缩走第二 bmm；N>HCU_MQA_VEC_MAXROWS(32) "
                 "自动回落 loop（prefill 保内存）；loop=(64,2048) 双层 tile 循环"
                 "参考实现（=loop 可回归对照）",
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
                 "|vendor=decode/verify 短窗走 DTK flash_attn wheel（gfx936 "
                 "wheel paged 短窗存在词汤级数值问题，2026-10-12 bw7 实测，"
                 "待 DTK 修）|emu=恒 torch 模拟",
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
        "env": "HCU_KDA_PREFILL_IMPL", "default": "emu",
        "about": "GLM-5.x KDA prefill（chunk_kda，42 层）内核选择："
                 "fork=vendored sglang-das hcu triton（对拍 emu cos 0.99999，"
                 "快 473-1009×；beta 约定=v0.5.21 后端已激活直通；**serving 门 RED**：服务上下文 Berlin 锚点翻转（standalone 全绿含服务形状/步长视图，二分定位 fork-in-serving 独有，待 serving dump 对拍根因），默认不启用）；"
                 "emu=顺序 torch 步进（词汤急救期参考实现，T=2048 要 414ms）；"
                 "stock=官方 v0.5.21 内核（gfx938 Aqk 全错，gate 测试红常驻）。"
                 "旧键 HCU_KDA_PREFILL_EMU=1 等价本键 =emu",
    },
    "HCU_CONV1D_IMPL": {
        "env": "HCU_CONV1D_IMPL", "default": "emu",
        "about": "KDA causal_conv1d_fn（prefill varlen）+ causal_conv1d_update"
                 "（decode 每步）内核选择：fork=vendored sglang-das hcu triton"
                 "（对拍 emu cos 0.99999+/conv 状态位精确相等；fn 16.4×、"
                 "update 4.3×；serving 门同上 RED 待根因）；emu=torch 步进（历史 stopgap，默认）；stock=官方内核"
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
