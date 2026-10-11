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
    # custom all-reduce
    "HCU_AITER_AR": {
        "env": "HCU_AITER_AR", "default": "0",
        "auto": _auto_aiter_ar,
        "about": "gfx93* 旧 aiter AR 自动拒用（图重放损坏）；=1 显式夺回",
    },
    # DSA / attention
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
    # debug dumps (see references/precision-triage.md)
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
    """One startup line listing every resolved var + its source."""
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
