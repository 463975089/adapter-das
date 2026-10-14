"""fused-MoE triton config guard — gfx93x 64KB shared-memory clamp.

The upstream fused-MoE triton kernel config resolution
(`try_get_optimal_moe_config`) has no tuned file for HCU device names, and
its NV fraternal default asks ~72KB shared memory per fused_moe_kernel
launch — gfx93x caps at 64KB, so decode/prefill dies at the first MoE
forward with `OutOfResources: shared memory, Required: 73728`.

Neither upstream seam fits: `SGLANG_MOE_CONFIG_DIR` keys files off an exact
(E,N,device,dtype,block,per-channel,down) tuple and the resolution happens
deep per-call, while the module-level `override_config` context manager is
a transient override, not a default. This adapter patches the durable
seam: wrap `try_get_optimal_moe_config` so any config whose implied shared
memory exceeds the gfx93x budget is clamped to a conservative
(BN=BK=64, stages<=2) tile — correctness-neutral, a pure tile-size choice.
"""

import logging

LOG = logging.getLogger("hcu.adapter")

# gfx93x: 64 KiB LDS per workgroup. fused_moe_kernel smem ≈ stages *
# (BM*BK + BN*BK) * dtype_bytes(=1 fp8 / 2 bf16) + epilogue; keep the raw
# tile product well under budget in both dtypes.
_SMEM_BUDGET = 64 * 1024

_SAFE = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
}


def _implied_smem(cfg):
    bm = cfg.get("BLOCK_SIZE_M", 16)
    bn = cfg.get("BLOCK_SIZE_N", 32)
    bk = cfg.get("BLOCK_SIZE_K", 64)
    stages = cfg.get("num_stages", 2)
    # fp8 payloads are 1 byte; bf16 intermediates 2 — assume the fat case.
    return stages * (bm * bk + bn * bk) * 2


def install():
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe_triton_config as fmc,
    )

    if getattr(fmc.try_get_optimal_moe_config, "_hcu_smem_guard", False):
        return

    _orig = fmc.try_get_optimal_moe_config

    def _guarded(*args, **kwargs):
        out = _orig(*args, **kwargs)
        if isinstance(out, tuple):
            config, rest = out
        else:
            config, rest = out, None
        if config and _implied_smem(config) > _SMEM_BUDGET:
            config = dict(_SAFE)
        if rest is not None:
            down, extra = rest
            if down and _implied_smem(down) > _SMEM_BUDGET:
                down = dict(_SAFE)
            return config, (down, extra)
        return config

    _guarded._hcu_smem_guard = True
    fmc.try_get_optimal_moe_config = _guarded
    LOG.warning(
        "HCU adapter: fused-MoE triton config smem guard installed"
        " (clamp to BN=BK=64/stages=2 over %dB budget)",
        _SMEM_BUDGET,
    )
