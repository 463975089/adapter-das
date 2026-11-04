# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DTK aiter fused-MoE bridge (recipe ported from the das fork's
``fused_experts_impl_aiter`` hook in moe_runner/triton_utils/fused_moe.py).

What it does
------------
When the stock triton MoE runner would call ``fused_experts`` for a
w8a8 (fp8 or int8) bf16-activation MoE layer, this bridge first tries the
DTK aiter ``aiter_moe`` kernel:

  * blockwise quant (``block_shape=[128,128]``) -> ASM solution, keyed by
    the tuned_fmoe_asm_w8a8_group.csv rows;
  * channelwise quant (``block_shape=None``) -> moe_c solution, keyed by
    the moe_c_configs JSONs.

``get_aiter_moe_config`` returning no solution (shape absent from the
tuned tables) is NOT an error: the bridge falls back to the stock triton
kernel and logs the miss once per shape.

Validation (nmz18, BW200/gfx938, dtk26041 image 20261010):
  E=256 / inter=512 / K=4096 / topk=8, fp8_w8a8 blockwise, solution=asm
  need_shuffle=False -> rel 0.043 (bf16 in, dynamic act quant inside
  kernel) and rel 0.024 (pre-quantized fp8 + per-token-group a1_scale)
  against a dequantized bf16 reference.
GLM-5.3-Flash MoE shapes (E=288 inter-per-rank 256; E=36 inter 2048 for
EP8) are absent from both tuned tables -> they ride the triton fallback
until Hygon ships csv rows (or a local tuning pass fills them).

Knob: HCU_FUSED_MOE_IMPL = auto | aiter | triton (auto default).
"""

from __future__ import annotations

import logging
import os

import torch

LOG = logging.getLogger("hcu.adapter")

_MODE_TRITON = ("triton", "torch", "off", "0")
_MODE_AITER_ONLY = ("aiter",)

# (E, N1, N2, K, top_k, block_size) keys already logged as "no aiter
# config" -- keep the serve log at one line per shape.
_miss_log_once: set = set()
_hit_log: set = set()

# tensors already converted to the moe_c shuffled layout carry this attr
_SHUF_FLAG = "_hcu_aiter_moe_c_shuffled"


def _mode() -> str:
    from . import hcu_env

    v, _src = hcu_env.resolve("HCU_FUSED_MOE_IMPL")
    return (v or "auto").strip().lower()


def _dbg(reason: str):
    if os.environ.get("HCU_AITER_MOE_DEBUG", ""):
        LOG.warning("HCU aiter moe debug: %s", reason)


def _activation_str(activation) -> str:
    """stock MoeRunnerConfig.activation may be int / str / enum."""
    if isinstance(activation, bool):
        return "silu"
    if isinstance(activation, int):
        return {0: "silu", 1: "gelu", 2: "situ"}.get(activation, "silu")
    s = str(activation).split(".")[-1].lower()
    if "gelu" in s:
        return "gelu"
    if "situ" in s:
        return "situ"
    return "silu"


def _shuffled(w: torch.Tensor) -> torch.Tensor:
    cached = getattr(w, _SHUF_FLAG, None)
    if cached is not None and cached.shape == w.shape:
        return cached
    from aiter.ops.shuffle import moe_layout_shuffle_gemm2

    out = moe_layout_shuffle_gemm2(w).view(w.shape)
    try:
        setattr(w, _SHUF_FLAG, out)
    except Exception:
        pass  # plain tensors (tests) just skip the cache
    return out


def try_aiter_fused_moe(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output,
    moe_runner_config,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    w1_scale,
    w2_scale,
    a1_scale,
    block_shape,
    b1=None,
    b2=None,
):
    """Run the DTK aiter fused-MoE for this call, or return None to fall
    back to the stock triton kernel."""
    mode = _mode()
    if mode in _MODE_TRITON:
        _dbg(f"mode={mode}")
        return None

    if (
        hidden_states.dtype != torch.bfloat16
        or not (use_fp8_w8a8 or use_int8_w8a8)
        or b1 is not None
        or b2 is not None
        or w1.dim() != 3
        or getattr(moe_runner_config, "no_combine", False)
    ):
        _dbg(f"guard dtype={hidden_states.dtype} fp8={use_fp8_w8a8} "
             f"i8={use_int8_w8a8} b1={b1 is not None} b2={b2 is not None} "
             f"dim={w1.dim()} no_combine="
             f"{getattr(moe_runner_config, 'no_combine', False)}")
        return None
    # v1 contract (validated): kernel quantizes activations itself
    if a1_scale is not None:
        _dbg("a1_scale set (pre-quant path) -> triton")
        return None

    try:
        from aiter.moe import MoeQuantType, aiter_moe, get_aiter_moe_config
    except ImportError:
        _dbg("aiter not importable")
        return None

    if use_fp8_w8a8:
        quant_type = getattr(MoeQuantType, "FP8_W8A8", None)
    else:
        quant_type = getattr(MoeQuantType, "W8A8", None)
    if quant_type is None:
        _dbg("quant enum missing")
        return None

    M, K = hidden_states.shape
    E, N1, _ = w1.shape
    N2 = w2.shape[1]
    topk_weights, topk_ids, _ = topk_output
    top_k = topk_ids.shape[-1]
    block_size = 0 if not block_shape else int(block_shape[1])
    activation = _activation_str(getattr(moe_runner_config, "activation", "silu"))

    try:
        status, cfg = get_aiter_moe_config(
            M=M, E=E, N1=N1, N2=N2, K=K, top_k=top_k,
            block_size=block_size, dtype=hidden_states.dtype,
            quant_type=quant_type, activation=activation,
        )
    except Exception:
        LOG.warning("HCU aiter moe: config lookup raised", exc_info=True)
        return None
    if not status:
        key = (E, N1, N2, K, top_k, block_size)
        if key not in _miss_log_once:
            _miss_log_once.add(key)
            LOG.warning(
                "HCU aiter moe: no tuned config for E=%d N1=%d N2=%d K=%d "
                "topk=%d block=%d -> triton kernel (log-once)", *key
            )
        if mode in _MODE_AITER_ONLY:
            LOG.warning(
                "HCU_FUSED_MOE_IMPL=aiter but shape %s has no aiter config; "
                "falling back to triton for this call", key
            )
        return None

    aw1 = _shuffled(w1) if getattr(cfg, "need_shuffle", False) else w1
    aw2 = _shuffled(w2) if getattr(cfg, "need_shuffle", False) else w2

    routed = getattr(moe_runner_config, "routed_scaling_factor", None)
    if len(_hit_log) < 3:
        _hit_log.add((E, N1, N2, K, top_k, block_size))
        LOG.warning(
            "HCU aiter moe: kernel ENGAGED E=%d N1=%d N2=%d K=%d topk=%d "
            "block=%d sol=%s", E, N1, N2, K, top_k, block_size,
            getattr(cfg, "solution_type", "?"))
    output = aiter_moe(
        hidden_states,
        aw1,
        aw2,
        topk_weights.to(torch.float32),
        topk_ids.to(torch.int32),
        cfg,
        bool(getattr(moe_runner_config, "inplace", False)),
        activation,
        w1_scale,
        w2_scale,
        None,
        None,
        None,  # a1_scale: dynamic act quant inside the kernel
        None,
        block_shape,
        E,
        None,
        float(routed) if routed is not None else 1.0,
        output_dtype=hidden_states.dtype,
        gemm1_alpha=getattr(moe_runner_config, "gemm1_alpha", None),
        gemm1_limit=getattr(moe_runner_config, "gemm1_clamp_limit", None),
    )
    if bool(getattr(moe_runner_config, "inplace", False)):
        return hidden_states
    return output


def install():
    """Wrap stock ``fused_experts`` (triton MoE runner core entry)."""
    if _mode() in _MODE_TRITON:
        LOG.info("HCU aiter moe bridge: off (HCU_FUSED_MOE_IMPL=%s)", _mode())
        return

    mod_name = "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
    pkg_name = "sglang.srt.layers.moe.moe_runner.triton_utils"
    try:
        import importlib

        fm = importlib.import_module(mod_name)
    except ImportError:
        LOG.info("HCU aiter moe bridge: %s not importable -> skip", mod_name)
        return

    orig = getattr(fm, "fused_experts", None)
    if orig is None or getattr(orig, "_hcu_aiter_moe", False):
        return

    def fused_experts_hcu(
        hidden_states, w1, w2, topk_output, moe_runner_config, **kwargs
    ):
        try:
            out = try_aiter_fused_moe(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_output=topk_output,
                moe_runner_config=moe_runner_config,
                use_fp8_w8a8=kwargs.get("use_fp8_w8a8", False),
                use_int8_w8a8=kwargs.get("use_int8_w8a8", False),
                w1_scale=kwargs.get("w1_scale"),
                w2_scale=kwargs.get("w2_scale"),
                a1_scale=kwargs.get("a1_scale"),
                block_shape=kwargs.get("block_shape"),
                b1=kwargs.get("b1"),
                b2=kwargs.get("b2"),
            )
            if out is not None:
                return out
        except Exception:
            # never take the server down for a vendor-kernel experiment
            LOG.warning("HCU aiter moe: call failed -> triton", exc_info=True)
        return orig(hidden_states, w1, w2, topk_output, moe_runner_config, **kwargs)

    fused_experts_hcu._hcu_aiter_moe = True
    fm.fused_experts = fused_experts_hcu
    # the triton_utils package re-exports fused_experts at import time --
    # rebind that copy too so from-imports pick the wrapper up.
    try:
        import importlib

        pkg = importlib.import_module(pkg_name)
        if getattr(pkg, "fused_experts", None) is orig:
            pkg.fused_experts = fused_experts_hcu
    except Exception:
        pass
    LOG.warning("HCU aiter moe bridge: fused_experts wrapped (mode=%s)", _mode())
