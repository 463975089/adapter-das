# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Channel-wise W4A16 packed MoE: kwarg rewrite so the stock triton GPTQ
branch engages (box16 dsv41f-INT4 bring-up, 2026-10-24).

Why
---
compressed-tensors pack-quantized int4 MoE with ``strategy="channel"``
(``group_size=-1``) -- e.g. DeepSeek-V4.1-Flash-INT4 -- flows through
``CompressedTensorsWNA16TritonMoE`` with ``block_shape=[0, -1]``. The
tree's ``invoke_fused_moe_kernel`` only takes the GPTQ/AWQ int4 dequant
branch when ``block_shape[1] > 0`` (group-wise); with -1 the u8-packed
weights fall into the plain ``fused_moe_kernel`` and triton aborts at
compile time ("Both operands must be same dtype. Got bf16 and uint8").
First hit is decode CUDA-graph capture, so the server never comes up.

Fix
---
Channel-wise IS the degenerate group-wise layout: one scale per output
row spanning the whole K. The scheme already transposed the scales to
``[E, N, 1]``; rewriting ``block_shape`` to ``[0, G]`` with
``G = max(K_up, K_down)`` makes ``cdiv(K, G) == 1`` on both expert GEMMs,
so the gptq branch fires and the kernel reads ``scale[..., 0]`` for
every k -- numerically identical to channel semantics, zero weight
conversion, capture-safe (pure kwarg rewrite on already-3D scales; the
2D defensive path allocates once, never during replay).

Scope guard: only rewrites ``use_int4_w4a16`` calls whose group is
non-positive; every other face (fp8/int8 w8a8, group-wise int4, bf16)
passes through untouched. The w4a16 deepgemm bridge (10.5) and the
aiter bridge (10) chain through unchanged -- neither engages for int4
(deepgemm arm: group 2K not expandable + eager-only; aiter arm: w8a8
only), so the stock triton kernel this shim fixes is the only tier that
serves channel int4, capture included.

Knob: ``HCU_WNA16_CHANNEL_MOE = 1 | 0`` (default 1; 0 restores stock).
"""

from __future__ import annotations

import logging

import torch

LOG = logging.getLogger("hcu.adapter")

_REWRITE_LOG: set = set()


def _channel_group_fix(w1: torch.Tensor, w2: torch.Tensor, kwargs: dict) -> bool:
    """Rewrite channel-wise int4 kwargs to the degenerate group-wise form.

    Returns True when the call was rewritten. Idempotent: a rewritten
    ``block_shape`` no longer matches the channel predicate.
    """
    bs = kwargs.get("block_shape")
    if bs is not None and len(bs) >= 2 and bs[1] > 0:
        return False  # genuine group-wise: the stock path is correct
    # None / malformed / non-positive group: channel-wise (or degenerate)
    # -- the plain kernel would tl.dot(bf16, u8) instead of dequantizing.
    # packed layout is [..., K // 2] u8 -> logical K; one group must span
    # the larger of the two expert GEMM K dims so cdiv(K, G) == 1 on both.
    group = max(w1.shape[-1], w2.shape[-1]) * 2
    for key in ("w1_scale", "w2_scale"):
        s = kwargs.get(key)
        if s is not None and s.dim() == 2:  # untransposed [E, N] channel scales
            kwargs[key] = s.unsqueeze(-1).contiguous()
    kwargs["block_shape"] = [0, group]
    tag = (tuple(w1.shape), tuple(w2.shape))
    if tag not in _REWRITE_LOG:
        _REWRITE_LOG.add(tag)
        LOG.info(
            "HCU wna16-channel moe: channel int4 -> group_size=%d rewrite "
            "(w1 %s, w2 %s) so the stock triton gptq branch engages",
            group,
            tuple(w1.shape),
            tuple(w2.shape),
        )
    return True


def install():
    """Wrap the stock triton runner's ``fused_experts`` for channel int4."""
    from . import hcu_env

    enabled, _src = hcu_env.resolve("HCU_WNA16_CHANNEL_MOE")
    if (enabled or "1").strip().lower() in ("0", "off", "false"):
        LOG.info("HCU wna16-channel moe rewrite: off")
        return

    import importlib

    mod_name = "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
    try:
        fm = importlib.import_module(mod_name)
    except ImportError:
        LOG.info("HCU wna16-channel moe rewrite: %s not importable -> skip", mod_name)
        return

    orig = getattr(fm, "fused_experts", None)
    if orig is None or getattr(orig, "_hcu_wna16_channel", False):
        return

    def fused_experts_channel(
        hidden_states, w1, w2, topk_output, moe_runner_config, **kwargs
    ):
        if kwargs.get("use_int4_w4a16"):
            _channel_group_fix(w1, w2, kwargs)
        return orig(hidden_states, w1, w2, topk_output, moe_runner_config, **kwargs)

    fused_experts_channel._hcu_wna16_channel = True
    fm.fused_experts = fused_experts_channel
    # triton_utils re-exports fused_experts at package level; rebind that
    # copy so from-imports land in the wrapper, same as the aiter arm
    try:
        pkg = importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.triton_utils"
        )
        if getattr(pkg, "fused_experts", None) is orig:
            pkg.fused_experts = fused_experts_channel
    except ImportError:
        pass
    try:
        from . import registry

        registry.register(
            "w4a16_moe",
            __name__ + ".install.<locals>.fused_experts_channel",
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts",
            "rebind",
        )
    except Exception:
        pass
    LOG.info(
        "HCU wna16-channel moe rewrite: installed (channel-wise int4 gets "
        "block_shape=[0, max(K_up, K_down)] so triton's gptq dequant branch "
        "engages instead of tl.dot(bf16, u8))"
    )
