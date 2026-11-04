# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Explicit provider for the FlashAttention entry used by upstream ViT."""

from . import hcu_env
from hcu_adapter import registry


def install():
    import flash_attn.cute.interface as interface
    from hcu_adapter.kernels.vision_attention_emu import flash_attn_fwd

    op = "flash_attn.cute.interface._flash_attn_fwd"
    if op in hcu_env.resolve("HCU_DISABLE_OPS"):
        raise RuntimeError("HCU_DISABLE_OPS disables ViT attention")
    current = getattr(interface, "_flash_attn_fwd", None)
    if current is not None and current is not flash_attn_fwd and not getattr(current, "_hcu_unimplemented", False):
        raise RuntimeError("An existing FlashAttention ViT provider is already installed")
    interface._flash_attn_fwd = flash_attn_fwd
    registry.register(op, "hcu_adapter.kernels.vision_attention_emu.flash_attn_fwd", op, "rebind")
    print("HCU adapter runtime: ViT varlen attention -> bounded FP32 torch candidate", flush=True)
