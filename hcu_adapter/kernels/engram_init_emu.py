# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Native Engram initialization providers for the pinned tile-kernels API."""

import torch


def fused_weight(weight_hidden: torch.Tensor, weight_embed: torch.Tensor) -> torch.Tensor:
    """Multiply BF16 norm weights after conversion to FP32, as the GPU kernel does."""
    if weight_hidden.ndim != 2 or weight_hidden.shape != weight_embed.shape:
        raise ValueError("Engram norm weights must have the same [hc_mult, hidden_size] shape")
    if weight_hidden.dtype != torch.bfloat16 or weight_embed.dtype != torch.bfloat16:
        raise TypeError("Engram norm weights must be BF16")
    if weight_hidden.device != weight_embed.device:
        raise ValueError("Engram norm weights must share a device")
    return weight_hidden.float() * weight_embed.float()


def make_offsets(vocab_sizes: torch.Tensor) -> torch.Tensor:
    """Return each layer's flattened, exclusive INT32 table offsets."""
    if vocab_sizes.ndim != 3 or vocab_sizes.shape[-1] * vocab_sizes.shape[-2] == 0:
        raise ValueError("Expected nonempty [layers, max_ngram_size - 1, tables] vocabulary sizes")
    if vocab_sizes.dtype != torch.int32:
        raise TypeError("Engram vocabulary sizes must be INT32")
    flat = vocab_sizes.flatten(1)
    first = torch.zeros_like(flat[:, :1])
    return torch.cat((first, flat[:, :-1].cumsum(-1, dtype=torch.int32)), dim=-1)
