# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn_2 import (
    AttentionMetadata,
    FlashAttention2Backend,
    FlashAttention2Impl,
)


def _dtk_forward_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    cu_seqlens = cu_seqlens.to(device=query.device, dtype=torch.int32).contiguous()
    output = flash_attn_varlen_func(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    return output[0] if isinstance(output, tuple) else output


class HcuFlashAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[HcuFlashAttentionImpl]:
        return HcuFlashAttentionImpl


class HcuFlashAttentionImpl(FlashAttentionImpl):
    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        use_skip_softmax, _ = self._request_skip_softmax_threshold()
        if use_skip_softmax:
            return super().forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        return _dtk_forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )


class HcuFlashAttention2Backend(FlashAttention2Backend):
    @staticmethod
    def get_impl_cls() -> type[HcuFlashAttention2Impl]:
        return HcuFlashAttention2Impl


class HcuFlashAttention2Impl(FlashAttention2Impl):
    def __init__(self, *args, dropout_p: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_p = float(dropout_p)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        output = flash_attn_func(
            query,
            key,
            value,
            dropout_p=self.dropout_p,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return output[0] if isinstance(output, tuple) else output

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        return _dtk_forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=self.dropout_p,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
