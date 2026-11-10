# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Framework-free FlashMLA sparse prefill ABI adapter (gathered BF16 KV).

No cache format conversion, RoPE or attention math lives here. Unsupported
metadata returns False before touching output; vendor launch errors propagate.
"""
from __future__ import annotations

import torch

SUPPORTED_ARCHES = frozenset({'gfx936', 'gfx938'})
SUPPORTED_HEADS = frozenset({4, 8, 16, 64, 128})


def _supported_device(q):
    if q.device.type != 'cuda' or torch.version.hip is None:
        return False
    return (torch.cuda.get_device_properties(q.device).gcnArchName.split(':')[0]
            in SUPPORTED_ARCHES)


def try_prefill(q, kv, indices, topk_length, scale, head_dim, nope_head_dim,
                rope_head_dim, attn_sink, output, ragged_indices=None,
                ragged_indptr=None, *, provider):
    """Adapt dense selectors without mutating caller indices or reading GPU data.

    The provider's optional topk_length can still read NaN padding in its tiled
    path. Mask inactive columns to -1 as well as passing the original lengths.
    Clone even without lengths: the vendor Python wrapper may sort in place.
    """
    if (ragged_indices is not None or ragged_indptr is not None
            or (head_dim, nope_head_dim, rope_head_dim) != (512, 448, 64)):
        return False
    if (q.ndim != 3 or kv.ndim != 3 or kv.shape[1:] != (1, 512)
            or q.shape[-1] != 512 or q.shape[0] == 0 or kv.shape[0] == 0
            or q.shape[1] not in SUPPORTED_HEADS or output.shape != q.shape
            or q.dtype != torch.bfloat16 or kv.dtype != q.dtype
            or output.dtype != q.dtype or kv.device != q.device
            or output.device != q.device or q.stride(-1) != 1
            or kv.stride(-1) != 1):
        return False
    if (indices is None or indices.ndim not in (2, 3)
            or indices.shape[0] != q.shape[0]
            or (indices.ndim == 3 and indices.shape[1] != 1)
            or indices.shape[-1] == 0 or indices.shape[-1] % 128
            or indices.dtype != torch.int32 or indices.device != q.device):
        return False
    if topk_length is not None and (
            topk_length.shape != (q.shape[0],) or topk_length.dtype != torch.int32
            or topk_length.device != q.device):
        return False
    if attn_sink is not None and (
            attn_sink.ndim != 1 or attn_sink.numel() < q.shape[1]
            or attn_sink.dtype != torch.float32 or attn_sink.device != q.device):
        return False
    if not _supported_device(q):
        return False

    dense = indices.reshape(q.shape[0], 1, -1)
    if topk_length is None:
        dense = dense.clone()
    else:
        columns = torch.arange(dense.shape[-1], device=q.device)
        dense = torch.where(columns[None, None, :] < topk_length[:, None, None],
                            dense, -1)
    sink = None if attn_sink is None else attn_sink[:q.shape[1]].contiguous()
    result, _, _ = provider(q, kv, dense, scale, 512, sink,
                           None if topk_length is None else topk_length.contiguous())
    output.copy_(result)
    return True
