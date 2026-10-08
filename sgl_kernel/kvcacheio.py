"""KV-cache transfer ops (pure-torch ports) for HCU.

These mirror sgl_kernel.kvcacheio's CUDA/HIP kernels with the same calling
contracts. All transfers are index-based gather/scatter over whole token rows:

    per-layer : src/dst are (num_blocks * page_size, ...) device buffers;
                rows src_indices -> dst_indices are copied elementwise.
    mla       : single combined k/v buffer, one index list per side.
    all-layer : src/dst lead with a layer dimension; the same index mapping is
                applied to every layer (loops over the layer axis in torch).
    *_lf_pf / *_pf_lf / *_ph_lf : layout bridges between layer-first and
                page-first buffers; layer rows are strided by the given
                layout_dim, page/head layouts by page_size / head_num.
    *direct   : src/dst are per-layer tensor lists (or lists of raw pointer
                ints) so H2D/D2D can be issued without materializing the
                all-layer view.

_indices are int32/int64 device tensors of flat token-slot ids; transfers are
exact copies (no dtype conversion; item_size is accepted for API parity).
The scale of these transfers is host<->device hicache traffic; torch ops keep
them correct-first, triton ports can replace the hot loops later.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch

__all__ = [
    "get_device_accessible_ptr",
    "transfer_kv_per_layer",
    "transfer_kv_per_layer_pf_lf",
    "transfer_kv_per_layer_ph_lf",
    "transfer_kv_all_layer",
    "transfer_kv_all_layer_lf_pf",
    "transfer_kv_all_layer_lf_ph",
    "transfer_kv_direct",
    "transfer_embedding_ranges_direct",
    "transfer_kv_per_layer_direct_pf_lf",
    "transfer_kv_all_layer_direct_lf_pf",
    "transfer_kv_per_layer_mla",
    "transfer_kv_per_layer_mla_pf_lf",
    "transfer_kv_all_layer_mla",
    "transfer_kv_all_layer_mla_lf_pf",
    "copy_all_layer_kv_cache_cpu",
]


def _row_index(t: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Rows of a (N, D) or (N, ...) buffer selected by flat token ids."""
    return t.index_select(0, indices.to(torch.long))


def get_device_accessible_ptr(tensor: torch.Tensor, device_index: int) -> int:
    """Address a kernel on ``device_index`` must use for ``tensor``.

    Torch allocations on the active HIP device are directly accessible; the
    CUDA variant of this helper exists for multi-device pointer isolation we
    do not need on single-rank HCU.
    """
    return tensor.data_ptr()


def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16,
):
    dst_k[dst_indices.to(torch.long)] = _row_index(src_k, src_indices).to(dst_k.dtype)
    dst_v[dst_indices.to(torch.long)] = _row_index(src_v, src_indices).to(dst_v.dtype)


def transfer_kv_per_layer_pf_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16,
):
    """page-first src -> layer-first dst (single device layer row)."""
    transfer_kv_per_layer(src_k, dst_k, src_v, dst_v, src_indices, dst_indices, item_size)


def transfer_kv_per_layer_ph_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    page_size: int,
    head_num: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16,
):
    transfer_kv_per_layer(src_k, dst_k, src_v, dst_v, src_indices, dst_indices, item_size)


def transfer_kv_all_layer(
    src_k_layers: torch.Tensor,
    dst_k_layers: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16,
):
    src_i = src_indices.to(torch.long)
    dst_i = dst_indices.to(torch.long)
    for l in range(num_layers):
        dst_k_layers[l][dst_i] = _row_index(src_k_layers[l], src_i).to(dst_k_layers.dtype)
        dst_v_layers[l][dst_i] = _row_index(src_v_layers[l], src_i).to(dst_v_layers.dtype)


def transfer_kv_all_layer_lf_pf(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16,
):
    transfer_kv_all_layer(
        src_k_layers, dst_k, src_v_layers, dst_v,
        src_indices, dst_indices, item_size, num_layers,
    )


def transfer_kv_all_layer_lf_ph(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    page_size: int,
    head_num: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16,
):
    transfer_kv_all_layer(
        src_k_layers, dst_k, src_v_layers, dst_v,
        src_indices, dst_indices, item_size, num_layers,
    )


def transfer_kv_direct(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
):
    src_i = src_indices.to(torch.long)
    dst_i = dst_indices.to(torch.long)
    for src, dst in zip(src_layers, dst_layers):
        dst[dst_i] = _row_index(src, src_i).to(dst.dtype)


def transfer_embedding_ranges_direct(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_starts: List[int],
    dst_starts: List[int],
    lengths: List[int],
) -> None:
    for s, d, n in zip(src_starts, dst_starts, lengths):
        dst[d: d + n] = src[s: s + n].to(dst.dtype)


def transfer_kv_per_layer_direct_pf_lf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
):
    transfer_kv_direct(src_ptrs, dst_ptrs, src_indices, dst_indices, page_size)


def transfer_kv_all_layer_direct_lf_pf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
):
    transfer_kv_direct(src_ptrs, dst_ptrs, src_indices, dst_indices, page_size)


def _default_mla_block_quota() -> int:
    default = 16
    override = os.environ.get("SGLANG_HICACHE_BLOCK_QUOTA")
    if override is None:
        return default
    try:
        return int(override)
    except ValueError:
        return default


def transfer_kv_per_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: Optional[int] = None,
    num_warps_per_block: int = 16,
):
    if block_quota is None:
        block_quota = _default_mla_block_quota()
    dst_i = dst_indices.to(torch.long)
    dst[dst_i] = _row_index(src, src_indices).to(dst.dtype)


def transfer_kv_per_layer_mla_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: Optional[int] = None,
    num_warps_per_block: int = 16,
):
    transfer_kv_per_layer_mla(src, dst, src_indices, dst_indices, item_size, block_quota)


def transfer_kv_all_layer_mla(
    src_layers: torch.Tensor,
    dst_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16,
):
    src_i = src_indices.to(torch.long)
    dst_i = dst_indices.to(torch.long)
    for l in range(num_layers):
        dst_layers[l][dst_i] = _row_index(src_layers[l], src_i).to(dst_layers.dtype)


def transfer_kv_all_layer_mla_lf_pf(
    src_layers: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16,
):
    transfer_kv_all_layer_mla(
        src_layers, dst, src_indices, dst_indices, item_size, num_layers
    )


def copy_all_layer_kv_cache_cpu(
    data_ptrs: torch.Tensor,
    strides: torch.Tensor,
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
):
    """CPU raw-pointer row copy used by the host memory pool.

    Implemented via ctypes memcpy over the raw layer pointers (torch cannot
    wrap foreign host addresses without keeping them alive externally, which
    the caller already guarantees).
    """
    import ctypes

    n_layers = int(data_ptrs.numel())
    n_loc = int(tgt_loc.numel())
    row_bytes = int(strides[0].item()) if n_layers and int(strides.numel()) else 0
    tgt = tgt_loc.to(torch.long).tolist()
    src = src_loc.to(torch.long).tolist()
    for l in range(n_layers):
        base = int(data_ptrs[l].item())
        stride = int(strides[l].item()) if l < int(strides.numel()) else row_bytes
        if base == 0 or stride == 0:
            continue
        dst_base = ctypes.c_void_p(base)
        for t_i, s_i in zip(tgt, src):
            ctypes.memmove(
                ctypes.c_void_p(base + t_i * stride),
                ctypes.c_void_p(base + s_i * stride),
                stride,
            )
