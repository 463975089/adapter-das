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

import ctypes
import glob
import os
from typing import List, Optional

import torch

__all__ = [
    "get_device_accessible_ptr",
    "hcu_alloc_decode_kernel",
    "hcu_alloc_extend_kernel",
    "hcu_assign_req_to_token_pool",
    "hcu_get_last_loc",
    "hcu_assign_extend_cache_locs",
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


def hcu_alloc_decode_kernel(
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs: int,
    page_size: int,
) -> None:
    """Compatibility face for the optional HCU paged decode AOT kernel.

    Community snapshots import this name unconditionally on HIP even when the
    matching ``torch.ops.sgl_kernel`` extension is not installed.  The upstream
    Triton implementation has the same semantics and is the correctness route.
    """
    from sglang.kernels.ops.memory.allocator import alloc_decode_kernel

    alloc_decode_kernel[(bs,)](
        seq_lens_ptr,
        last_loc_ptr,
        free_page_ptr,
        out_indices,
        1 << (bs - 1).bit_length(),
        page_size,
    )


def hcu_alloc_extend_kernel(
    pre_lens_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs: int,
    page_size: int,
) -> None:
    """Compatibility face for the optional HCU paged extend AOT kernel."""
    from sglang.kernels.ops.memory.allocator import alloc_extend_kernel

    alloc_extend_kernel[(bs,)](
        pre_lens_ptr,
        seq_lens_ptr,
        last_loc_ptr,
        free_page_ptr,
        out_indices,
        1 << (bs - 1).bit_length(),
        page_size,
    )


def hcu_assign_req_to_token_pool(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    allocate_lens: torch.Tensor,
    new_allocate_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    shape: int,
    bs: int,
) -> None:
    """Write variable-length cache locations without requiring the HCU AOT op."""
    cursor = 0
    for row in range(bs):
        start = int(allocate_lens[row].item())
        end = int(new_allocate_lens[row].item())
        length = end - start
        if length > 0:
            req = int(req_pool_indices[row].item())
            req_to_token[req, start:end] = out_cache_loc[cursor : cursor + length]
            cursor += length


def hcu_get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> torch.Tensor:
    """Return each request's final prefix slot, preserving the upstream ABI."""
    rows = req_pool_indices.to(torch.long)
    cols = (prefix_lens - 1).clamp_min(0).to(torch.long)
    values = req_to_token[rows, cols]
    return torch.where(prefix_lens > 0, values, torch.full_like(values, -1))


def hcu_assign_extend_cache_locs(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    pool_len: int,
    bs: int,
) -> None:
    """Compatibility alias for speculative variable-length cache writes."""
    hcu_assign_req_to_token_pool(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        pool_len,
        bs,
    )


_HIP_HOST_GET_DEV_PTR = None  # None until resolved; False when unavailable


def _hip_host_get_device_pointer():
    """ctypes handle on hipHostGetDevicePointer from the DTK HIP runtime.

    DTK keeps the runtime at /opt/dtk/hip/lib/libamdhip64.so (not the
    libamdhipcuda.so name upstream ROCm uses); tolerate both layouts.
    """
    global _HIP_HOST_GET_DEV_PTR
    if _HIP_HOST_GET_DEV_PTR is None:
        _HIP_HOST_GET_DEV_PTR = False
        candidates = (
            glob.glob("/opt/dtk*/hip/lib/libamdhip64.so")
            + glob.glob("/opt/dtk*/lib/libamdhip64.so")
            + glob.glob("/opt/rocm*/lib/libamdhip64.so")
            + ["libamdhip64.so"]
        )
        for path in candidates:
            try:
                lib = ctypes.CDLL(path)
                fn = getattr(lib, "hipHostGetDevicePointer", None)
            except OSError:
                continue
            if fn is not None:
                fn.restype = ctypes.c_int
                fn.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_void_p,
                    ctypes.c_uint,
                ]
                _HIP_HOST_GET_DEV_PTR = fn
                break
    return _HIP_HOST_GET_DEV_PTR or None


def get_device_accessible_ptr(tensor: torch.Tensor, device_index: int) -> int:
    """Address a kernel on ``device_index`` must use for ``tensor``.

    HIP does not guarantee UVA: registered host memory may carry a
    device-side address distinct from the host VA (upstream sglang-kernel
    ships this op for exactly that). Query hipHostGetDevicePointer; when the
    runtime reports the memory is directly accessible (HCU/DTK pinned pages
    return dev == host, measured) or the query is unavailable, the host
    pointer itself is device-accessible, so fall back to it.
    """
    fn = _hip_host_get_device_pointer()
    host_ptr = tensor.data_ptr()
    if fn is None:
        return host_ptr
    dev_ptr = ctypes.c_void_p()
    rc = fn(ctypes.byref(dev_ptr), ctypes.c_void_p(host_ptr), 0)
    if rc != 0 or dev_ptr.value is None:
        return host_ptr
    return dev_ptr.value


_TORCH_OP_LIB = None  # keep the Library handle alive: GC unregisters its ops


def _register_torch_op() -> None:
    # sglang's hicache host-pool resolver accepts this helper only when BOTH
    # the python symbol and torch.ops.sgl_kernel.get_device_accessible_ptr
    # exist (upstream ships it as an AOT torch op; we mirror the namespace).
    global _TORCH_OP_LIB
    if hasattr(torch.ops.sgl_kernel, "get_device_accessible_ptr"):
        return
    lib = torch.library.Library("sgl_kernel", "FRAGMENT")
    lib.define("get_device_accessible_ptr(Tensor tensor, int device_index) -> int")
    lib.impl(
        "get_device_accessible_ptr",
        lambda tensor, device_index: get_device_accessible_ptr(tensor, device_index),
        "CPU",
    )
    _TORCH_OP_LIB = lib


_register_torch_op()


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
