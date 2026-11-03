"""HCU paged allocator routing and SGLang Triton compatibility fallbacks.

The shadow ``sgl_kernel`` package owns the public import surface.  This
module owns the optional host-side fallback so the shadow package does not
depend on SGLang internals at import time.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch


LOG = logging.getLogger("hcu.adapter")
_ROUTE_LOGGED: set[str] = set()


def _log_route(name: str, route: str) -> None:
    key = f"{name}:{route}"
    if key not in _ROUTE_LOGGED:
        _ROUTE_LOGGED.add(key)
        LOG.debug("[HCU kvcacheio] %s route=%s", name, route)


def _hcu_alloc_decode_kernel_fallback(
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs: int,
    page_size: int,
) -> None:
    """Use the upstream Triton allocator when the HCU op is absent."""
    from sglang.kernels.ops.memory.allocator import alloc_decode_kernel

    alloc_decode_kernel[(bs,)](
        seq_lens_ptr,
        last_loc_ptr,
        free_page_ptr,
        out_indices,
        1 << (bs - 1).bit_length(),
        page_size,
    )


def _hcu_alloc_extend_kernel_fallback(
    pre_lens_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs: int,
    page_size: int,
) -> None:
    """Use the upstream Triton allocator when the HCU op is absent."""
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


def _hcu_assign_req_to_token_pool_fallback(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    allocate_lens: torch.Tensor,
    new_allocate_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    shape: int,
    bs: int,
) -> None:
    """Write variable-length cache locations without the HCU AOT op."""
    cursor = 0
    for row in range(bs):
        start = int(allocate_lens[row].item())
        end = int(new_allocate_lens[row].item())
        length = end - start
        if length > 0:
            req = int(req_pool_indices[row].item())
            req_to_token[req, start:end] = out_cache_loc[cursor : cursor + length]
            cursor += length


def _hcu_get_last_loc_fallback(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> torch.Tensor:
    """Return each request's final prefix slot."""
    rows = req_pool_indices.to(torch.long)
    cols = (prefix_lens - 1).clamp_min(0).to(torch.long)
    values = req_to_token[rows, cols]
    return torch.where(prefix_lens > 0, values, torch.full_like(values, -1))


def _hcu_assign_extend_cache_locs_fallback(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    pool_len: int,
    bs: int,
) -> None:
    """Compatibility alias for speculative variable-length cache writes."""
    _hcu_assign_req_to_token_pool_fallback(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        pool_len,
        bs,
    )


_FALLBACKS: dict[str, Callable[..., object]] = {
    "hcu_alloc_decode_kernel": _hcu_alloc_decode_kernel_fallback,
    "hcu_alloc_extend_kernel": _hcu_alloc_extend_kernel_fallback,
    "hcu_assign_req_to_token_pool": _hcu_assign_req_to_token_pool_fallback,
    "hcu_get_last_loc": _hcu_get_last_loc_fallback,
    "hcu_assign_extend_cache_locs": _hcu_assign_extend_cache_locs_fallback,
}


def resolve(name: str) -> Callable[..., object]:
    """Return the registered vendor op or the compatibility fallback."""
    fallback = _FALLBACKS[name]
    op = getattr(torch.ops.sgl_kernel, name, None)
    if op is None:
        _log_route(name, "fallback")
        return fallback
    _log_route(name, "vendor")
    return op
