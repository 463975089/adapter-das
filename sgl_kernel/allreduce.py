"""sgl_kernel.allreduce — custom all-reduce ops for sglang's HIP branch.

sglang's ``custom_all_reduce_ops`` imports this module and dispatches
through it when building ``CustomAllreduce`` (the graph-safe all-reduce
that aiter's custom AR also occupies on ROCm). The triton implementation
lives in ``sgl_kernel._adapters.triton_ar``; this module only re-exports
the interface sglang expects:

* ``init_custom_ar(meta, rank_data, handles, offsets, rank, full_nvlink)``
  and the reg/unreg all-reduce entry points (both take the same staging
  path, so graph-pool addresses need no IPC registration),
* ``allocate_meta_buffer`` / ``get_meta_buffer_ipc_handle`` / ``meta_size``
  (the sync+payload buffer is one HIP-IPC allocation),
* ``register_buffer`` / ``register_graph_buffers`` /
  ``get_graph_buffer_ipc_meta`` (no-ops by design),
* the quick-AR family (stubs — aiter quickreduce is not provided here).
"""

from sgl_kernel._adapters.triton_ar import (  # noqa: F401
    all_reduce_reg,
    all_reduce_unreg,
    allocate_meta_buffer,
    deterministic_all_reduce_reg,
    deterministic_all_reduce_unreg,
    dispose,
    get_graph_buffer_ipc_meta,
    get_meta_buffer_ipc_handle,
    init_custom_ar,
    init_custom_qr,
    meta_size,
    qr_all_reduce,
    qr_destroy,
    qr_get_handle,
    qr_max_size,
    qr_open_handles,
    register_buffer,
    register_graph_buffers,
)

__all__ = [
    "init_custom_ar",
    "all_reduce_reg",
    "all_reduce_unreg",
    "deterministic_all_reduce_reg",
    "deterministic_all_reduce_unreg",
    "dispose",
    "meta_size",
    "register_buffer",
    "get_graph_buffer_ipc_meta",
    "register_graph_buffers",
    "allocate_meta_buffer",
    "get_meta_buffer_ipc_handle",
    "init_custom_qr",
    "qr_get_handle",
    "qr_open_handles",
    "qr_all_reduce",
    "qr_destroy",
    "qr_max_size",
]
