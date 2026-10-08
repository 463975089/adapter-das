"""sgl_kernel — HCU triton port of the sgl_kernel AOT kernel backend.

Drop-in replacement for the official sgl-kernel wheel on HCU (Hygon DCU,
DTK/HIP stack): same package name, same public functions, triton kernels
(pure-torch fallbacks where noted). sglang's fused-op layer resolves the AOT
backend via ``import sgl_kernel``; with this package installed, upstream
sglang runs on HCU without any source modification.
"""

from sgl_kernel.version import __version__

from sgl_kernel.elementwise import (
    gelu_and_mul,
    gelu_quick,
    gelu_tanh_and_mul,
    silu_and_mul,
)
from sgl_kernel.layernorm import (
    fused_add_rmsnorm,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    rmsnorm,
)
from sgl_kernel.rotary import rotary_embedding
from sgl_kernel.moe import (
    moe_align_block_size,
    moe_fused_gate,
    moe_sum,
    moe_sum_reduce,
    topk_sigmoid,
    topk_softmax,
)

# Register HCU shims for AMD/NVIDIA-only dependencies that upstream sglang
# imports on HIP devices (`aiter`, `flashinfer`, `flash_attn`, ...). Must run
# before any sglang module import; unimplemented symbol *calls* raise loudly.
from sgl_kernel import _adapters

_adapters.register_all()
from sgl_kernel.kvcacheio import (  # noqa: F401
    copy_all_layer_kv_cache_cpu,
    get_device_accessible_ptr,
    transfer_embedding_ranges_direct,
    transfer_kv_all_layer,
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_all_layer_lf_pf,
    transfer_kv_all_layer_lf_ph,
    transfer_kv_all_layer_mla,
    transfer_kv_all_layer_mla_lf_pf,
    transfer_kv_direct,
    transfer_kv_per_layer,
    transfer_kv_per_layer_direct_pf_lf,
    transfer_kv_per_layer_mla,
    transfer_kv_per_layer_mla_pf_lf,
    transfer_kv_per_layer_pf_lf,
    transfer_kv_per_layer_ph_lf,
)

__all__ = [
    "__version__",
    # elementwise
    "silu_and_mul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "gelu_quick",
    # layernorm
    "rmsnorm",
    "fused_add_rmsnorm",
    "gemma_rmsnorm",
    "gemma_fused_add_rmsnorm",
    # rope
    "rotary_embedding",
    # moe
    "topk_softmax",
    "topk_sigmoid",
    "moe_fused_gate",
    "moe_align_block_size",
    "moe_sum",
    "moe_sum_reduce",
]
