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
