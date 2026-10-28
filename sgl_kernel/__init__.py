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
    silu_and_mul_clamp,
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
from sgl_kernel.gemm_fp8 import fp8_scaled_mm
from sgl_kernel.gemm_int8 import int8_scaled_mm
from sgl_kernel.weak_ref import weak_ref_tensor, weak_ref_tensors  # noqa: F401
from sgl_kernel import utils as _utils_mod  # noqa: F401  # arch helper surface
from sgl_kernel import utils as _sk_utils  # re-export for `sgl_kernel.utils.X`

_sk_utils.is_arch_support_pdl = _utils_mod.is_arch_support_pdl
from sgl_kernel.sampling import (  # §0.5 is_cuda sampler surface
    min_p_sampling_from_probs,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
)
from sgl_kernel.flash_attn import (  # noqa: F401
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    merge_state_v2,
)
from sgl_kernel.flash_mla import (  # noqa: F401
    flash_mla_with_kvcache,
    get_mla_metadata,
)

# Register HCU shims for AMD/NVIDIA-only dependencies that upstream sglang
# imports on HIP devices (`aiter`, `flashinfer`, `flash_attn`, ...). Must run
# before any sglang module import; unimplemented symbol *calls* raise loudly.
# Shims live in the `hcu_adapter` package (shared with the vLLM host bridge);
# they bind torch at module level, so they are gated behind THIS import —
# `import hcu_adapter` / its plugin seam stays torch-free, and the finders
# still install here, before any sglang module import.
from hcu_adapter.stubs import register_all as _hcu_register_stubs

_hcu_register_stubs()
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
    "silu_and_mul_clamp",
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
    # gemm
    "fp8_scaled_mm",
    "int8_scaled_mm",
    # moe
    "topk_softmax",
    "topk_sigmoid",
    "moe_fused_gate",
    "moe_align_block_size",
    "moe_sum",
    "moe_sum_reduce",
]
