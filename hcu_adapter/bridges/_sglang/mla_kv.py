from functools import wraps
from types import FunctionType


def install():
    import torch

    from hcu_adapter import hcu_env, registry

    if torch.version.hip is None:
        return
    if hcu_env.resolve("HCU_PLATFORM_SEMANTICS")[0] != "cuda":
        return

    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
    from sgl_kernel.flash_mla_fp8_emu import sparse_decode
    from sgl_kernel.flash_mla_sparse import flash_mla_sparse_fwd

    registry.register(
        "sglang_mla_sparse", flash_mla_sparse_fwd,
        target="sgl_kernel.flash_mla_sparse:flash_mla_sparse_fwd",
        mechanism=registry.MECH_SEAM,
        detail="SGLang-only configured BF16 native sparse MLA with compatibility fallback",
    )

    registry.register(
        "sglang_mla_fp8", sparse_decode,
        target="sgl_kernel.flash_mla:flash_mla_with_kvcache[V32_NO_ROPE]",
        mechanism=registry.MECH_SEAM,
        detail="SGLang-only packed E4M3 sparse attention compatibility",
    )

    original = MLATokenToKVPool._write_mla_kv_buffer
    if getattr(original, "_hcu_packed_mla", False):
        return

    adapted = FunctionType(
        original.__code__, dict(original.__globals__, _is_hip=False),
        original.__name__, original.__defaults__, original.__closure__,
    )
    adapted.__kwdefaults__ = original.__kwdefaults__

    @wraps(original)
    def write(pool, *args, **kwargs):
        if not pool.dsa_kv_cache_store_fp8:
            return original(pool, *args, **kwargs)
        return adapted(pool, *args, **kwargs)

    write._hcu_packed_mla = True
    MLATokenToKVPool._write_mla_kv_buffer = write
    registry.register(
        "platform_detect", write,
        target="sglang.srt.mem_cache.memory_pool:MLATokenToKVPool._write_mla_kv_buffer",
        mechanism=registry.MECH_REBIND,
        detail="packed DSA FP8 KV uses the upstream scaled-row writer",
    )
