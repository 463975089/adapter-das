"""Pinned import surface with individually registered numerical providers.

No numerical operation in this shim returns a placeholder result. Missing
operators raise at call time, naming the actual unresolved production symbol.
"""

import sys
import types

from . import hcu_env
from hcu_adapter import registry

_MODULES = {
    "": ("set_num_sms", "get_num_sms", "set_pdl", "use_deterministic_algorithms", "set_token_alignment"),
    "config": (),
    "utils": ("dtype_to_str",),
    "quant": ("per_token_cast", "per_token_cast_and_cast_back", "batched_transpose_weight_sf",
              "swiglu_forward", "swiglu_forward_and_per_token_cast"),
    "transform": ("apply_rotary",),
    "torch": ("cast", "cast_back"),
    "torch.engram": ("make_offsets",),
    "engram": ("fused_weight", "engram_gate_fwd", "engram_hash"),
    "moe": ("moe_topk_gate",),
    "mhc": (),
    "mhc.expand_kernel": ("expand_to_mhc_fwd",),
    "mhc.pre_apply_mix_kernel": ("mhc_pre_apply_mix_fwd",),
    "mhc.head_compute_mix_kernel": ("mhc_head_compute_mix_fwd",),
    "mhc.pre_split_mixes_kernel": ("mhc_pre_split_mixes_fwd",),
    "mhc.sinkhorn_kernel": ("mhc_sinkhorn_fwd",),
    "mhc.post_kernel": ("mhc_post_fwd",),
    "mhc.norm_fn_kernel": ("mhc_fn_normw_merge_fwd", "mhc_reduce_partials_and_rmsnorm_fwd"),
    "mhc.pre_big_fuse_kernel": ("mhc_pre_big_fuse_fwd",),
}
_installed = False


def _missing(symbol):
    def unavailable(*args, **kwargs):
        raise NotImplementedError(
            "HCU provider missing for " + symbol + ". The import surface is a stub; "
            "install a verified Triton/native/vendor implementation (see docs/config.md).")

    unavailable.__name__ = symbol.rsplit(".", 1)[-1]
    return unavailable


def _is_ascend():
    return False


def install():
    """Opt in before any real tile_kernels package has been imported."""
    global _installed
    if _installed:
        return
    native = hcu_env.lightop_ops_enabled()
    if native:
        from lightop.version import git_hash

        if git_hash != 'b2f8c9d':
            raise RuntimeError('HCU_LIGHTOP_OPS requires the explicitly qualified LightOp b2f8c9d build')
    if any(name == "tile_kernels" or name.startswith("tile_kernels.") for name in sys.modules):
        raise RuntimeError("Install the HCU tile_kernels surface before importing the vendor package")
    for suffix, symbols in _MODULES.items():
        name = "tile_kernels" + ("." + suffix if suffix else "")
        module = types.ModuleType(name)
        module.__doc__ = "HCU import surface; unimplemented numerical symbols raise on call."
        module.__package__ = name
        module.__path__ = []
        for symbol in symbols:
            setattr(module, symbol, _missing(name + "." + symbol))
        sys.modules[name] = module
        if suffix:
            parent, _, child = name.rpartition(".")
            setattr(sys.modules[parent], child, module)
    sys.modules["tile_kernels.config"].is_ascend = _is_ascend
    from hcu_adapter.kernels.engram_init_emu import fused_weight, make_offsets
    from hcu_adapter.kernels.engram_runtime_emu import engram_gate_fwd, engram_hash
    from hcu_adapter.kernels import mhc_emu as mhc_providers
    from hcu_adapter.kernels.quant_emu import per_token_cast
    if native:
        from hcu_adapter.kernels.quant_lightop import per_token_cast
    from hcu_adapter.kernels.weight_scale_emu import batched_transpose_weight_sf

    for suffix, symbol, provider in (
        ("engram", "fused_weight", fused_weight),
        ("torch.engram", "make_offsets", make_offsets),
    ):
        op = "tile_kernels." + suffix + "." + symbol
        setattr(sys.modules["tile_kernels." + suffix], symbol, provider)
        registry.register(op, "hcu_adapter.kernels.engram_init_emu." + symbol, op, "seam")
    for symbol, provider in (("engram_gate_fwd", engram_gate_fwd), ("engram_hash", engram_hash)):
        op = "tile_kernels.engram." + symbol
        setattr(sys.modules["tile_kernels.engram"], symbol, provider)
        registry.register(op, "hcu_adapter.kernels.engram_runtime_emu." + symbol, op, "seam")
    sys.modules["tile_kernels.quant"].batched_transpose_weight_sf = batched_transpose_weight_sf
    registry.register("tile_kernels.quant.batched_transpose_weight_sf",
                      "hcu_adapter.kernels.weight_scale_emu.batched_transpose_weight_sf",
                      "tile_kernels.quant.batched_transpose_weight_sf", "seam")
    sys.modules["tile_kernels.quant"].per_token_cast = per_token_cast
    registry.register("tile_kernels.quant.per_token_cast", per_token_cast.__module__ + '.per_token_cast',
                      "tile_kernels.quant.per_token_cast", "seam")
    from hcu_adapter.kernels.quant_emu import per_token_cast_and_cast_back, swiglu_forward, swiglu_forward_and_per_token_cast
    if native:
        from hcu_adapter.kernels.swiglu_lightop import swiglu_forward, swiglu_forward_and_per_token_cast

    sys.modules["tile_kernels.quant"].swiglu_forward = swiglu_forward
    registry.register("tile_kernels.quant.swiglu_forward", swiglu_forward.__module__ + '.swiglu_forward',
                      "tile_kernels.quant.swiglu_forward", "seam")

    sys.modules["tile_kernels.quant"].swiglu_forward_and_per_token_cast = swiglu_forward_and_per_token_cast
    registry.register("tile_kernels.quant.swiglu_forward_and_per_token_cast",
                      swiglu_forward_and_per_token_cast.__module__ + '.swiglu_forward_and_per_token_cast',
                      "tile_kernels.quant.swiglu_forward_and_per_token_cast", "seam")

    sys.modules["tile_kernels.quant"].per_token_cast_and_cast_back = per_token_cast_and_cast_back
    registry.register("tile_kernels.quant.per_token_cast_and_cast_back",
                      "hcu_adapter.kernels.quant_emu.per_token_cast_and_cast_back",
                      "tile_kernels.quant.per_token_cast_and_cast_back", "seam")
    from deepinfer import envs
    from hcu_adapter.kernels.moe_gate_emu import moe_topk_gate
    if native:
        from hcu_adapter.kernels.moe_gate_lightop import moe_topk_gate

    if envs.DEEPINFER_FORCE_LOAD_BALANCE:
        raise NotImplementedError("HCU moe_topk_gate does not implement force_random (DEEPINFER_FORCE_LOAD_BALANCE)")
    sys.modules["tile_kernels.moe"].moe_topk_gate = moe_topk_gate
    registry.register("tile_kernels.moe.moe_topk_gate", moe_topk_gate.__module__ + '.moe_topk_gate',
                      "tile_kernels.moe.moe_topk_gate", "seam")
    from hcu_adapter.kernels.rope_triton import apply_rotary
    if native:
        from hcu_adapter.kernels.rope_lightop import apply_rotary

    sys.modules["tile_kernels.transform"].apply_rotary = apply_rotary
    registry.register("tile_kernels.transform.apply_rotary", apply_rotary.__module__ + '.apply_rotary',
                      "tile_kernels.transform.apply_rotary", "seam")
    from hcu_adapter.kernels import runtime_config

    sys.modules["tile_kernels.utils"].dtype_to_str = runtime_config.dtype_to_str
    registry.register("tile_kernels.utils.dtype_to_str", "hcu_adapter.kernels.runtime_config.dtype_to_str",
                      "tile_kernels.utils.dtype_to_str", "seam")

    for symbol, impl in (("set_num_sms", "tile_kernels_set_num_sms"),
                         ("get_num_sms", "tile_kernels_get_num_sms"),
                         ("set_token_alignment", "set_token_alignment"),
                         ("get_token_alignment", "get_token_alignment")):
        provider = getattr(runtime_config, impl)
        # The vendor re-exports config knobs at package level.
        for name in ("tile_kernels", "tile_kernels.config"):
            setattr(sys.modules[name], symbol, provider)
        registry.register("tile_kernels." + symbol, "hcu_adapter.kernels.runtime_config." + impl,
                          "tile_kernels." + symbol, "seam")
    mhc_symbols = {
        "mhc.expand_kernel": ("expand_to_mhc_fwd",),
        "mhc.pre_apply_mix_kernel": ("mhc_pre_apply_mix_fwd",),
        "mhc.head_compute_mix_kernel": ("mhc_head_compute_mix_fwd",),
        "mhc.pre_split_mixes_kernel": ("mhc_pre_split_mixes_fwd",),
        "mhc.sinkhorn_kernel": ("mhc_sinkhorn_fwd",),
        "mhc.post_kernel": ("mhc_post_fwd",),
        "mhc.norm_fn_kernel": ("mhc_fn_normw_merge_fwd", "mhc_reduce_partials_and_rmsnorm_fwd"),
        "mhc.pre_big_fuse_kernel": ("mhc_pre_big_fuse_fwd",),
    }
    for suffix, symbols in mhc_symbols.items():
        module = sys.modules["tile_kernels." + suffix]
        for symbol in symbols:
            provider = getattr(mhc_providers, symbol)
            setattr(module, symbol, provider)
            op = "tile_kernels." + suffix + "." + symbol
            registry.register(op, "hcu_adapter.kernels.mhc_emu." + symbol, op, "seam")
    _installed = True
    registry.register("tile_kernels.import_surface", "hcu_adapter.bridges._deepinfer.tile_kernels.install", "tile_kernels", "seam")
    print("HCU adapter runtime: native weight-scale packing/Engram/mHC; remaining numerical APIs are stubs")
