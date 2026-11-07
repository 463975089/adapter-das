# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Keep audited import-time hardware queries free of CUDA contexts."""

import ast
import hashlib
import importlib.abc
import sys

_AUDIT = {}

_RESOLUTION_TARGETS = {
    "sglang.srt.arg_groups.attention_hook": ("handle_linear_attn_backend",),
    "sglang.srt.arg_groups.model_hook": ("handle_model_specific_adjustments",),
    "sglang.srt.arg_groups.overrides": (
        "_dsa_kv_cache_dtype_default",
        "_check_dsa_backend_constraints",
        "_dsa_split_backend_resolution",
    ),
}

_TARGETS = {
    "vsa",
    "vsa.block_sparse_wrapper",
    "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention",

    "sglang.kernels.ops.diffusion.norm.norm_triton",
    "sglang.kernels.ops.attention.cutedsl_gdn_mtp_ring",
    "triton.backends.driver",
    "sglang.kernels.ops.attention.extend_attention",
    "sglang.kernels.ops.attention.fla.utils",
    "sglang.srt.layers.quantization.quark_int4fp8_moe",
    "sglang.srt.layers.moe.fused_moe_int8_marlin_for_minimax_m2",
    "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe",

    "quack.cute_dsl_utils",
    "sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate",
    "sglang.kernels.ops.attention.prefill_attention",
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe",
    "hcu_sglang.utils.detection",
    "hcu_sglang.layers.quantization.cache_config",
    "hcu_sglang.layers.quantization.hcu_deepgemm_w8a8_utils",
    "hcu_sglang.quick_all_reduce",
    "hcu_sglang.layers.attention.hcu_int8_index_k_cache",
    "hcu_sglang.mem_cache.dsa_index_cache_mode",

    "lightop.envs",
    "lightop._lmslim_native.quantize.utils.template_v2",
    "lightop._lmslim_native.quantize.utils.awq_gemm_get_best_template",
    "lightop._lmslim_native.vllm_compat.w8a8_get_cache_json",
    "lightop._lmslim_native.layers.groupgemm.w4a8",
    "lightop._lmslim_native.layers.fused_moe.int8_marlin",
    "lightop._lmslim_native.layers.fused_moe.int4_cuda",
    "lightop._lmslim_native.layers.fused_moe.w4a8",
    "lightop._lmslim_native.layers.fused_moe.w4a16",
    "lightop._lmslim_native.layers.fused_moe.fp8_marlin",
    "lightop._lmslim_native.layers.fused_moe.int4",
    "lightop._lmslim_native.layers.fused_moe.w4a8_marlin",
}


_TARGETS.update(_RESOLUTION_TARGETS)


class _MetadataCalls(ast.NodeTransformer):
    def __init__(self):
        self.calls = []

    def visit_Attribute(self, node):
        replacements = {
            "torch.get_device_module().get_device_properties": "device_properties",
            "torch.get_device_module().current_device": "current_device",
            "torch.cuda.get_device_properties": "device_properties",
            "torch.cuda.current_device": "current_device",
            "torch.cuda.get_device_capability": "compute_capability",
        }
        original = ast.unparse(node)
        replacement = replacements.get(original)
        if replacement is not None:
            self.calls.append(original)
            return ast.copy_location(ast.Attribute(
                value=ast.Name(id="_hcu_metadata", ctx=ast.Load()),
                attr=replacement, ctx=ast.Load(),
            ), node)
        return self.generic_visit(node)



class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        from hcu_adapter import hip_metadata

        source = self.original.get_source(module.__name__)
        tree = ast.parse(source, filename=module.__file__)
        transform = _MetadataCalls()
        tree = ast.fix_missing_locations(transform.visit(tree))
        _AUDIT[module.__name__] = {
            "source": module.__file__,
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "calls": transform.calls,
        }
        module.__dict__['_hcu_metadata'] = hip_metadata
        exec(compile(tree, module.__file__, 'exec'), module.__dict__)
        from hcu_adapter import registry

        registry.register(
            "platform_detect", "hip-metadata-import", target=module.__name__,
            mechanism=registry.MECH_TEXT,
            detail="AST: context-free HIP metadata queries in audited modules",
        )

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        for finder in tuple(sys.meta_path):
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                spec.loader = _Loader(spec.loader)
                return spec
        return None


def install():
    # meta_size is sizeof(aiter::Signal), but the vendor decorator allocates
    # a CUDA dispatch dummy and fetches a stream even for this host-only query.
    import aiter
    import torch
    from aiter.jit.core import get_module
    from hcu_adapter import registry

    original_meta_size = aiter.meta_size
    if not getattr(original_meta_size, "_hcu_context_free", False):
        def meta_size():
            if not torch.cuda.is_initialized():
                return get_module("module_custom_all_reduce").meta_size()
            return original_meta_size()

        meta_size._hcu_context_free = True
        aiter.meta_size = meta_size
        registry.register(
            "platform_detect", meta_size, target="aiter.meta_size",
            mechanism=registry.MECH_REBIND,
            detail="same installed native sizeof query, without CUDA dummy or stream",
        )
    loaded = sys.modules.get("triton.backends.driver")
    if loaded is not None:
        loaded.GPUDriver.__init__ = rewrite_function(loaded.GPUDriver.__init__)
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    for module_name, names in _RESOLUTION_TARGETS.items():
        module = sys.modules.get(module_name)
        if module is not None:
            for name in names:
                function = getattr(module, name, None)
                if function is not None:
                    rewrite_function(function, in_place=True)


def rewrite_function(function, *, in_place=False):
    """Redirect only metadata calls in an already-imported framework function."""
    import functools
    import inspect
    import textwrap
    import types

    from hcu_adapter import hip_metadata

    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    tree.body[0].decorator_list = []
    transform = _MetadataCalls()
    tree = ast.fix_missing_locations(transform.visit(tree))
    _AUDIT[function.__module__ + "." + function.__qualname__] = {
        "source": function.__code__.co_filename,
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "calls": transform.calls,
    }
    code = compile(tree, function.__code__.co_filename, 'exec')
    function_code = next(item for item in code.co_consts if isinstance(item, types.CodeType))
    function.__globals__['_hcu_metadata'] = hip_metadata
    replacement = types.FunctionType(
        function_code, function.__globals__, function.__name__, function.__defaults__,
        function.__closure__,
    )
    replacement.__kwdefaults__ = function.__kwdefaults__
    if in_place:
        if function_code.co_freevars != function.__code__.co_freevars:
            raise RuntimeError('Metadata rewrite changed function closure')
        function.__code__ = function_code
        replacement = function
    from hcu_adapter import registry

    registry.register(
        "platform_detect", replacement,
        target=function.__module__ + "." + function.__qualname__,
        mechanism=registry.MECH_TEXT,
        detail="AST: context-free HIP metadata queries in audited modules",
    )
    return replacement if in_place else functools.update_wrapper(replacement, function)


def audit():
    return dict(_AUDIT)
