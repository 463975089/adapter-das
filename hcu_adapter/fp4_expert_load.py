"""Community FP4 expert loading through runtime interfaces, without writes.

The DTK DeepEP grouped-FP8 path accepts per-channel scales, while the DSV4
cast produces 128x128 block scales. Explicit SGLANG_HCU_EP_BF16=1 retains
the BF16 fallback for that mismatch; default loading stays on the original
FP8 route. Both routes keep block metadata consistent without changing a
configuration shared with unrelated checkpoint layers.
"""

import copy
import dis
import inspect
import logging
import threading
import types

LOG = logging.getLogger("hcu.adapter")
_LOCK = threading.RLock()
_TARGET = "_hcu_fp4_expert_load_impl"
_MARKER = "_hcu_fp4_expert_load_original"
_CAST = "cast_e2m1fn_to_e4m3fn"


def _align_blocks(method):
    config = method.quant_config
    if config.weight_block_size != [128, 128]:
        private = copy.copy(config)
        if private is config:
            raise RuntimeError("fp4_expert_load: quant config cannot be isolated")
        private.weight_block_size = [128, 128]
        method.quant_config = private
    method.weight_block_size = [128, 128]


def _block_to_bf16(weight, scale):
    import torch

    if weight.ndim != 3 or scale.ndim != 3:
        raise RuntimeError("fp4_expert_load: expected [E,N,K] block-FP8 weights")
    experts, rows, columns = weight.shape
    if (rows % 128 or columns % 128
            or tuple(scale.shape) != (experts, rows // 128, columns // 128)):
        raise RuntimeError("fp4_expert_load: FP8 scale grid is not 128x128")
    if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        raise RuntimeError("fp4_expert_load: community FP4 cast did not produce E4M3")
    data = weight.float().reshape(experts, rows // 128, 128, columns // 128, 128)
    scales = scale.float().reshape(experts, rows // 128, 1, columns // 128, 1)
    return (data * scales).reshape(experts, rows, columns).to(torch.bfloat16)


def _bf16_experts(method, layer):
    import torch

    method._dequantize_aiter_fp4_experts(layer)
    # Compute both before replacing either parameter. Failure must not leave
    # the first weight BF16 while the second still has FP8 scale metadata.
    w13 = _block_to_bf16(layer.w13_weight.data, layer.w13_weight_scale_inv.data)
    w2 = _block_to_bf16(layer.w2_weight.data, layer.w2_weight_scale_inv.data)
    _align_blocks(method)
    layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
    layer.w13_weight_scale_inv = None
    layer.w2_weight_scale_inv = None


def _clone(function, namespace):
    result = types.FunctionType(
        function.__code__, namespace, function.__name__,
        function.__defaults__, function.__closure__,
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def _loader(original):
    def load(self, layer):
        from . import hcu_env

        method = self
        if (hcu_env.resolve("SGLANG_HCU_EP_BF16")[0] == "1"
                and method.is_fp4_expert
                and layer.w13_weight.dtype != original.__globals__["torch"].bfloat16):
            _bf16_experts(method, layer)
            return
        if not (method.is_fp4_expert and method.dequant_fp4_to_fp8):
            return original(method, layer)

        cast = original.__globals__[_CAST]

        def cast_with_blocks(*args, **kwargs):
            weight, scale = cast(*args, **kwargs)
            if (weight.ndim != 2 or scale.ndim != 2
                    or weight.shape[0] % 128 or weight.shape[1] % 128
                    or tuple(scale.shape) != (weight.shape[0] // 128, weight.shape[1] // 128)):
                raise RuntimeError("fp4_expert_load: community cast block recipe changed")
            # Align before the original loader's remaining requantization.
            # Isolate the config from other methods using the checkpoint.
            _align_blocks(method)
            return weight, scale

        namespace = dict(original.__globals__, **{_CAST: cast_with_blocks})
        return _clone(original, namespace)(method, layer)

    return load


def _dispatch(*args, **kwargs):
    return _hcu_fp4_expert_load_impl(*args, **kwargs)


def _install_method(cls):
    with _LOCK:
        function = cls.process_weights_after_loading_block_quant
        if not isinstance(function, types.FunctionType) or function.__closure__:
            raise RuntimeError("fp4_expert_load: expected closure-free community loader")
        if hasattr(function, _MARKER):
            original, implementation = getattr(function, _MARKER)
            if (function.__code__ is not _dispatch.__code__
                    or function.__globals__.get(_TARGET) is not implementation):
                raise RuntimeError("fp4_expert_load: loader changed after installation")
            return function
        signature = inspect.signature(function)
        if (list(signature.parameters) != ["self", "layer"]
                or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
                       or p.default is not inspect.Parameter.empty
                       for p in signature.parameters.values())):
            raise RuntimeError("fp4_expert_load: upstream loader signature changed")
        casts = [i for i in dis.get_instructions(function)
                 if i.opname == "LOAD_GLOBAL" and i.argval == _CAST]
        if (len(casts) != 1 or not callable(function.__globals__.get(_CAST))
                or not callable(getattr(cls, "_dequantize_aiter_fp4_experts", None))
                or not {"is_fp4_expert", "dequant_fp4_to_fp8"}.issubset(function.__code__.co_names)):
            raise RuntimeError("fp4_expert_load: upstream FP4 loading contract changed")
        if _TARGET in function.__globals__:
            raise RuntimeError("fp4_expert_load: foreign private dispatch binding")
        original = _clone(function, dict(function.__globals__))
        implementation = _loader(original)
        function.__globals__[_TARGET] = implementation
        function.__signature__ = signature
        setattr(function, _MARKER, (original, implementation))
        function.__code__ = _dispatch.__code__
        return function


def install():
    import torch

    if torch.version.hip is None:
        return
    import sglang.srt.layers.quantization.fp8 as fp8
    from . import registry

    _install_method(fp8.Fp8MoEMethod)
    registry.register(
        "platform_detect", "fp4-expert-runtime-load",
        target="sglang.srt.layers.quantization.fp8.Fp8MoEMethod.process_weights_after_loading_block_quant",
        mechanism=registry.MECH_REBIND,
        detail="private cast and per-method block config; explicit DeepEP BF16; no source writes",
    )
    LOG.warning("HCU adapter: FP4 expert runtime loading installed (SGLANG_HCU_EP_BF16=1 selects BF16)")
