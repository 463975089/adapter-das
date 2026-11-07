"""Candidate CT channel-INT4/token-INT8 MoE route.

Reuse the community WNA16 Triton weight loader only; retain the original
activation quantization config. Execution is explicitly BF16 DeepEP LL.
Other dispatch formats and quantization contracts fail rather than dropping
activation quantization or falling into the community tensor-FP8 scheme.
"""

import dis
import inspect
import logging
import math
import types

LOG = logging.getLogger("hcu.adapter")
_MARKER = "_hcu_ct_channel_token_int8"
_installed = False
_LOADER_SCHEMES = (
    "CompressedTensorsWNA16MarlinMoE",
    "CompressedTensorsWNA16MoE",
    "CompressedTensorsWNA16TritonMoE",
)
_CANDIDATE_SCHEME = "CTChannelTokenInt8MoE"


def _adapted_loader_code(function):
    """Extend only the upstream packed-layout name test, retaining aliases."""
    if not isinstance(function, types.FunctionType) or function.__closure__:
        raise RuntimeError("CT token-INT8 expected a closure-free expert weight loader")
    parameters = inspect.signature(function).parameters
    if list(parameters) != [
        "self", "param", "loaded_weight", "weight_name", "shard_id", "expert_id",
    ] or any(
        p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
        or p.default is not inspect.Parameter.empty for p in parameters.values()
    ):
        raise RuntimeError("CT token-INT8 expert weight loader signature changed")
    code = function.__code__
    accepted = (_LOADER_SCHEMES, _LOADER_SCHEMES + (_CANDIDATE_SCHEME,))
    instructions = list(dis.get_instructions(code))
    sites = [i for i, op in enumerate(instructions)
             if op.opname == "LOAD_CONST" and op.argval in accepted]
    if len(sites) != 1:
        raise RuntimeError("CT token-INT8 packed-layout scheme test changed")
    i = sites[0]
    if (i < 3 or instructions[i - 3].opname != "LOAD_FAST"
            or instructions[i - 3].argval != "method"
            or instructions[i - 2].opname != "LOAD_ATTR"
            or instructions[i - 2].argval != "__class__"
            or instructions[i - 1].opname != "LOAD_ATTR"
            or instructions[i - 1].argval != "__name__"
            or instructions[i + 1].opname != "CONTAINS_OP"
            or instructions[i + 1].arg != 0):
        raise RuntimeError("CT token-INT8 packed-layout test is no longer a class-name check")
    if instructions[i].argval == accepted[1]:
        return code
    return code.replace(co_consts=tuple(
        _LOADER_SCHEMES + (_CANDIDATE_SCHEME,) if value == _LOADER_SCHEMES else value
        for value in code.co_consts
    ))


def _name(value):
    return str(getattr(value, "value", value)).lower()


def _supported_group(entry):
    import torch

    if (not isinstance(entry, dict) or entry.get("format") != "pack-quantized"
            or entry.get("output_activations") is not None):
        return False
    for label, bits, strategy, dynamic, group_size in (
        ("weights", 4, "channel", False, -1),
        ("input_activations", 8, "token", True, None),
    ):
        args = entry.get(label)
        if (args is None or getattr(args, "num_bits", None) != bits
                or _name(getattr(args, "type", None)) != "int"
                or _name(getattr(args, "strategy", None)) != strategy
                or getattr(args, "symmetric", None) is not True
                or getattr(args, "dynamic", None) is not dynamic
                or getattr(args, "group_size", None) != group_size
                or getattr(args, "observer_kwargs", None) not in (None, {})
                or any(getattr(args, key, None) is not None for key in (
                    "actorder", "block_structure", "scale_dtype",
                ))
                # CT 0.17 fills an unspecified zero-point dtype with int8,
                # including symmetric groups that have no zero-point tensor.
                or getattr(args, "zp_dtype", None) not in (None, torch.int8)):
            return False
    return True


def _make_scheme_class(base, run_core=None):
    """Keep official allocation/load conversion; add a distinct execution tag."""
    class CTChannelTokenInt8MoE(base):
        def create_weights(
            self, layer, num_experts, hidden_size,
            intermediate_size_per_partition, params_dtype, **extra_weight_attrs,
        ):
            import torch

            if params_dtype != torch.bfloat16:
                raise NotImplementedError("CT token-INT8 serving requires BF16 storage")
            if (hidden_size < 16 or intermediate_size_per_partition < 16
                    or hidden_size % 8 or intermediate_size_per_partition % 8):
                raise NotImplementedError(
                    "Community raw INT4 loader requires K/I >= 16 and divisible by 8"
                )
            return super().create_weights(
                layer, num_experts, hidden_size, intermediate_size_per_partition,
                params_dtype, **extra_weight_attrs,
            )

        def process_weights_after_loading(self, layer):
            import torch

            # Publish the marker only after conversion and all guards succeed.
            setattr(layer, _MARKER, False)
            super().process_weights_after_loading(layer)
            w1, w2 = layer.w13_weight_packed, layer.w2_weight_packed
            s1, s2 = layer.w13_weight_scale, layer.w2_weight_scale
            if (w1.dtype != torch.uint8 or w2.dtype != torch.uint8
                    or w1.ndim != 3 or w2.ndim != 3
                    or w1.shape[0] < 1 or w1.shape[1] < 2 or w1.shape[1] % 2
                    or w1.shape[2] < 1 or w2.shape[2] < 1
                    or w1.shape[2] * 2 != w2.shape[1]
                    or w2.shape[2] * 4 != w1.shape[1]
                    or w2.shape[0] != w1.shape[0]
                    or s1.shape != w1.shape[:2] + (1,)
                    or s2.shape != w2.shape[:2] + (1,)
                    or s1.dtype != torch.bfloat16 or s2.dtype != torch.bfloat16
                    or any(t.device != w1.device for t in (w2, s1, s2))
                    or getattr(layer, "w13_weight_zero_point", None) is not None
                    or getattr(layer, "w2_weight_zero_point", None) is not None):
                raise ValueError("CT token-INT8 requires converted raw channel-INT4 weights")
            for scale in (s1, s2):
                valid = torch.all(torch.isfinite(scale) & (scale > 0))
                if scale.is_cuda:
                    torch._assert_async(valid, "CT token-INT8 channel scales must be positive")
                elif not bool(valid):
                    raise ValueError("CT token-INT8 channel scales must be positive")
            setattr(layer, _MARKER, True)

        def apply_weights(self, layer, dispatch_output):
            if run_core is None:
                raise NotImplementedError("CT token-INT8 serving requires DeepEP low_latency")
            if not getattr(layer, _MARKER, False):
                raise RuntimeError("CT token-INT8 weights must be processed before execution")
            return run_core(layer, dispatch_output)

        def get_triton_quant_info(self, layer):
            # The inherited loader's metadata describes W4A16 execution and
            # would silently omit the checkpoint's token-INT8 activation QDQ.
            raise NotImplementedError(
                "CT token-INT8 has no standard Triton/LoRA quant metadata; "
                "use the BF16 DeepEP low_latency execution path"
            )

        def get_marlin_quant_info(self, layer):
            raise NotImplementedError(
                "CT token-INT8 has no standard Marlin/LoRA quant metadata; "
                "use the BF16 DeepEP low_latency execution path"
            )

    return CTChannelTokenInt8MoE


def _make_get_moe_scheme(original, scheme_class):
    def get_moe_scheme(self, layer, layer_name=None):
        if layer_name is None:
            return original(self, layer, layer_name=layer_name)
        self._add_fused_moe_to_target_scheme_map()
        entries = [
            self.get_scheme_dict(layer, layer_name + suffix)
            for suffix in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
        ]
        if not all(entry == entries[0] for entry in entries):
            raise ValueError("All MoE projections need the same quantization scheme")
        entry = entries[0]
        if _supported_group(entry):
            return scheme_class(self, weight_quant=entry["weights"])
        # Keep unrelated schemes intact, but never silently treat a requested
        # INT4/INT8 group as FP8 or W4A16 when its details are unsupported.
        if entry is not None:
            weight, activation = entry.get("weights"), entry.get("input_activations")
            if (getattr(weight, "num_bits", None) == 4
                    and _name(getattr(weight, "type", None)) == "int"
                    and _name(getattr(activation, "type", None)) == "int"):
                raise NotImplementedError("Unsupported CT INT4/token-INT8 contract")
        return original(self, layer, layer_name=layer_name)

    return get_moe_scheme


def _make_run_moe_core(original, combine_class, is_low_latency, kernel):
    def run_moe_core(self, dispatch_output):
        if not getattr(self, _MARKER, False):
            return original(self, dispatch_output)
        if not is_low_latency(dispatch_output):
            raise NotImplementedError("CT token-INT8 serving requires DeepEP low_latency")
        import torch

        if (not isinstance(dispatch_output.hidden_states, torch.Tensor)
                or dispatch_output.hidden_states.dtype != torch.bfloat16
                or getattr(dispatch_output, "hidden_states_scale", None) is not None):
            raise NotImplementedError("CT token-INT8 requires unquantized BF16 dispatch")
        config = self.moe_runner_config
        factor = getattr(config, "routed_scaling_factor", None)
        if factor is not None and (
            not isinstance(factor, (int, float)) or not math.isfinite(factor)
        ):
            raise NotImplementedError("CT token-INT8 requires a finite routed scaling factor")
        if (getattr(config, "activation", None) != "silu"
                or not getattr(config, "is_gated", False)
                or getattr(config, "apply_router_weight_on_input", False)
                or any(getattr(config, key, None) is not None for key in (
                    "gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit",
                ))):
            raise NotImplementedError("CT token-INT8 requires plain SiLU gating")
        # Upstream's inplace flag is a suggestion, not an aliasing contract.
        # LL experts produce raw expert rows: combine consumes the original
        # TopK weights, and the model caller owns any unfused routed factor.
        # Scaling this buffer would apply the factor twice in DeepSeek.
        output = kernel(
            dispatch_output.hidden_states,
            self.w13_weight_packed, self.w2_weight_packed,
            self.w13_weight_scale, self.w2_weight_scale,
            dispatch_output.masked_m,
            swiglu_limit=getattr(config, "swiglu_limit", None),
        )
        if (not isinstance(output, torch.Tensor) or output.dtype != torch.bfloat16
                or output.device != dispatch_output.hidden_states.device
                or output.shape != dispatch_output.hidden_states.shape):
            raise RuntimeError("CT token-INT8 returned an invalid combine tensor")
        return combine_class(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    return run_moe_core


def install():
    global _installed
    if _installed:
        return
    from sglang.srt.layers.quantization.compressed_tensors import compressed_tensors as ct
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput
    from .kernels.w4a8_int8_native_triton import w4a8_int8_native_masked_experts
    from . import registry

    loader = FusedMoE._weight_loader_impl
    loader_code = _adapted_loader_code(loader)
    core = _make_run_moe_core(
        DeepEPMoE.run_moe_core, DeepEPLLCombineInput,
        DispatchOutputChecker.format_is_deepep_ll, w4a8_int8_native_masked_experts,
    )
    scheme = _make_scheme_class(ct.CompressedTensorsWNA16TritonMoE, core)
    selector = _make_get_moe_scheme(ct.CompressedTensorsConfig.get_moe_scheme, scheme)
    targets = (
        "sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE._weight_loader_impl",
        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors."
        "CompressedTensorsConfig.get_moe_scheme",
        "sglang.srt.layers.moe.ep_moe.layer.DeepEPMoE.run_moe_core",
    )
    with registry.registration_transaction():
        for target in targets:
            registry.register(
                "ct_channel_w4a8_components", __name__, target,
                registry.MECH_REBIND,
                detail="Explicit CT channel INT4/token INT8 BF16 DeepEP LL candidate",
            )
    ct.CompressedTensorsConfig.get_moe_scheme = selector
    DeepEPMoE.run_moe_core = core
    loader.__code__ = loader_code
    _installed = True
    LOG.warning("HCU adapter: explicit CT token-INT8 DeepEP LL candidate installed")
