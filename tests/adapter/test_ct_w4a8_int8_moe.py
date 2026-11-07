"""Native INT8 MoE selection/load/combine contracts; CPU doubles, not serving."""

import copy
from enum import Enum
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_adapter import ct_w4a8_int8_moe as native


def entry():
    return dict(
        format="pack-quantized",
        weights=NS(num_bits=4, type="int", strategy="channel", symmetric=True,
                   dynamic=False, group_size=-1),
        input_activations=NS(num_bits=8, type="int", strategy="token", symmetric=True,
                             dynamic=True, group_size=None),
    )


def test_supported_group_preserves_input_quantization():
    value = entry()
    original = copy.deepcopy(value)
    assert native._supported_group(value)
    assert value == original
    class Type(Enum):
        INT = "int"
    value["weights"].type = Type.INT
    value["input_activations"].type = Type.INT
    assert native._supported_group(value)


def test_supported_group_accepts_official_symmetric_zero_point_dtype_default():
    from compressed_tensors.quantization import QuantizationArgs

    value = entry()
    for label in ("weights", "input_activations"):
        value[label] = QuantizationArgs(**vars(value[label]))
        assert value[label].zp_dtype == torch.int8
    original = copy.deepcopy(value)
    assert native._supported_group(value)
    assert value == original


@pytest.mark.parametrize("label,key,value", [
    ("weights", "num_bits", 8), ("weights", "type", "float"),
    ("weights", "strategy", "group"), ("weights", "group_size", 128),
    ("weights", "dynamic", True), ("weights", "symmetric", False),
    ("weights", "actorder", "group"), ("weights", "scale_dtype", "float16"),
    ("input_activations", "num_bits", 4), ("input_activations", "type", "float"),
    ("input_activations", "strategy", "tensor"),
    ("input_activations", "dynamic", False),
    ("input_activations", "symmetric", False),
    ("input_activations", "group_size", 128),
    ("input_activations", "zp_dtype", "int8"),
    ("input_activations", "observer_kwargs", {"unknown": 1}),
])
def test_other_quantization_contracts_are_rejected(label, key, value):
    group = entry()
    setattr(group[label], key, value)
    assert not native._supported_group(group)


def test_nonpacked_and_output_quantization_are_rejected():
    group = entry()
    group["format"] = "int-quantized"
    assert not native._supported_group(group)
    group["format"] = "pack-quantized"
    group["output_activations"] = group["input_activations"]
    assert not native._supported_group(group)


class Base:
    def __init__(self, config, weight_quant):
        self.config, self.weight_quant = config, weight_quant

    def create_weights(self, *args, **kwargs):
        return args, kwargs

    def process_weights_after_loading(self, layer):
        layer.converted = True


def loaded():
    return NS(
        w13_weight_packed=torch.zeros(2, 32, 8, dtype=torch.uint8),
        w2_weight_packed=torch.zeros(2, 16, 8, dtype=torch.uint8),
        w13_weight_scale=torch.ones(2, 32, 1, dtype=torch.bfloat16),
        w2_weight_scale=torch.ones(2, 16, 1, dtype=torch.bfloat16),
    )


def test_scheme_reuses_allocation_and_marks_only_after_valid_conversion():
    scheme = native._make_scheme_class(Base)("config", entry()["weights"])
    layer = loaded()
    args, attrs = scheme.create_weights(layer, 2, 16, 16, torch.bfloat16, loader="same")
    assert args == (layer, 2, 16, 16, torch.bfloat16) and attrs == {"loader": "same"}
    scheme.process_weights_after_loading(layer)
    assert layer.converted and getattr(layer, native._MARKER)
    scheme.process_weights_after_loading(layer)
    assert getattr(layer, native._MARKER)
    with pytest.raises(NotImplementedError, match="DeepEP"):
        scheme.apply_weights(layer, NS())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_scheme_refuses_other_storage_dtypes(dtype):
    scheme = native._make_scheme_class(Base)("config", entry()["weights"])
    with pytest.raises(NotImplementedError, match="BF16"):
        scheme.create_weights(loaded(), 2, 16, 8, dtype)


@pytest.mark.parametrize("hidden,intermediate", [(8, 16), (16, 8), (18, 16), (16, 18)])
def test_scheme_refuses_unsupported_packed_dimensions(hidden, intermediate):
    scheme = native._make_scheme_class(Base)("config", entry()["weights"])
    with pytest.raises(NotImplementedError, match="K/I"):
        scheme.create_weights(loaded(), 2, hidden, intermediate, torch.bfloat16)


@pytest.mark.parametrize("fault", ["shape", "scale-shape", "scale-dtype",
                                  "zero-point", "zero-scale", "nan-scale"])
def test_invalid_loaded_weights_do_not_publish_marker(fault):
    layer = loaded()
    setattr(layer, native._MARKER, True)
    if fault == "shape":
        layer.w2_weight_packed = layer.w2_weight_packed[..., :3]
    elif fault == "scale-shape":
        layer.w13_weight_scale = layer.w13_weight_scale[..., :0]
    elif fault == "scale-dtype":
        layer.w13_weight_scale = layer.w13_weight_scale.float()
    elif fault == "zero-point":
        layer.w2_weight_zero_point = torch.ones(1)
    elif fault == "zero-scale":
        layer.w2_weight_scale.zero_()
    else:
        layer.w2_weight_scale.fill_(float("nan"))
    scheme = native._make_scheme_class(Base)("config", entry()["weights"])
    with pytest.raises(ValueError):
        scheme.process_weights_after_loading(layer)
    assert not getattr(layer, native._MARKER)


class Config:
    def __init__(self, entries):
        self.entries = entries
        self.target_scheme_map = {"re:experts": entries[0]}
        self.calls = []

    def _add_fused_moe_to_target_scheme_map(self):
        pass

    def get_scheme_dict(self, layer, name):
        self.calls.append(name)
        return self.entries[len(self.calls) - 1]


def test_selector_preserves_regex_config_and_uses_matched_weight_args():
    group = entry()
    config = Config([group] * 3)
    original = copy.deepcopy(config.target_scheme_map)
    selector = native._make_get_moe_scheme(
        lambda *a, **kw: pytest.fail("stock FP8 scheme must not be selected"),
        native._make_scheme_class(Base),
    )
    scheme = selector(config, loaded(), layer_name="model.layers.0.mlp.experts")
    assert scheme.weight_quant is group["weights"]
    assert config.target_scheme_map == original
    assert len(config.calls) == 3 and "Linear" not in config.target_scheme_map


def test_selector_preserves_unrelated_and_rejects_mixed_or_unvalidated_int8():
    def original(*args, **kwargs):
        return "upstream"
    selector = native._make_get_moe_scheme(original, native._make_scheme_class(Base))
    assert selector(Config([None] * 3), loaded(), layer_name="ignored") == "upstream"
    group = entry()
    group["input_activations"].type = "float"
    assert selector(Config([group] * 3), loaded(), layer_name="fp8") == "upstream"
    with pytest.raises(ValueError, match="same quantization"):
        selector(Config([entry(), None, entry()]), loaded(), layer_name="mixed")
    group = entry()
    group["weights"].group_size = 128
    with pytest.raises(NotImplementedError, match="Unsupported"):
        selector(Config([group] * 3), loaded(), layer_name="grouped")


def dispatch():
    return NS(
        hidden_states=torch.zeros(2, 3, 16, dtype=torch.bfloat16),
        hidden_states_scale=None,
        masked_m=torch.tensor([3, 0], dtype=torch.int32),
        topk_ids=object(), topk_weights=object(), low_latency=True,
    )


def layer():
    result = loaded()
    setattr(result, native._MARKER, True)
    result.moe_runner_config = NS(activation="silu", is_gated=True, swiglu_limit=.75)
    return result


def core(kernel):
    return native._make_run_moe_core(
        lambda self, output: ("upstream", output), NS,
        lambda output: output.low_latency, kernel,
    )


def test_core_preserves_expert_layout_counts_and_combine_metadata():
    item, output, seen = layer(), dispatch(), []
    result = core(lambda *a, **kw: seen.append((a, kw)) or a[0])(item, output)
    assert result.hidden_states is output.hidden_states
    assert result.topk_ids is output.topk_ids
    assert result.topk_weights is output.topk_weights
    assert seen[0][0][-1] is output.masked_m
    assert seen[0][1] == {"swiglu_limit": .75}
    setattr(item, native._MARKER, False)
    assert core(lambda *a: pytest.fail("unmarked route"))(item, output) == (
        "upstream", output,
    )


def test_scheme_apply_uses_the_same_ll_core_and_requires_processed_weights():
    item, output = layer(), dispatch()
    execute = core(lambda value, *a, **kw: value)
    scheme = native._make_scheme_class(Base, execute)("config", entry()["weights"])
    result = scheme.apply_weights(item, output)
    assert result.hidden_states is output.hidden_states
    assert result.topk_ids is output.topk_ids and result.topk_weights is output.topk_weights
    output.low_latency = False
    with pytest.raises(NotImplementedError, match="DeepEP"):
        scheme.apply_weights(item, output)
    setattr(item, native._MARKER, False)
    with pytest.raises(RuntimeError, match="processed"):
        scheme.apply_weights(item, output)


@pytest.mark.parametrize("backend", ["triton", "marlin"])
def test_int8_scheme_does_not_inherit_w4a16_execution_metadata(backend):
    class Loader(Base):
        def get_triton_quant_info(self, layer):
            pytest.fail("W4A16 metadata would omit token-INT8 activation quantization")

        get_marlin_quant_info = get_triton_quant_info

    scheme = native._make_scheme_class(Loader)("config", entry()["weights"])
    with pytest.raises(NotImplementedError, match="standard .*?/LoRA quant metadata"):
        getattr(scheme, f"get_{backend}_quant_info")(loaded())


@pytest.mark.parametrize("field,value", [
    ("activation", "gelu"), ("is_gated", False),
    ("apply_router_weight_on_input", True),
    ("gemm1_alpha", 1.), ("gemm1_beta", 1.), ("gemm1_clamp_limit", .75),
])
def test_core_rejects_unvalidated_runner_options(field, value):
    item = layer()
    setattr(item.moe_runner_config, field, value)
    with pytest.raises(NotImplementedError):
        core(lambda *a: pytest.fail("unsupported configuration executed"))(item, dispatch())


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("factor", [None, 1., 1.5, .25])
def test_ll_core_leaves_scaling_to_topk_combine_and_model_caller(inplace, factor):
    item, output = layer(), dispatch()
    item.moe_runner_config.inplace = inplace
    item.moe_runner_config.routed_scaling_factor = factor
    before = output.hidden_states.clone()

    def kernel(value, *args, **kwargs):
        return value + 1

    result = core(kernel)(item, output)
    torch.testing.assert_close(result.hidden_states, before + 1, rtol=0, atol=0)
    torch.testing.assert_close(output.hidden_states, before, rtol=0, atol=0)
    assert result.hidden_states.data_ptr() != output.hidden_states.data_ptr()
    assert result.topk_ids is output.topk_ids
    assert result.topk_weights is output.topk_weights


@pytest.mark.parametrize("factor", [float("nan"), float("inf"), "1.5", torch.tensor(1.5)])
def test_ll_core_rejects_invalid_scaling_metadata_before_execution(factor):
    item = layer()
    item.moe_runner_config.routed_scaling_factor = factor
    with pytest.raises(NotImplementedError, match="finite routed scaling"):
        core(lambda *a: pytest.fail("invalid factor executed"))(item, dispatch())


@pytest.mark.parametrize("fault", ["normal", "scaled", "fp32", "tuple"])
def test_core_rejects_unvalidated_dispatch(fault):
    output = dispatch()
    if fault == "normal":
        output.low_latency = False
    elif fault == "scaled":
        output.hidden_states_scale = torch.ones(1)
    elif fault == "fp32":
        output.hidden_states = output.hidden_states.float()
    else:
        output.hidden_states = (output.hidden_states, torch.ones(1))
    with pytest.raises(NotImplementedError):
        core(lambda *a: pytest.fail("unsupported dispatch executed"))(layer(), output)


@pytest.mark.parametrize("fault", ["dtype", "shape"])
def test_core_rejects_invalid_combine_output(fault):
    def kernel(value, *args, **kwargs):
        return value.float() if fault == "dtype" else value[:, :1]
    with pytest.raises(RuntimeError, match="combine"):
        core(kernel)(layer(), dispatch())


def test_explicit_int8_install_does_not_wrap_config_constructor(monkeypatch):
    import sys
    from types import ModuleType
    from hcu_adapter import ct_w4a8_moe, hcu_env, registry

    class OfficialConfig:
        def __init__(self):
            self.target_scheme_map = {"re:experts": entry()}

    constructor = OfficialConfig.__init__
    module = ModuleType("sglang.srt.layers.quantization.compressed_tensors")
    module.compressed_tensors = NS(CompressedTensorsConfig=OfficialConfig)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calls = []
    monkeypatch.setattr(native, "install", lambda: calls.append("native"))
    monkeypatch.setattr(hcu_env, "resolve", lambda name: ("int8", "test"))
    monkeypatch.setattr(registry, "register", lambda *a, **kw: None)
    monkeypatch.setattr(ct_w4a8_moe, "_installed", False)
    ct_w4a8_moe.install()
    assert calls == ["native"] and OfficialConfig.__init__ is constructor
    assert OfficialConfig().target_scheme_map["re:experts"]["input_activations"] is not None


def test_failed_registration_preserves_all_bindings_and_can_retry(monkeypatch):
    import sys
    from types import ModuleType
    from hcu_adapter import registry

    def original(*args, **kwargs):
        return "original"

    class OfficialConfig:
        get_moe_scheme = original

    class OfficialMoE:
        run_moe_core = original

    def loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        return "raw"

    def adapted_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        return "transposed"

    monkeypatch.setattr(native, "_adapted_loader_code", lambda fn: adapted_loader.__code__)
    original_loader_code = loader.__code__
    class OfficialFusedMoE:
        _weight_loader_impl = loader

    modules = {
        "sglang.srt.layers.quantization.compressed_tensors": dict(
            compressed_tensors=NS(
                CompressedTensorsConfig=OfficialConfig,
                CompressedTensorsWNA16TritonMoE=Base,
            )),
        "sglang.srt.layers.moe.ep_moe.layer": dict(DeepEPMoE=OfficialMoE),
        "sglang.srt.layers.moe.fused_moe_triton.layer": dict(FusedMoE=OfficialFusedMoE),
        "sglang.srt.layers.moe.token_dispatcher": dict(
            DispatchOutputChecker=NS(format_is_deepep_ll=lambda value: True)),
        "sglang.srt.layers.moe.token_dispatcher.deepep": dict(DeepEPLLCombineInput=NS),
        "hcu_adapter.kernels.w4a8_int8_native_triton": dict(
            w4a8_int8_native_masked_experts=lambda *a, **kw: None),
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    target = "sglang.srt.layers.moe.ep_moe.layer.DeepEPMoE.run_moe_core"
    monkeypatch.setattr(registry, "_RECORDS", {})
    monkeypatch.setattr(registry, "_BOUND", {})
    registry.register("platform_detect", "foreign", target, registry.MECH_REBIND)
    before = registry.records()
    monkeypatch.setattr(native, "_installed", False)
    with pytest.raises(ValueError, match="already bound"):
        native.install()
    assert registry.records() == before and registry._BOUND == {target: "platform_detect"}
    assert OfficialConfig.get_moe_scheme is OfficialMoE.run_moe_core is original
    assert loader.__code__ is original_loader_code
    assert not native._installed
    monkeypatch.setattr(registry, "_RECORDS", {})
    monkeypatch.setattr(registry, "_BOUND", {})
    native.install()
    selector, core = OfficialConfig.get_moe_scheme, OfficialMoE.run_moe_core
    assert selector is not original and core is not original and native._installed
    assert loader.__code__ is not original_loader_code
    native.install()
    assert OfficialConfig.get_moe_scheme is selector and OfficialMoE.run_moe_core is core
