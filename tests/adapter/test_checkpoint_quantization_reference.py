"""Independent CT FP8 CPU reference: strict formats, loading and executed QDQ."""

import copy
import json

import pytest

from tools.checkpoint_quantization_reference import load_reference, validate_configuration
from tools.checkpoint_files import fingerprint


def configuration():
    def arguments(strategy, dynamic):
        return dict(num_bits=8, type="float", symmetric=True, strategy=strategy,
                    dynamic=dynamic, group_size=None)
    return {
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 1,
        "quantization_config": {
            "quant_method": "compressed-tensors", "format": "float-quantized",
            "quantization_status": "compressed", "ignore": ["lm_head"],
            "config_groups": {"group_0": {
                "format": "float-quantized", "targets": ["Linear"],
                "weights": arguments("channel", False),
                "input_activations": arguments("token", True),
                "output_activations": None,
            }},
        },
    }


@pytest.mark.parametrize("fault", [
    "moe", "composite", "int8", "static", "asymmetric", "group", "output",
    "kv", "transform", "ignore", "multiple", "not-compressed", "bad-target",
])
def test_unvalidated_quantization_formats_are_refused(fault):
    config = configuration()
    quant = config["quantization_config"]
    group = quant["config_groups"]["group_0"]
    if fault == "moe":
        config["model_type"] = "qwen3_moe"
    elif fault == "composite":
        config["text_config"] = {}
    elif fault == "int8":
        group["weights"]["type"] = "int"
    elif fault == "static":
        group["input_activations"]["dynamic"] = False
    elif fault == "asymmetric":
        group["weights"]["symmetric"] = False
    elif fault == "group":
        group["weights"]["group_size"] = 128
    elif fault == "output":
        group["output_activations"] = copy.deepcopy(group["weights"])
    elif fault == "kv":
        quant["kv_cache_scheme"] = {}
    elif fault == "transform":
        quant["transform_config"] = {"rotation": True}
    elif fault == "ignore":
        quant["ignore"] = ["lm_head", "model.layers.0"]
    elif fault == "multiple":
        quant["config_groups"]["group_1"] = copy.deepcopy(group)
    elif fault == "not-compressed":
        quant["quantization_status"] = "decompressed"
    else:
        group["targets"] = ["re:.*"]
    with pytest.raises(ValueError):
        validate_configuration(config)


def test_validation_does_not_mutate_the_checkpoint_configuration():
    config = configuration()
    original = copy.deepcopy(config)
    result = validate_configuration(config)
    result["config_groups"]["group_0"]["weights"]["num_bits"] = 4
    assert config == original


def tiny_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("compressed_tensors")
    from safetensors.torch import save_file

    torch.set_num_threads(1)
    config = transformers.Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=128, tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen3ForCausalLM(config).float()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name != "lm_head":
            weight = state[name + ".weight"]
            scale = weight.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
            state[name + ".weight"] = (weight / scale).to(torch.float8_e4m3fn)
            state[name + ".weight_scale"] = scale
    document = config.to_dict()
    document["architectures"] = ["Qwen3ForCausalLM"]
    document["quantization_config"] = configuration()["quantization_config"]
    (tmp_path / "config.json").write_text(json.dumps(document))
    save_file(state, str(tmp_path / "model.safetensors"))
    return torch, transformers, document


def test_official_loader_checks_weights_and_real_activation_qdq(tmp_path, request):
    from _cpu_process import rerun_if_host_loaded
    if rerun_if_host_loaded(request):
        return
    torch, transformers, config = tiny_checkpoint(tmp_path)
    before = fingerprint(tmp_path)
    model, evidence, audit = load_reference(tmp_path, config, transformers, before)
    assert evidence["audited_tensors"] == len(model.state_dict())
    assert len(evidence["quantized_linear_modules"]) == 7
    layer = model.model.layers[0].self_attn.q_proj
    # Binary-exact scales avoid ambiguous FP32 reciprocal half-way rounding.
    x = torch.linspace(-448, 448, 32)[None, :] * torch.tensor([[1.], [2.], [0.]])
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
    qdq = (x / scale).to(torch.float8_e4m3fn).float() * scale
    expected = torch.nn.functional.linear(qdq, layer.weight)
    torch.testing.assert_close(layer(x), expected, rtol=1e-6, atol=1e-6)
    with torch.inference_mode(), audit.observe():
        result = model(input_ids=torch.tensor([[1, 2, 3, 5, 8]]), use_cache=False)
    assert torch.isfinite(result.logits).all()
    assert audit.forwards == [{"quantized_linear_calls": 7, "activation_qdq_calls": 7}]
    # Decompressed weights alone are insufficient: disabling one activation
    # wrapper must make the reference refuse to report a valid comparison.
    layer.quantization_enabled = False
    with pytest.raises(ValueError, match="missing or repeated"):
        with torch.inference_mode(), audit.observe():
            model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    assert fingerprint(tmp_path) == before


def test_bad_decompression_and_incomplete_loading_are_refused(tmp_path, request, monkeypatch):
    from _cpu_process import rerun_if_host_loaded
    if rerun_if_host_loaded(request):
        return
    torch, transformers, config = tiny_checkpoint(tmp_path)
    binding = fingerprint(tmp_path)
    model, _, _ = load_reference(tmp_path, config, transformers, binding)
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight.add_(1)
    loading = dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda *args, **kwargs: (model, loading))
    with pytest.raises(ValueError, match="does not match checkpoint bytes"):
        load_reference(tmp_path, config, transformers, binding)
    loading["missing_keys"] = ["model.layers.0.self_attn.q_proj.weight"]
    with pytest.raises(ValueError, match="complete checkpoint loading"):
        load_reference(tmp_path, config, transformers, binding)


def test_unbound_quantized_checkpoint_is_refused_before_loading():
    class NoLoading:
        pass
    with pytest.raises(ValueError, match="bound to HCU execution"):
        load_reference("/no-model", configuration(), NoLoading(), None)
