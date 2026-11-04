# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Explicit CPU QDQ reference for the validated CT channel-FP8 Qwen3 format.

Use the independent Transformers/compressed-tensors implementation, checking
every loaded tensor and observing actual activation-QDQ calls. This module
never changes checkpoint files, host kernels, or probability thresholds.
"""

from contextlib import contextmanager
import copy
from importlib.metadata import version
from pathlib import Path
import sys


REFERENCE_KIND = "ct-fp8-token-channel"


def validate_configuration(config):
    if (not isinstance(config, dict) or config.get("model_type") != "qwen3"
            or config.get("architectures") != ["Qwen3ForCausalLM"]
            or "text_config" in config
            or type(config.get("num_hidden_layers")) is not int
            or config["num_hidden_layers"] < 1):
        raise ValueError("CT FP8 reference currently requires a Dense Qwen3 checkpoint")
    quant = config.get("quantization_config", {})
    groups = quant.get("config_groups", {}) if isinstance(quant, dict) else {}
    if (not isinstance(quant, dict) or quant.get("quant_method") != "compressed-tensors"
            or quant.get("format") != "float-quantized"
            or quant.get("quantization_status") != "compressed"
            or quant.get("ignore") != ["lm_head"]
            or quant.get("kv_cache_scheme") is not None
            or quant.get("transform_config") not in (None, {})
            or quant.get("sparsity_config") not in (None, {})
            or not isinstance(groups, dict) or len(groups) != 1):
        raise ValueError("requires the validated compressed-tensors FP8 configuration")
    group = next(iter(groups.values()))
    if (not isinstance(group, dict) or group.get("targets") != ["Linear"]
            or group.get("format") != "float-quantized"
            or group.get("output_activations") is not None):
        raise ValueError("requires one FP8 Linear group without output quantization")
    for label, strategy, dynamic in (
        ("weights", "channel", False), ("input_activations", "token", True),
    ):
        args = group.get(label)
        if (not isinstance(args, dict) or args.get("num_bits") != 8
                or args.get("type") != "float" or args.get("symmetric") is not True
                or args.get("strategy") != strategy or args.get("dynamic") is not dynamic
                or any(args.get(key) is not None for key in (
                    "actorder", "block_structure", "group_size", "scale_dtype", "zp_dtype",
                ))
                or args.get("observer_kwargs") not in (None, {})):
            raise ValueError(f"unsupported CT FP8 {label} quantization contract")
    return copy.deepcopy(quant)


class ActivationQDQAudit:
    def __init__(self, modules):
        self.modules = modules
        self.forwards = []

    @contextmanager
    def observe(self):
        """Observe real library calls without replacing their implementation."""
        from compressed_tensors.quantization.lifecycle.forward import (
            fake_quantize, forward_quantize,
        )
        import inspect
        import torch

        if sys.getprofile() is not None:
            raise ValueError("activation QDQ audit requires an unused Python profiler")
        identities = {id(module): name for name, module in self.modules.items()}
        calls, fake_calls = {}, {}
        fake_code = inspect.unwrap(fake_quantize).__code__

        def record(frame, event, result):
            if event != "call":
                return
            if frame.f_code is forward_quantize.__code__:
                name = identities.get(id(frame.f_locals.get("module")))
                if name is None or frame.f_locals.get("base_name") != "input":
                    raise ValueError("unexpected quantization operation in CPU reference")
                value = frame.f_locals["value"]
                if value.device.type != "cpu" or value.dtype != torch.float32:
                    raise ValueError("activation QDQ must receive CPU FP32 activations")
                calls[name] = calls.get(name, 0) + 1
            elif frame.f_code is fake_code:
                parent = frame.f_back
                # fake_quantize is wrapped by torch.no_grad; its immediate
                # caller is the shared contextlib wrapper, not forward_quantize.
                while parent is not None and parent.f_code is not forward_quantize.__code__:
                    parent = parent.f_back
                name = identities.get(id(parent.f_locals.get("module"))) if parent else None
                if parent is None or name is None:
                    raise ValueError("unexpected fake-quantization call in CPU reference")
                fake_calls[name] = fake_calls.get(name, 0) + 1

        try:
            sys.setprofile(record)
            yield
        finally:
            sys.setprofile(None)
        expected = {name: 1 for name in self.modules}
        if calls != expected or fake_calls != expected:
            raise ValueError("missing or repeated per-layer activation QDQ execution")
        self.forwards.append({
            "quantized_linear_calls": len(calls), "activation_qdq_calls": len(fake_calls),
        })


def load_reference(model_path, config, transformers, binding):
    """Return a fully audited CPU FP32 model and an execution observer."""
    quant = validate_configuration(config)
    if not binding:
        raise ValueError("quantized reference requires checkpoint bytes bound to HCU execution")
    import torch
    from safetensors import safe_open
    from compressed_tensors.quantization import QuantizationStatus

    quant["run_compressed"] = False
    model, loading = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.float32, attn_implementation="eager",
        local_files_only=True, trust_remote_code=False,
        quantization_config=transformers.CompressedTensorsConfig.from_dict(quant),
        output_loading_info=True,
    )
    if (not isinstance(loading, dict) or any(
        key not in loading or loading[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    )):
        raise ValueError("CPU quantized reference requires complete checkpoint loading")
    model.eval()
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("quantized reference weights must be CPU FP32")
    modules = {name: module for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and name != "lm_head"}
    expected_names = {
        f"model.layers.{index}.{part}"
        for index in range(config["num_hidden_layers"])
        for part in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    }
    if set(modules) != expected_names:
        raise ValueError("unexpected Qwen3 quantized Linear coverage")
    for module in modules.values():
        scheme = getattr(module, "quantization_scheme", None)
        if (scheme is None or module.quantization_status != QuantizationStatus.DECOMPRESSED
                or getattr(module, "quantization_enabled", True) is not True
                or scheme.input_activations is None
                or scheme.weights is None or scheme.output_activations is not None
                or any(
                    args.num_bits != 8 or args.type != "float"
                    or args.symmetric is not True or args.strategy != strategy
                    or args.dynamic is not dynamic
                    for args, strategy, dynamic in (
                        (scheme.input_activations, "token", True),
                        (scheme.weights, "channel", False),
                    )
                )):
            raise ValueError("loaded reference must retain dynamic activation quantization")

    # Loading-info alone can miss incorrect decompression or inconsistent tied
    # tensors. Check all raw parameters, including every ignored/unquantized one.
    state = model.state_dict()
    raw = {}
    for relative in sorted(name for name in binding if name.endswith(".safetensors")):
        with safe_open(str(Path(model_path) / relative), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in raw:
                    raise ValueError(f"duplicate checkpoint tensor: {name}")
                raw[name] = reader.get_tensor(name)
    if set(raw) != set(state):
        raise ValueError("raw and loaded quantized checkpoint tensor inventories disagree")
    for name, tensor in raw.items():
        prefix = name.removesuffix(".weight")
        if name.endswith(".weight") and prefix in modules:
            scale = raw.get(prefix + ".weight_scale")
            if (tensor.dtype != torch.float8_e4m3fn or tensor.ndim != 2
                    or scale is None or scale.shape != (tensor.shape[0], 1)
                    or scale.dtype not in (torch.bfloat16, torch.float32)
                    or not torch.isfinite(scale).all() or not (scale > 0).all()):
                raise ValueError(f"invalid FP8 channel weight/scale: {name}")
            expected = tensor.float() * scale.float()
        else:
            if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f"unexpected non-Linear tensor dtype: {name}")
            expected = tensor.float()
        if not torch.isfinite(expected).all() or not torch.equal(state[name], expected):
            raise ValueError(f"reference tensor does not match checkpoint bytes: {name}")
    evidence = {
        "kind": REFERENCE_KIND, "compressed_tensors": version("compressed-tensors"),
        "run_compressed": False, "loading_info_clean": True,
        "audited_tensors": len(raw), "quantized_linear_modules": sorted(modules),
        "weights": "exact FP8 channel dequantization checked against every raw tensor",
        "activations": "official dynamic per-token FP8 QDQ observed at every Linear",
    }
    return model, evidence, ActivationQDQAudit(modules)
