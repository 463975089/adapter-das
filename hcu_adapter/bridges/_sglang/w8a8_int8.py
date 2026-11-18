from functools import wraps
from types import FunctionType


def install():
    import torch

    from hcu_adapter import hcu_env, registry

    if torch.version.hip is None:
        return
    if hcu_env.resolve("HCU_PLATFORM_SEMANTICS")[0] != "cuda":
        return

    from sglang.srt.configs.model_config import ModelConfig

    original = ModelConfig._verify_quantization
    if getattr(original, "_hcu_w8a8_int8", False):
        return

    @wraps(original)
    def verify(model_config):
        requested = model_config.quantization
        if requested is None:
            config = model_config._parse_quant_hf_config() or {}
            requested = config.get("quant_method")
        if requested != "w8a8_int8":
            return original(model_config)
        local_globals = dict(original.__globals__, is_hip=lambda: False)
        adapted = FunctionType(
            original.__code__, local_globals, original.__name__,
            original.__defaults__, original.__closure__,
        )
        adapted.__kwdefaults__ = original.__kwdefaults__
        return adapted(model_config)

    verify._hcu_w8a8_int8 = True
    ModelConfig._verify_quantization = verify
    registry.register(
        "platform_detect", verify,
        target="sglang.srt.configs.model_config:ModelConfig._verify_quantization",
        mechanism=registry.MECH_REBIND,
        detail="w8a8_int8 uses HCU kernels with upstream format validation",
    )
