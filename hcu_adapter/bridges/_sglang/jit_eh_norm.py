import importlib
from functools import wraps


def install():
    from hcu_adapter import registry

    try:
        module = importlib.import_module("sglang.kernels.ops.layernorm.fused_eh_norm")
    except ModuleNotFoundError as error:
        if error.name == "sglang.kernels.ops.layernorm.fused_eh_norm":
            return
        raise
    original = module.load_jit
    if getattr(original, "_hcu_eh_norm_launch_bounds", False):
        return

    @wraps(original)
    def load_jit(*args, **kwargs):
        flags = list(kwargs.get("extra_cuda_cflags") or ())
        # The upstream 8192-wide norm launches 1024 threads. DTK otherwise
        # compiles kernels without launch_bounds for at most 256 threads.
        flags.append("--gpu-max-threads-per-block=1024")
        kwargs["extra_cuda_cflags"] = flags
        return original(*args, **kwargs)

    load_jit._hcu_eh_norm_launch_bounds = True
    module.load_jit = load_jit
    registry.register(
        "rms_norm", load_jit,
        target="sglang.kernels.ops.layernorm.fused_eh_norm:load_jit",
        mechanism=registry.MECH_REBIND,
        detail="SGLang fused EH norm HIP build covers the upstream CTA thread count",
    )
