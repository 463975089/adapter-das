def top_k_renorm_probs(probs, top_k):
    # 替代档: sgl_kernel.sampling.top_k_renorm_prob；此桥复用上游 Triton。
    from sglang.kernels.ops.sampling.renorm_triton import top_k_renorm_probs_triton

    return top_k_renorm_probs_triton(probs, top_k)


def top_p_renorm_probs(probs, top_p):
    # 替代档: sgl_kernel.sampling.top_p_renorm_prob；此桥复用上游 Triton。
    from sglang.kernels.ops.sampling.renorm_triton import top_p_renorm_probs_triton

    return top_p_renorm_probs_triton(probs, top_p)


def install():
    install_grammar_bitmask()

    import importlib
    import sys

    import flashinfer.sampling as sampling

    from hcu_adapter.infra import registry

    for name, implementation in (
        ("top_k_renorm_probs", top_k_renorm_probs),
        ("top_p_renorm_probs", top_p_renorm_probs),
    ):
        setattr(sampling, name, implementation)
        registry.register(
            "sampling", implementation, f"flashinfer.sampling.{name}",
            registry.MECH_REBIND,
            detail="SGLang native HIP Triton renormalization through CUDA import face",
        )
    from hcu_adapter.kernels.sglang_tree_sampling_triton import (
        tree_speculative_sampling_target_only,
    )

    module = importlib.import_module("sglang.kernels.ops.speculative.sampling")
    original = module.tree_speculative_sampling_target_only
    for name, loaded in list(sys.modules.items()):
        if name.startswith("sglang.") and loaded is not None:
            if vars(loaded).get("tree_speculative_sampling_target_only") is original:
                loaded.tree_speculative_sampling_target_only = tree_speculative_sampling_target_only
    registry.register(
        "sglang_tree_sampling", tree_speculative_sampling_target_only,
        "sglang.kernels.ops.speculative.sampling.tree_speculative_sampling_target_only",
        registry.MECH_REBIND,
        detail="Target-only tree traversal and residual sampling without FlashInfer headers",
    )


def install_grammar_bitmask():
    import functools

    import torch

    if not getattr(torch.version, "hip", None):
        return

    # JIT 唯一实现：只约束原 kernel 的 launch 参数，保留计算体与别名。
    from sglang.kernels.ops.grammar.bitmask_ops import apply_token_bitmask_inplace_kernel

    from hcu_adapter.infra import registry

    kernel = apply_token_bitmask_inplace_kernel
    original = kernel.run
    if getattr(original, "_hcu_grammar_launch", False):
        return

    @functools.wraps(original)
    def run(*args, **kwargs):
        if args and args[0].device.type == "cuda":
            properties = torch.cuda.get_device_properties(args[0].device)
            warp_size = getattr(properties, "warp_size", 64)
            kwargs["num_warps"] = min(kwargs.get("num_warps", 4), 1024 // warp_size)
        return original(*args, **kwargs)

    run._hcu_grammar_launch = True
    registry.register(
        "sampling", run,
        "sglang.kernels.ops.grammar.bitmask_ops.apply_token_bitmask_inplace_kernel.run",
        registry.MECH_REBIND,
        detail="Bound grammar Triton launch to 1024 threads using the hardware warp size",
    )
    kernel.run = run
