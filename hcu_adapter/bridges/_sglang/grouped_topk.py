from functools import wraps


def wrap(original, epsilon):
    @wraps(original)
    def grouped_topk(
        hidden_states, gating_output, correction_bias, topk, renormalize,
        num_expert_group=None, topk_group=None, num_fused_shared_experts=0,
        routed_scaling_factor=None, apply_routed_scaling_factor_on_output=False,
    ):
        import torch

        if (
            gating_output.ndim == 2
            and gating_output.shape[0] >= 32
            and gating_output.shape[1] == 288
            and gating_output.dtype == torch.float32
            and gating_output.is_cuda
            and gating_output.is_contiguous()
            and correction_bias.shape == (288,)
            and correction_bias.dtype == torch.float32
            and correction_bias.is_contiguous()
            and correction_bias.device == gating_output.device
            and topk == 8
            and num_expert_group == topk_group == 1
            and num_fused_shared_experts == 0
            and hidden_states.shape[0] == gating_output.shape[0]
            and (not apply_routed_scaling_factor_on_output or routed_scaling_factor is not None)
        ):
            # 替代档: 当前 original provider；形状门外保持原实现。
            from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

            return moe_fused_gate(
                gating_output, correction_bias, topk,
                scoring_func="sigmoid", renormalize=renormalize,
                routed_scaling_factor=(
                    routed_scaling_factor if routed_scaling_factor is not None else 1.0
                ),
                apply_routed_scaling_factor_on_output=bool(
                    renormalize and apply_routed_scaling_factor_on_output
                ),
                renormalize_epsilon=epsilon,
            )
        return original(
            hidden_states, gating_output, correction_bias, topk, renormalize,
            num_expert_group, topk_group, num_fused_shared_experts,
            routed_scaling_factor, apply_routed_scaling_factor_on_output,
        )

    grouped_topk._hcu_grouped_topk = True
    return grouped_topk


def install():
    import importlib
    import sys

    from hcu_adapter.infra import registry
    from hcu_adapter.ops.selector import select

    if select("sglang_grouped_topk", "torch") != "triton":
        return
    module = importlib.import_module("sglang.srt.layers.moe.topk")
    original = module.biased_grouped_topk_gpu
    if getattr(original, "_hcu_grouped_topk", False):
        return
    wrapped = wrap(original, module._RENORMALIZE_SUM_EPSILON)
    for name, loaded in list(sys.modules.items()):
        if not name.startswith("sglang.") or loaded is None:
            continue
        for symbol in ("biased_grouped_topk_gpu", "biased_grouped_topk"):
            if vars(loaded).get(symbol) is original:
                setattr(loaded, symbol, wrapped)
    registry.register(
        "sglang_grouped_topk",
        "sglang.kernels.ops.moe.moe_fused_gate.moe_fused_gate",
        "sglang.srt.layers.moe.topk.biased_grouped_topk_gpu",
        registry.MECH_REBIND,
        detail="Explicit Triton FP32 288-expert top8 single-group prefill; other calls retain current provider",
    )
