"""Vendor ABI for official vLLM unquantized AITER experts on validated HCU.

No operator is published by importing this module or querying config support.
Raw checkpoint weights remain raw across official loading/reload and vendor
per-shape selection. This bridge does not change the global AITER gates.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import torch

from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle

_PUBLICATION = ProcessLocalLifecycle(runtime_action="vllm-aiter-moe-publication")
_OP = None
_OP_STATE = "empty"
LOG = logging.getLogger("hcu.adapter.vllm.moe")


@lru_cache(maxsize=1)
def _compatibility_for_process(pid):
    from hcu_adapter import hcu_env

    return HcuUnquantizedAiterCompatibility(hcu_env.arch_name())


def get_compatibility():
    return _compatibility_for_process(os.getpid())


def configure_padding_policy(moe_backend):
    """Keep valid padded routes before workers spawn/cached env reads begin."""
    from vllm import envs

    from hcu_adapter import hcu_env

    requested = moe_backend == "aiter" or (
        moe_backend == "auto"
        and envs.VLLM_ROCM_USE_AITER
        and envs.VLLM_ROCM_USE_AITER_MOE
    )
    if not requested or hcu_env.arch_name() != "gfx936":
        return
    value, _source = hcu_env.resolve("VLLM_MOE_SKIP_PADDING", host="vllm")
    os.environ["VLLM_MOE_SKIP_PADDING"] = "0"
    LOG.info(
        "HCU AITER padded routes: VLLM_MOE_SKIP_PADDING=0 (was %s, %s)", value, _source
    )
    if envs.VLLM_MOE_SKIP_PADDING:
        raise RuntimeError("HCU AITER padding policy must run before env caching")


def _publish_operator() -> None:
    global _OP, _OP_STATE
    if _OP_STATE == "ready":
        return
    if _OP_STATE != "empty":
        raise RuntimeError("incomplete HCU AITER publication; use a fresh process")
    # Refusal precedes all imports/registration and survives BaseException/fork.
    _OP_STATE = "incomplete"

    @torch.library.custom_op(
        "hcu_adapter::vllm_aiter_unquantized_moe",
        mutates_args=(),
        device_types="cuda",
    )
    def execute(
        hidden: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        weights: torch.Tensor,
        ids: torch.Tensor,
        activation: int,
    ) -> torch.Tensor:
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe

        return fused_moe(
            hidden,
            w1,
            w2,
            weights,
            ids,
            activation=ActivationType(activation),
            quant_type=QuantType(0),
            dtype=hidden.dtype,
            swiglu_limit=None,
        )

    @execute.register_fake
    def fake(hidden, w1, w2, weights, ids, activation):
        return torch.empty_like(hidden)

    _OP = execute
    _OP_STATE = "ready"


class HcuUnquantizedAiterCompatibility:
    """Narrow public platform provider; initial evidence is gfx936 only."""

    supports_pre_processed_weights = False

    def __init__(self, architecture: str):
        self.architecture = architecture

    def supports_config(self, moe_config):
        from vllm import envs
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        m = moe_config
        if self.architecture != "gfx936":
            return False, "unquantized HCU AITER is only validated on gfx936"
        if m.moe_backend != "aiter" and not (
            envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MOE
        ):
            return False, "HCU AITER MoE requires explicit official opt-in"
        if envs.VLLM_MOE_SKIP_PADDING:
            return False, "HCU AITER requires startup normalization of padded routes"
        if m.in_dtype not in (torch.bfloat16, torch.float16):
            return False, "HCU AITER requires BF16/FP16 activations and weights"
        if m.activation not in (MoEActivation.SILU, MoEActivation.GELU):
            return False, "HCU AITER supports gated SiLU/GELU only"
        if m.has_bias or m.is_lora_enabled:
            return False, "HCU AITER does not support expert bias or LoRA"
        if getattr(m, "apply_router_weight_on_input", False):
            return False, "HCU AITER router-weight-on-input is unsupported"
        p = m.moe_parallel_config
        if p.use_ep or p.enable_eplb or p.dp_size != 1 or p.pcp_size != 1:
            return False, "HCU AITER EP/EPLB/DP/PCP is not validated"
        if m.aiter_fmoe_shared_expert_enabled:
            return False, "HCU AITER shared expert fusion is not validated"
        if (
            m.hidden_dim_unpadded != m.hidden_dim
            or m.intermediate_size_per_partition_unpadded
            != m.intermediate_size_per_partition
            or m.intermediate_pad not in (None, 0)
            or m.tp_shard_with_padding
        ):
            return False, "HCU AITER padded weights are not supported"
        n = m.intermediate_size_per_partition
        alignment = 64 if n <= 192 else 128
        if m.hidden_dim % 32 or n % alignment:
            return False, "HCU AITER dimensions would require weight padding"
        if m.swiglu_limit not in (None, 0.0) or any(
            value is not None
            for value in (
                m.swiglu_alpha,
                m.swiglu_beta,
                m.activation_situ_beta,
                m.activation_situ_linear_beta,
            )
        ):
            return False, "HCU AITER activation parameters are unsupported"
        return True, None

    @staticmethod
    def _validate_raw_weights(w1, w2):
        for weight in (w1, w2):
            if getattr(weight, "is_shuffled", False) or getattr(
                weight, "weight_layout", "raw"
            ) not in (None, "raw"):
                raise ValueError("HCU AITER requires canonical raw weight layout")
            if weight.dtype not in (torch.bfloat16, torch.float16):
                raise ValueError("HCU AITER requires BF16/FP16 weights")

    def convert_weights(self, moe_config, w13_weight, w2_weight):
        supported, reason = self.supports_config(moe_config)
        if not supported:
            raise ValueError(reason)
        self._validate_raw_weights(w13_weight, w2_weight)
        return w13_weight, w2_weight

    @staticmethod
    def prepare():
        # Called only when the official selected expert is constructed, also
        # for preprocessed weights. Partial publication is latched, not retried.
        _PUBLICATION.run_once("vllm-aiter-moe-publication", _publish_operator)

    def apply(
        self,
        moe_config,
        quant_config,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        _PUBLICATION.require_runtime_healthy()
        if _OP_STATE != "ready":
            raise RuntimeError("HCU AITER operator must be prepared before apply")
        supported, reason = self.supports_config(moe_config)
        if not supported:
            raise ValueError(reason)
        self._validate_raw_weights(w1, w2)
        if (
            expert_map is not None
            or expert_tokens_meta is not None
            or apply_router_weight_on_input
            or a1q_scale is not None
            or a2_scale is not None
            or global_num_experts not in (-1, w1.shape[0])
            or activation != moe_config.activation
            or quant_config.w1_scale is not None
            or quant_config.w2_scale is not None
            or quant_config.w1_bias is not None
            or quant_config.w2_bias is not None
            or quant_config.gemm1_alpha is not None
            or quant_config.gemm1_beta is not None
            or quant_config.gemm1_clamp_limit not in (None, 0.0)
        ):
            raise NotImplementedError("unsupported HCU unquantized AITER apply ABI")
        if hidden_states.dtype != w1.dtype or w2.dtype != w1.dtype:
            raise ValueError("HCU AITER input/weight dtype mismatch")
        if (
            hidden_states.ndim != 2
            or w1.ndim != 3
            or w2.ndim != 3
            or w1.shape
            != (
                moe_config.num_local_experts,
                2 * moe_config.intermediate_size_per_partition,
                moe_config.hidden_dim,
            )
            or w2.shape
            != (
                w1.shape[0],
                moe_config.hidden_dim,
                moe_config.intermediate_size_per_partition,
            )
            or hidden_states.shape[1] != moe_config.hidden_dim
            or topk_ids.shape != topk_weights.shape
            or topk_ids.shape != (hidden_states.shape[0], moe_config.experts_per_token)
            or output.shape != hidden_states.shape
            or output.dtype != hidden_states.dtype
            or not all(
                t.is_contiguous()
                for t in (hidden_states, w1, w2, topk_ids, topk_weights)
            )
            or not all(
                t.device == hidden_states.device
                for t in (w1, w2, topk_ids, topk_weights, output)
            )
        ):
            raise ValueError("unsupported HCU AITER tensor geometry/storage")
        if _OP is None:
            raise RuntimeError("HCU AITER operator must be prepared before apply")
        result = _OP(
            hidden_states,
            w1,
            w2,
            topk_weights.to(torch.float32),
            topk_ids.to(torch.int32),
            0 if activation == MoEActivation.SILU else 1,
        )
        output.copy_(result)
