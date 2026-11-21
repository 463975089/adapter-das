# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Explicit vLLM/plugin paged-MQA seam; no global DeepGEMM rebinding.

The caller selects this provider only for HCU and writes row-major page64
FP8 keys with FP32 page-tail scales. Existing DeepInfer MX and SGLang
preshuffled-cache routes are not repurposed or modified by registration.
"""

from hcu_adapter import hcu_config, registry
from hcu_adapter.kill_switch import op_disabled

OP = "vllm_paged_mqa_logits"


def get_paged_mqa_logits_metadata(context_lens, block_size, num_sms, indices=None):
    from hcu_adapter.kernels import mla_attention_emu

    if block_size != 64:
        raise NotImplementedError("HCU paged MQA supports page64 only")
    return mla_attention_emu.get_paged_mqa_logits_metadata(
        context_lens, block_size, num_sms, indices=indices
    )


def fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables,
                             schedule_metadata, max_model_len, indices=None):
    from hcu_adapter.kernels import mla_attention_emu

    if q[1] is not None:
        raise NotImplementedError("This HCU vLLM paged MQA seam requires non-MX FP8 inputs")
    return mla_attention_emu.fp8_fp4_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_tables, schedule_metadata,
        max_model_len, indices=indices,
    )



def get_triton_paged_mqa_metadata(context_lens, block_size, num_sms, indices=None):
    if block_size != 64:
        raise NotImplementedError("HCU paged MQA supports page64 only")
    # The official Triton helper uses device lengths directly; no scheduling
    # allocation or fabricated native scheduler is needed.
    return None


def triton_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables,
                            schedule_metadata, max_model_len, indices=None):
    from hcu_adapter.kernels.vllm_paged_mqa_triton import fp32_paged_mqa_logits

    return fp32_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_tables, schedule_metadata,
        max_model_len, indices=indices,
    )


def get_paged_mqa_provider():
    """Return paired scheduler/scorer at model construction (also in workers).

    No binding/cache survives registration: repeated or spawn/fork imports
    select the same explicit functions. Kill-switch policy is startup-only;
    disabling refuses this route rather than falling into a CUDA kernel.
    """
    if op_disabled(OP):
        raise NotImplementedError(f"HCU {OP} disabled via HCU_DISABLE_OPS; restart after changing policy")
    # Use the existing public config, not a new model-private environment
    # variable. Keep Torch as default until model/Graph qualification completes.
    backend = hcu_config.load()["ops"].get(OP, "emu")
    if backend == "triton":
        metadata, scorer = get_triton_paged_mqa_metadata, triton_paged_mqa_logits
    elif backend == "emu":
        metadata, scorer = get_paged_mqa_logits_metadata, fp8_fp4_paged_mqa_logits
    else:
        raise ValueError(f"Unsupported {OP} backend: {backend!r}")
    registry.register(
        op=OP, impl=backend,
        target=__name__ + "." + scorer.__name__,
        mechanism=registry.MECH_SEAM, source=__name__,
    )
    return metadata, scorer
