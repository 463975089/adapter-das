"""Real v0.33 config/CPU tensor contracts for the HCU AITER MoE ABI.

Run in an isolated v0.33 environment, HOST=none and GPUs hidden:
python -m pytest -q tests/adapter/test_vllm_moe_aiter.py
"""

from dataclasses import replace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)

from hcu_adapter.bridges.vllm_moe_aiter import HcuUnquantizedAiterCompatibility


@pytest.fixture(autouse=True)
def padding_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "0")


def config(**kwargs):
    values = dict(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=128,
        num_local_experts=8,
        num_logical_experts=8,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        moe_backend="aiter",
    )
    values.update(kwargs)
    return FusedMoEConfig(**values)


def test_exact_hardware_and_requested_policy():
    for arch in ("gfx928", "gfx92a", "gfx938", "gfx942", "gfx936:unknown"):
        assert not HcuUnquantizedAiterCompatibility(arch).supports_config(config())[0]
    provider = HcuUnquantizedAiterCompatibility("gfx936")
    assert provider.supports_config(config()) == (True, None)
    assert provider.supports_config(config(in_dtype=torch.float16)) == (True, None)
    assert not provider.supports_config(config(moe_backend="auto"))[0]


@pytest.mark.parametrize(
    "change",
    [
        {"has_bias": True},
        {"is_lora_enabled": True},
        {"in_dtype": torch.float32},
        {"activation": MoEActivation.SWIGLUOAI},
        {"hidden_dim_unpadded": 255},
        {"intermediate_size_per_partition_unpadded": 127},
        {"intermediate_size": 129},
        {"swiglu_limit": 7.0},
        {"swiglu_alpha": 2.0},
        {"swiglu_beta": 1.0},
        {"activation_situ_beta": 1.0},
        {"intermediate_pad": 1},
        {"apply_router_weight_on_input": True},
    ],
)
def test_unsupported_semantics_refused_before_weights(change):
    supported, reason = HcuUnquantizedAiterCompatibility("gfx936").supports_config(
        config(**change)
    )
    assert not supported
    assert isinstance(reason, str) and reason


def test_ep_rejected_with_real_parallel_config():
    parallel = replace(
        FusedMoEParallelConfig.make_no_parallel(), use_ep=True, ep_size=2, dp_size=2
    )
    assert not HcuUnquantizedAiterCompatibility("gfx936").supports_config(
        config(moe_parallel_config=parallel)
    )[0]


def test_raw_layout_contract_is_lossless_and_repeatable():
    provider = HcuUnquantizedAiterCompatibility("gfx936")
    w1 = torch.arange(8 * 256 * 256).reshape(8, 256, 256).to(torch.bfloat16)
    w2 = torch.arange(8 * 256 * 128).reshape(8, 256, 128).to(torch.bfloat16)
    snapshots = [w.clone() for w in (w1, w2)]
    for _ in range(2):
        a, b = provider.convert_weights(config(), w1, w2)
        assert a is w1 and b is w2
        assert torch.equal(a, snapshots[0]) and torch.equal(b, snapshots[1])
        assert not getattr(a, "is_shuffled", False)
        assert not getattr(b, "is_shuffled", False)
    w1.is_shuffled = True
    with pytest.raises(ValueError, match="layout"):
        provider.convert_weights(config(), w1, w2)


def test_no_publication_on_import_or_config_query():
    provider = HcuUnquantizedAiterCompatibility("gfx936")
    provider.supports_config(config())
    with pytest.raises(AttributeError):
        torch.ops.hcu_adapter.vllm_aiter_unquantized_moe.default


def test_padding_default_materialized_before_cache(monkeypatch):
    from vllm import envs

    from hcu_adapter import hcu_env
    from hcu_adapter.bridges.vllm_moe_aiter import configure_padding_policy

    # Only the external hardware probe is substituted; real env/config readers.
    monkeypatch.setattr(hcu_env, "arch_name", lambda: "gfx936")
    monkeypatch.delenv("VLLM_MOE_SKIP_PADDING", raising=False)
    assert envs.VLLM_MOE_SKIP_PADDING
    configure_padding_policy("aiter")
    assert not envs.VLLM_MOE_SKIP_PADDING
