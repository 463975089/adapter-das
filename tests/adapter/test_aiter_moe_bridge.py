from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from hcu_adapter import aiter_moe


@pytest.mark.parametrize(
    "swiglu_limit,gemm1_clamp_limit,expected",
    [(10.0, None, 10.0), (10.0, 7.0, 10.0), (None, 7.0, 7.0),
     (None, None, None), (0.0, 7.0, 0.0)],
)
def test_model_swiglu_limit_reaches_vendor(
    monkeypatch, swiglu_limit, gemm1_clamp_limit, expected
):
    calls = []
    vendor = ModuleType("aiter.moe")
    vendor.MoeQuantType = SimpleNamespace(FP8_W8A8="fp8", W8A8="int8")
    vendor.get_aiter_moe_config = lambda **kwargs: (
        True, SimpleNamespace(need_shuffle=False, solution_type="test")
    )

    def execute(*args, **kwargs):
        calls.append(kwargs)
        return args[0]

    vendor.aiter_moe = execute
    package = ModuleType("aiter")
    package.moe = vendor
    monkeypatch.setitem(sys.modules, "aiter", package)
    monkeypatch.setitem(sys.modules, "aiter.moe", vendor)
    monkeypatch.setenv("HCU_FUSED_MOE_IMPL", "aiter")
    config = SimpleNamespace(
        inplace=False, no_combine=False, activation="silu",
        routed_scaling_factor=1.0, gemm1_alpha=None,
        gemm1_clamp_limit=gemm1_clamp_limit, swiglu_limit=swiglu_limit,
    )
    x = torch.ones(2, 4, dtype=torch.bfloat16)
    result = aiter_moe.try_aiter_fused_moe(
        hidden_states=x,
        w1=torch.ones(2, 8, 4),
        w2=torch.ones(2, 4, 4),
        topk_output=(torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int32), None),
        moe_runner_config=config,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        w1_scale=torch.ones(2, 8, 1),
        w2_scale=torch.ones(2, 4, 1),
        a1_scale=None,
        block_shape=None,
    )
    assert result is x
    assert len(calls) == 1
    assert calls[0]["gemm1_limit"] == expected
