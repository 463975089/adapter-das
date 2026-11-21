# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Zero-fallback regressions (governance 3b-8/3b-9, 2026-11-19).

Runtime tier dispatch must never descend: an explicit tier that cannot
serve a call raises; auto-mode per-shape guards remain deterministic
selection (returning to the stock path is legal and auditable). The
cases here run on plain CPU tensors -- the aiter_moe guards evaluate
tensor metadata before any device or vendor work, so the strict-vs-auto
split is testable without HCU.
"""

from types import SimpleNamespace

import pytest
import torch


def _moe_args(dtype=torch.float32):
    hidden = torch.zeros(8, 64, dtype=dtype)
    w1 = torch.zeros(16, 128, 64, dtype=torch.bfloat16)
    w2 = torch.zeros(16, 64, 32, dtype=torch.bfloat16)
    topk = (
        torch.zeros(8, 4),
        torch.zeros(8, 4, dtype=torch.int32),
        torch.zeros(8, 4, dtype=torch.int32),
    )
    return dict(
        hidden_states=hidden,
        w1=w1,
        w2=w2,
        topk_output=topk,
        moe_runner_config=SimpleNamespace(no_combine=False),
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        w1_scale=None,
        w2_scale=None,
        a1_scale=None,
        block_shape=(128, 128),
        b1=None,
        b2=None,
    )


def test_aiter_moe_explicit_refusal_is_a_hard_error(monkeypatch):
    monkeypatch.setenv("HCU_FUSED_MOE_IMPL", "aiter")
    from hcu_adapter.features.aiter import aiter_moe

    with pytest.raises(RuntimeError, match="explicit tier 'aiter' refused"):
        aiter_moe.try_aiter_fused_moe(**_moe_args())


def test_aiter_moe_auto_refusal_is_selection_passthrough(monkeypatch):
    monkeypatch.setenv("HCU_FUSED_MOE_IMPL", "auto")
    from hcu_adapter.features.aiter import aiter_moe

    assert aiter_moe.try_aiter_fused_moe(**_moe_args()) is None


def test_aiter_moe_config_lookup_exception_propagates(monkeypatch):
    """3b-8: a config-lookup crash is a bug, not a selection outcome."""
    monkeypatch.setenv("HCU_FUSED_MOE_IMPL", "auto")
    from hcu_adapter.features.aiter import aiter_moe

    real = aiter_moe.get_aiter_moe_config if hasattr(
        aiter_moe, "get_aiter_moe_config") else None
    # get_aiter_moe_config is imported inside the function; patch the
    # module it comes from via a broken aiter.moe shim.
    import sys
    import types

    shim = types.ModuleType("aiter")
    moe = types.ModuleType("aiter.moe")

    class _Q:
        FP8_W8A8 = 1

    moe.MoeQuantType = _Q
    moe.aiter_moe = lambda *a, **k: None

    def boom(**kwargs):
        raise OSError("corrupt tuned csv")

    moe.get_aiter_moe_config = boom
    shim.moe = moe
    monkeypatch.setitem(sys.modules, "aiter", shim)
    monkeypatch.setitem(sys.modules, "aiter.moe", moe)
    try:
        args = _moe_args(dtype=torch.bfloat16)  # pass the dtype guard
        with pytest.raises(OSError, match="corrupt tuned csv"):
            aiter_moe.try_aiter_fused_moe(**args)
    finally:
        del real


def test_gdn_router_refuses_emu_runtime_request():
    """3b-9: HCU_KDA_PREFILL_IMPL / HCU_CONV1D_IMPL = emu raises at
    routing time; fork|stock are the runtime values."""
    from hcu_adapter.features.gdn import fla_hcu_router

    for var, fn in (
        ("HCU_KDA_PREFILL_IMPL", fla_hcu_router._install_kda_prefill),
        ("HCU_CONV1D_IMPL", fla_hcu_router._install_conv1d),
    ):
        with pytest.raises(RuntimeError, match="%s=emu is retired" % var):
            fn("emu")
