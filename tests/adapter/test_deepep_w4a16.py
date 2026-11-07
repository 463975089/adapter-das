# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DeepEP low-latency W4A16 routing contract.

These tests stay on CPU and fake only the upstream class seam.  Numeric
correctness of the masked grouped GEMM remains in the GPU kernel suite; this
file proves that CT W4A8's W4A16 fallback reaches the adapter-owned
expert-major branch without changing unrelated DeepEP routes.
"""

from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from hcu_adapter import w4a16_moe


def _install_fake_ep_moe(monkeypatch):
    root = ModuleType("sglang")
    srt = ModuleType("sglang.srt")
    layers = ModuleType("sglang.srt.layers")
    moe = ModuleType("sglang.srt.layers.moe")
    ep_moe = ModuleType("sglang.srt.layers.moe.ep_moe")
    ep_moe_layer = ModuleType("sglang.srt.layers.moe.ep_moe.layer")

    class DeepEPMoE:
        def run_moe_core(self, dispatch_output):
            return ("upstream", dispatch_output)

    ep_moe_layer.DeepEPMoE = DeepEPMoE
    for name, module in (
        ("sglang", root),
        ("sglang.srt", srt),
        ("sglang.srt.layers", layers),
        ("sglang.srt.layers.moe", moe),
        ("sglang.srt.layers.moe.ep_moe", ep_moe),
        ("sglang.srt.layers.moe.ep_moe.layer", ep_moe_layer),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return DeepEPMoE


def _w4a16_layer():
    return SimpleNamespace(
        w13_weight=torch.empty((2, 4, 2), dtype=torch.uint8),
        w2_weight=torch.empty((2, 2, 1), dtype=torch.uint8),
        w13_weight_scale=torch.ones((2, 4, 1), dtype=torch.bfloat16),
        w2_weight_scale=torch.ones((2, 2, 1), dtype=torch.bfloat16),
        use_w4afp8=False,
    )


def test_deepep_ll_routes_only_w4a16(monkeypatch):
    monkeypatch.setattr(w4a16_moe, "_deepep_installed", False)
    DeepEPMoE = _install_fake_ep_moe(monkeypatch)
    calls = []

    def fake_kernel(*args, **kwargs):
        calls.append(args)
        assert kwargs == {"swiglu_limit": None}
        return "adapter-w4a16"

    monkeypatch.setattr(w4a16_moe, "w4a16_deepep_ll", fake_kernel)
    w4a16_moe.install_deepep_w4a16()

    layer = _w4a16_layer()
    dispatch = SimpleNamespace(
        hidden_states="expert-major",
        masked_m="counts",
        expected_m=17,
        topk_ids="topk-ids",
        topk_weights="topk-weights",
    )
    combine_input = ModuleType("sglang.srt.layers.moe.token_dispatcher.deepep")
    combine_input.DeepEPLLCombineInput = (
        lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.moe.token_dispatcher.deepep",
        combine_input,
    )
    assert DeepEPMoE().run_moe_core.__self__ is not None
    result = DeepEPMoE.run_moe_core(layer, dispatch)
    assert result.hidden_states == "adapter-w4a16"
    assert result.topk_ids == "topk-ids"
    assert result.topk_weights == "topk-weights"
    assert calls == [
        (
            "expert-major",
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            "counts",
            17,
        )
    ]


def test_deepep_ll_preserves_upstream_for_other_routes(monkeypatch):
    monkeypatch.setattr(w4a16_moe, "_deepep_installed", False)
    DeepEPMoE = _install_fake_ep_moe(monkeypatch)
    w4a16_moe.install_deepep_w4a16()

    layer = _w4a16_layer()
    normal_dispatch = SimpleNamespace(hidden_states="normal")
    assert DeepEPMoE.run_moe_core(layer, normal_dispatch) == (
        "upstream",
        normal_dispatch,
    )

    fp8_layer = _w4a16_layer()
    fp8_layer.use_w4afp8 = True
    ll_dispatch = SimpleNamespace(hidden_states="ll", masked_m=1, expected_m=1)
    assert DeepEPMoE.run_moe_core(fp8_layer, ll_dispatch) == (
        "upstream",
        ll_dispatch,
    )


def test_deepep_w4a16_shape_contract():
    with pytest.raises(ValueError, match="expert-major"):
        w4a16_moe.deepgemm_w4a16_deepep_ll(
            torch.zeros(4, 8, dtype=torch.bfloat16),
            torch.empty((2, 4, 2), dtype=torch.uint8),
            torch.empty((2, 2, 1), dtype=torch.uint8),
            torch.ones((2, 4, 1), dtype=torch.bfloat16),
            torch.ones((2, 2, 1), dtype=torch.bfloat16),
            torch.ones(2, dtype=torch.int32),
        )


def test_deepep_ct_packed_weight_names():
    # Community 0.5.22 CT WNA16TritonMoE preserves these names after converting
    # the underlying storage from int32 to uint8.
    layer = _w4a16_layer()
    layer.w13_weight_packed = layer.w13_weight
    layer.w2_weight_packed = layer.w2_weight
    del layer.w13_weight
    del layer.w2_weight
    assert w4a16_moe._is_deepep_w4a16_layer(layer)


def test_prepare_channel_scales(monkeypatch):
    vendor = ModuleType("deepgemm")
    vendor.w4a16_marlin_weight = lambda weights: weights
    monkeypatch.setitem(sys.modules, "deepgemm", vendor)
    weights = torch.zeros((2, 16, 256), dtype=torch.uint8)
    scales = torch.arange(32).reshape(2, 16, 1).to(torch.bfloat16)
    _, actual = w4a16_moe._prepare(weights, scales)
    assert actual.shape == (2, 16, 16)
    assert torch.equal(actual, scales.expand(2, 16, 16))


@pytest.mark.parametrize("name", ["w13_weight_zero_point", "w2_weight_zero_point"])
def test_deepep_does_not_select_asymmetric_weights(name):
    layer = _w4a16_layer()
    setattr(layer, name, torch.ones(1))
    assert not w4a16_moe._is_deepep_w4a16_layer(layer)


def test_deepep_swiglu_clamp(monkeypatch):
    gate_up = torch.tensor([[[20.0, -20.0, -30.0, 30.0]]])
    seen = []
    monkeypatch.setattr(w4a16_moe, "_prepare", lambda w, s: (w, s))

    def gemm(a, *args):
        seen.append(a.clone())
        return gate_up if len(seen) == 1 else a

    monkeypatch.setattr(w4a16_moe, "_grouped_gemm", gemm)
    w4a16_moe.deepgemm_w4a16_deepep_ll(
        torch.ones(1, 1, 2),
        torch.empty(1, 4, 1, dtype=torch.uint8),
        torch.empty(1, 2, 1, dtype=torch.uint8),
        torch.ones(1, 4, 1),
        torch.ones(1, 2, 1),
        torch.ones(1, dtype=torch.int32),
        swiglu_limit=10.0,
    )
    expected = torch.nn.functional.silu(
        torch.tensor([[[10.0, -20.0]]])
    ) * torch.tensor([[[-10.0, 10.0]]])
    torch.testing.assert_close(seen[1], expected)


def test_deepep_expert_major_pipeline_masks_empty_experts(monkeypatch):
    calls = []

    def fake_prepare(qweight, scales):
        return qweight, scales

    def fake_grouped(a, b, scales, counts, capacity, out_feats, out_dtype):
        calls.append((tuple(a.shape), counts.clone(), capacity, out_feats))
        visible = a[..., :out_feats]
        if visible.shape[-1] < out_feats:
            visible = torch.nn.functional.pad(
                visible, (0, out_feats - visible.shape[-1])
            )
        rows = torch.arange(capacity).view(1, capacity, 1)
        return visible.to(out_dtype) * (rows < counts.view(-1, 1, 1))

    monkeypatch.setattr(w4a16_moe, "_prepare", fake_prepare)
    monkeypatch.setattr(w4a16_moe, "_grouped_gemm", fake_grouped)

    hidden = torch.ones((2, 4, 3), dtype=torch.bfloat16)
    w1 = torch.empty((2, 6, 2), dtype=torch.uint8)
    w2 = torch.empty((2, 3, 1), dtype=torch.uint8)
    s1 = torch.ones((2, 6, 1), dtype=torch.bfloat16)
    s2 = torch.ones((2, 3, 1), dtype=torch.bfloat16)
    out = w4a16_moe.deepgemm_w4a16_deepep_ll(
        hidden, w1, w2, s1, s2, torch.tensor([2, 0]), expected_m=2
    )

    assert out.shape == (2, 4, 3)
    assert torch.count_nonzero(out[1]) == 0
    assert [c[0] for c in calls] == [(2, 4, 3), (2, 4, 3)]
    assert all(torch.equal(c[1], torch.tensor([2, 0])) for c in calls)
    assert [c[3] for c in calls] == [6, 3]
