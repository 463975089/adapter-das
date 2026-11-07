"""Packed-layout name-test seam; real callback integration is checked separately."""

from types import SimpleNamespace as NS

import pytest
import torch

from hcu_adapter import ct_w4a8_int8_moe as native


def _loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
    method = self.scheme
    loaded_weight = (
        loaded_weight.t().contiguous()
        if method.__class__.__name__ in [
            "CompressedTensorsWNA16MarlinMoE",
            "CompressedTensorsWNA16MoE",
            "CompressedTensorsWNA16TritonMoE",
        ] and "zero" not in weight_name else loaded_weight
    )
    return loaded_weight


def test_layout_extension_retains_original_function_aliases_and_instructions():
    import types

    function = types.FunctionType(_loader.__code__, globals())
    alias, original = function, function.__code__
    value = torch.arange(6).reshape(2, 3)
    candidate = type(native._CANDIDATE_SCHEME, (), {})()
    item = NS(scheme=candidate)
    assert alias(item, None, value, "weight_scale", "w1", 0) is value
    adapted = native._adapted_loader_code(function)
    assert adapted.co_code == original.co_code
    function.__code__ = adapted
    assert function is alias and type(candidate).__name__ == native._CANDIDATE_SCHEME
    torch.testing.assert_close(
        alias(item, None, value, "weight_scale", "w1", 0), value.t(),
    )
    assert native._adapted_loader_code(function) is adapted
    assert alias(item, None, value, "weight_zero", "w1", 0) is value
    item.scheme = type("UnrelatedScheme", (), {})()
    assert alias(item, None, value, "weight_scale", "w1", 0) is value
    for name in native._LOADER_SCHEMES:
        item.scheme = type(name, (), {})()
        torch.testing.assert_close(
            alias(item, None, value, "weight_packed", "w1", 0), value.t(),
        )


def test_loader_signature_drift_is_refused_without_mutation():
    def changed(self, param, value):
        return value
    original = changed.__code__
    with pytest.raises(RuntimeError, match="signature"):
        native._adapted_loader_code(changed)
    assert changed.__code__ is original


def test_loader_layout_test_drift_is_refused_without_mutation():
    def changed(self, param, loaded_weight, weight_name, shard_id, expert_id):
        return loaded_weight
    original = changed.__code__
    with pytest.raises(RuntimeError, match="scheme test"):
        native._adapted_loader_code(changed)
    assert changed.__code__ is original
