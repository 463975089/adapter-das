# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Missing-symbol repair retains methods/aliases and validates both consumers."""

import types
from unittest.mock import patch

import pytest

from hcu_adapter import int8_linear


def consumer(path, owner, entry):
    module = types.ModuleType(path)
    exec(f"def {entry}(self, value):\n    return int8_scaled_mm(value)\n", vars(module))
    setattr(module, owner, type(owner, (), {entry: getattr(module, entry)}))
    return module


def test_both_consumers_and_cached_aliases():
    modules = [consumer(*target) for target in int8_linear._TARGETS]
    aliases = [getattr(getattr(m, target[1]), target[2])
               for m, target in zip(modules, int8_linear._TARGETS)]
    implementation = lambda value: value + 17
    with patch.object(int8_linear, "import_module", side_effect=lambda p: next(
            m for m in modules if m.__name__ == p)), \
            patch("sgl_kernel.int8_scaled_mm", implementation), \
            patch("hcu_adapter.registry.register") as register:
        int8_linear.install()
        int8_linear.install()
    assert register.call_count == 4
    assert {call.kwargs["target"] for call in register.call_args_list} == {
        module.__name__ + ".int8_scaled_mm" for module in modules}
    for module, target, alias in zip(modules, int8_linear._TARGETS, aliases):
        assert getattr(getattr(module, target[1]), target[2]) is alias
        assert module.int8_scaled_mm is implementation
        assert alias(None, 4) == 21


@pytest.mark.parametrize("bad_index", [0, 1])
def test_drift_refuses_before_any_binding(bad_index):
    modules = [consumer(*target) for target in int8_linear._TARGETS]
    target = int8_linear._TARGETS[bad_index]
    setattr(getattr(modules[bad_index], target[1]), target[2], lambda *args: None)
    with patch.object(int8_linear, "import_module", side_effect=lambda p: next(
            m for m in modules if m.__name__ == p)), \
            patch("hcu_adapter.registry.register") as register:
        with pytest.raises(RuntimeError, match="consumer changed"):
            int8_linear.install()
    register.assert_not_called()
    assert all(not hasattr(module, "int8_scaled_mm") for module in modules)
