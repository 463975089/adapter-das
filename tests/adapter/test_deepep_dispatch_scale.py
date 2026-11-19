# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import ast
import logging
import sys
from types import SimpleNamespace
from pathlib import Path
import unittest


class Tensor:
    pass


def load_bridge(native):
    source = Path(__file__).resolve().parents[2] / "hcu_adapter/bridges/sglang.py"
    tree = ast.parse(source.read_text())
    namespace = {"_ll_orig": native, "logging": logging, "_sys": sys,
                 "torch": SimpleNamespace(Tensor=Tensor)}
    for name in ("_ll_dispatch_upstream", "_ll_dispatch_v0521"):
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    binding = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Attribute)
                           and t.attr == "_hcu_quant_type_dispatch" for t in n.targets))
    exec(compile(ast.Module(body=[binding], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_ll_dispatch_v0521"]


class DeepEPDispatchScaleTests(unittest.TestCase):
    def test_channel_scale_override_and_blockwise_default_reach_native(self):
        calls = []
        def native(*args, **kwargs):
            calls.append((args, kwargs))
            return kwargs
        bridge = load_bridge(native)
        self.assertIs(bridge._hcu_quant_type_dispatch, native)
        self.assertEqual(bridge._hcu_quant_type_dispatch(quant_type=1)["quant_type"], 1)
        for extra, expected in [({}, 128), ({"quant_group_size": 0}, 0), ({"quant_group_size": 128}, 128)]:
            with self.subTest(extra=extra):
                weights = object()
                result = bridge(object(), object(), object(), 128, 192,
                                topk_weights=weights, use_fp8=True, **extra)
                self.assertEqual(result["quant_group_size"], expected)
                self.assertEqual(result["quant_type"], 2)
                self.assertIs(calls[-1][0][3], weights)
                self.assertFalse(result["fp8_round_scale"])
        result = bridge(object(), object(), object(), 128, 192,
                        topk_weights=object(), use_fp8=True, use_ue8m0=True)
        self.assertEqual(result["quant_type"], 3)
        self.assertEqual(result["quant_group_size"], 128)
        result = bridge(object(), object(), object(), 128, 192,
                        topk_weights=object(), use_fp8=False)
        self.assertEqual(result["quant_type"], 0)
        self.assertNotIn("quant_group_size", result)

    def test_native_int8_keywords_and_positional_weights_pass_through(self):
        calls = []
        def native(*args, **kwargs):
            calls.append((args, kwargs))
            return "native"
        bridge = load_bridge(native)
        prefix = (object(), object(), object())
        weights = Tensor()
        for args, kwargs in (
            ((weights, 96, 288), {"quant_type": 1, "return_recv_hook": False}),
            ((weights, 96, 288), {}),
            ((), {"topk_weight": weights, "num_experts": 288}),
        ):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(bridge(*prefix, *args, **kwargs), "native")
                self.assertEqual(calls[-1], (prefix + args, kwargs))

    def test_upstream_weights_from_original_caller_reach_translation(self):
        calls = []
        def native(*args, **kwargs):
            calls.append((args, kwargs))
        bridge = load_bridge(native)
        topk_weights = Tensor()
        bridge(object(), object(), object(), 96, 288, use_fp8=True)
        self.assertIs(calls[-1][0][3], topk_weights)
        self.assertEqual(calls[-1][1]["quant_type"], 2)


if __name__ == "__main__":
    unittest.main()
