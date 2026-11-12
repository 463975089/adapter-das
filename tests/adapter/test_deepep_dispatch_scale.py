# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import ast
import logging
from pathlib import Path
import unittest


def load_bridge(native):
    source = Path(__file__).resolve().parents[2] / "hcu_adapter/bridges/sglang.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_ll_dispatch_v0521")
    namespace = {"_ll_orig": native, "logging": logging}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[node.name]


class DeepEPDispatchScaleTests(unittest.TestCase):
    def test_channel_scale_override_and_blockwise_default_reach_native(self):
        calls = []
        def native(*args, **kwargs):
            calls.append((args, kwargs))
            return kwargs
        bridge = load_bridge(native)
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


if __name__ == "__main__":
    unittest.main()
