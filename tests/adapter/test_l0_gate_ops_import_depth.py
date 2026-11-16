# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""ops-import-depth 门禁的拼写覆盖单测（2026-11-12 review 修复）。

原实现只查 ``from sglang.kernels.ops.<域>.<模块> import ...`` 的 4 级
ImportFrom；``import sglang.kernels.ops.<域>.<模块>``（ast.Import）和
``from sglang.kernels.ops.<域> import <模块>``（3 级、绑定模块体时）两种
等价写法绕过检查。本文件把扩展后的规则钉住：

  * ast.Import 深路径 —— 无条件是模块体绑定，要注记
  * 3 级 from-import —— 名字与域同名（``...activation import activation``）
    或被仓内自己的深度 import 证明是模块体时，要注记
  * 同名包面再导出（quantization.per_token_group_quant）的 3 级写法绑定
    包面 wrapper —— 钦定模式，恒绿
  * 包面符号导入（from ...activation import silu_and_mul）恒绿

Run: pytest -q tests/adapter/test_l0_gate_ops_import_depth.py
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "l0_gate_ops_depth", _ROOT / "ci" / "l0_gate.py"
)
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    """Serve a synthetic hcu_adapter/ tree to the gate's file walkers."""

    def _install(files):
        paths = []
        for name, text in files.items():
            p = tmp_path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="\n")
            paths.append(str(p))
        monkeypatch.setattr(gate, "iter_files", lambda *exts: iter(paths))
        monkeypatch.setattr(
            gate, "rel", lambda p: "hcu_adapter/" + os.path.basename(p)
        )
        return paths

    return _install


def _fails():
    r = gate.Report()
    gate.check_ops_import_depth(r)
    return [f for f in r.fails if "ops-import-depth" in f]


def test_plain_deep_import_statement_is_flagged(synthetic):
    synthetic({"a.py": "import sglang.kernels.ops.activation.activation\n"})
    (fail,) = _fails()
    assert "activation.activation" in fail


def test_plain_deep_import_with_note_passes(synthetic):
    synthetic(
        {"a.py": "# 替代档: 挂载面\nimport sglang.kernels.ops.activation.activation\n"}
    )
    assert _fails() == []


def test_deep_from_import_still_flagged(synthetic):
    """原 4 级 ImportFrom 臂不回归。"""
    synthetic({"a.py": "from sglang.kernels.ops.activation.activation import x\n"})
    (fail,) = _fails()
    assert "activation.activation" in fail


def test_self_named_from_import_is_flagged(synthetic):
    synthetic({"a.py": "from sglang.kernels.ops.activation import activation\n"})
    (fail,) = _fails()
    assert "activation.activation (from-import)" in fail


def test_proven_module_from_import_is_flagged(synthetic):
    """仓内深度 import 证明 fp8_kernel 是模块体 -> 3 级写法也要注记。"""
    synthetic(
        {
            "deep.py": (
                "# 替代档: 挂载面\n"
                "import sglang.kernels.ops.quantization.fp8_kernel\n"
            ),
            "shallow.py": (
                "from sglang.kernels.ops.quantization import fp8_kernel\n"
            ),
        }
    )
    (fail,) = [f for f in _fails() if "shallow.py" in f]
    assert "fp8_kernel (from-import)" in fail


def test_face_reexport_from_import_stays_green(synthetic):
    """同名包面再导出：3 级写法绑定包面 wrapper，钦定模式不红。"""
    synthetic(
        {
            "deep.py": (
                "# JIT 唯一实现（特化）\n"
                "import sglang.kernels.ops.quantization.per_token_group_quant\n"
            ),
            "face.py": (
                "from sglang.kernels.ops.quantization "
                "import per_token_group_quant\n"
            ),
        }
    )
    assert [f for f in _fails() if "face.py" in f] == []


def test_package_face_symbol_import_stays_green(synthetic):
    """包面符号导入（未证明为模块体）恒绿——桥规则的钦定写法。"""
    synthetic(
        {"a.py": "from sglang.kernels.ops.activation import silu_and_mul\n"}
    )
    assert _fails() == []


def test_package_import_stays_green(synthetic):
    synthetic({"a.py": "import sglang.kernels.ops.layernorm as _ops\n"})
    assert _fails() == []
