# SPDX-License-Identifier: Apache-2.0
import importlib
import importlib.util
from pathlib import Path
import types
import pytest

from hcu_adapter.bridges import vllm_minimax_abi as bridge


def test_real_wrapper_alias_defaults_and_rejections(monkeypatch):
    # Fresh upstream function clone avoids changing the process-wide function.
    fixture = Path(__file__).parent / "fixtures" / "minimax31_known_26_wrapper.py"
    spec = importlib.util.spec_from_file_location("minimax_known_26", fixture)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    orig = module.fused_minimax_m3_qknorm_rope_kv_insert
    fn = types.FunctionType(orig.__code__, dict(orig.__globals__), orig.__name__,
                            orig.__defaults__, orig.__closure__)
    fn.__kwdefaults__ = orig.__kwdefaults__
    alias = fn
    captured = []
    class Native:
        default = types.SimpleNamespace(_schema=bridge._OLD_SCHEMA)
        def __call__(self, *args):
            captured.append(args)
    native = Native()
    fake_ops = types.SimpleNamespace(fused_minimax_m3_qknorm_rope_kv_insert=fn)
    fake_torch = types.SimpleNamespace(ops=types.SimpleNamespace(
        _C=types.SimpleNamespace(fused_minimax_m3_qknorm_rope_kv_insert=native)))
    monkeypatch.setattr(bridge, "import_module", lambda name: fake_ops if name == "vllm._custom_ops" else fake_torch)
    monkeypatch.setattr(bridge.registry, "register", lambda **kw: None)
    assert bridge.install()
    assert alias is fake_ops.fused_minimax_m3_qknorm_rope_kv_insert
    values = [object() for _ in range(25)]
    alias(*values)
    assert captured == [tuple(values)]
    with pytest.raises(NotImplementedError):
        alias(*values, write_qkv=False)
    assert captured == [tuple(values)]
    assert not bridge.install()


def test_schema_drift_refused(monkeypatch):
    fn = lambda: None
    native = types.SimpleNamespace(default=types.SimpleNamespace(_schema="unexpected"))
    ops = types.SimpleNamespace(fused_minimax_m3_qknorm_rope_kv_insert=fn)
    torch = types.SimpleNamespace(ops=types.SimpleNamespace(_C=types.SimpleNamespace(
        fused_minimax_m3_qknorm_rope_kv_insert=native)))
    monkeypatch.setattr(bridge, "import_module", lambda name: ops if name == "vllm._custom_ops" else torch)
    with pytest.raises(RuntimeError, match="schema drift"):
        bridge.install()



def test_new_native_rejects_unknown_wrapper_and_switch_refuses_old(monkeypatch):
    fn = lambda: None
    native = types.SimpleNamespace(default=types.SimpleNamespace(_schema=bridge._NEW_SCHEMA))
    ops = types.SimpleNamespace(fused_minimax_m3_qknorm_rope_kv_insert=fn)
    torch = types.SimpleNamespace(ops=types.SimpleNamespace(_C=types.SimpleNamespace(
        fused_minimax_m3_qknorm_rope_kv_insert=native)))
    monkeypatch.setattr(bridge, "import_module", lambda name: ops if name == "vllm._custom_ops" else torch)
    with pytest.raises(RuntimeError, match="wrapper drift"):
        bridge.install()
    assert ops.fused_minimax_m3_qknorm_rope_kv_insert is fn
    native.default._schema = bridge._OLD_SCHEMA
    monkeypatch.setattr(bridge, "op_disabled", lambda name: True)
    with pytest.raises(RuntimeError, match="disabled"):
        bridge.install()
    assert ops.fused_minimax_m3_qknorm_rope_kv_insert is fn
