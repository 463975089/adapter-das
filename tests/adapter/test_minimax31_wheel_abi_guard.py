"""Known 26/25 wrapper and native ABI pairing checks, no GPU."""
import hashlib
import importlib
import importlib.util
import types
import pytest
from hcu_adapter.bridges import vllm_minimax_abi as bridge
from pathlib import Path

SOURCE = Path(__file__).parent / 'fixtures' / 'minimax31_known_26_wrapper.py'
def load_source(path):
    spec=importlib.util.spec_from_file_location('fixture_'+path.stem,str(path))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.fused_minimax_m3_qknorm_rope_kv_insert

def setup(monkeypatch, fn, schema):
    native=types.SimpleNamespace(default=types.SimpleNamespace(_schema=schema))
    ops=types.SimpleNamespace(fused_minimax_m3_qknorm_rope_kv_insert=fn)
    torch=types.SimpleNamespace(ops=types.SimpleNamespace(_C=types.SimpleNamespace(fused_minimax_m3_qknorm_rope_kv_insert=native)))
    monkeypatch.setattr(bridge,'import_module',lambda name: ops if name=='vllm._custom_ops' else torch)
    return ops

def test_known_26_new_native_passes_untouched(monkeypatch):
    fn=load_source(SOURCE)
    assert hashlib.sha256(__import__('inspect').getsource(fn).encode()).hexdigest()==bridge._SOURCE_SHA256
    ops=setup(monkeypatch,fn,bridge._NEW_SCHEMA)
    assert bridge.install() is False and ops.fused_minimax_m3_qknorm_rope_kv_insert is fn

def test_25_wrapper_rejected_for_new_native(monkeypatch):
    fn=importlib.import_module('vllm._custom_ops').fused_minimax_m3_qknorm_rope_kv_insert
    setup(monkeypatch,fn,bridge._NEW_SCHEMA)
    with pytest.raises(RuntimeError,match='wrapper drift'):bridge.install()

def test_changed_26_source_rejected(monkeypatch,tmp_path):
    p=tmp_path/'changed.py'
    p.write_text(SOURCE.read_text().replace('    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(', '    # unapproved wrapper change\n    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(',1))
    fn=load_source(p)
    setup(monkeypatch,fn,bridge._NEW_SCHEMA)
    with pytest.raises(RuntimeError,match='wrapper drift'):bridge.install()

def test_marker_rejected(monkeypatch):
    fn=load_source(SOURCE);fn.__globals__[bridge._MARKER]=object()
    setup(monkeypatch,fn,bridge._NEW_SCHEMA)
    with pytest.raises(RuntimeError,match='wrapper drift'):bridge.install()

def test_release_25_old_native_passes(monkeypatch):
    fn=importlib.import_module('vllm._custom_ops').fused_minimax_m3_qknorm_rope_kv_insert
    ops=setup(monkeypatch,fn,bridge._OLD_SCHEMA)
    assert bridge.install() is False and ops.fused_minimax_m3_qknorm_rope_kv_insert is fn

def test_unknown_schema_rejected(monkeypatch):
    fn=load_source(SOURCE);setup(monkeypatch,fn,'unrecognized')
    with pytest.raises(RuntimeError,match='schema drift'):bridge.install()

def test_known_26_old_native_bridge_preserves_alias(monkeypatch):
    fn=load_source(SOURCE)
    original=fn
    captured=[]
    class Native:
        default=types.SimpleNamespace(_schema=bridge._OLD_SCHEMA)
        def __call__(self,*args): captured.append(args)
    native=Native()
    ops=types.SimpleNamespace(fused_minimax_m3_qknorm_rope_kv_insert=fn)
    torch=types.SimpleNamespace(ops=types.SimpleNamespace(_C=types.SimpleNamespace(fused_minimax_m3_qknorm_rope_kv_insert=native)))
    monkeypatch.setattr(bridge,'import_module',lambda name: ops if name=='vllm._custom_ops' else torch)
    monkeypatch.setattr(bridge.registry,'register',lambda **kw: None)
    assert bridge.install() is True and ops.fused_minimax_m3_qknorm_rope_kv_insert is original
    args=[object() for _ in range(25)]
    original(*args)
    assert captured == [tuple(args)]
    with pytest.raises(NotImplementedError):original(*args,write_qkv=False)
    assert captured == [tuple(args)]
    assert bridge.install() is False
