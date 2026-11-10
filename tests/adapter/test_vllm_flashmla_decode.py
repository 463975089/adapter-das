# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Decode bridge startup/gate contracts with host and vendor doubles (CPU only).
Run: HCU_ADAPTER_HOST=none pytest tests/adapter/test_vllm_flashmla_decode.py -q
CPU doubles do not qualify the DTK kernel, Graph, model or performance.
"""
import importlib
import inspect
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from hcu_adapter import hcu_config, hcu_env, registry
from hcu_adapter.bridges import vllm as host

# The target is a real module namespace: the bridge verifies that the method's
# globals ARE that module and that the global name is read at call time.
_MODULE_SOURCE = '''
def rocm_sparse_attn_decode(q, kv_cache, swa_k_cache, swa_only, topk_indices,
        topk_lens, swa_indices, swa_lens, swa_ragged_indices, swa_ragged_indptr,
        topk_ragged_indices, topk_ragged_indptr, attn_sink, scale, head_dim,
        nope_head_dim, rope_head_dim, output, extra_cache_nan_free=False,
        adaptive_splits=False, inv_rope_positions=None,
        inv_rope_cos_sin_cache=None, output_mxfp8=None) -> int:
    return 'stock'

def rocm_inverse_rope_mxfp8_rows(o, positions, cos_sin_cache, rope_head_dim,
        out_data, out_scale):
    return None

class DeepseekV41ROCMAiterMLAAttention:
    def _forward_decode(self, q, positions, kv_cache, swa_metadata,
            attn_metadata, swa_only, output, output_mxfp8=None) -> int:
        return rocm_sparse_attn_decode(q, kv_cache, None, swa_only, None, None,
            None, None, None, None, None, None, None, 1.0, 512, 448, 64,
            output, output_mxfp8=output_mxfp8)
'''

FACTORY_SOURCE = '''
def get_mla_metadata(*args, **kwargs):
    return (object(), None)
'''

PROVIDER_SOURCE = '''
def flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v,
        tile_scheduler_metadata, num_splits=None, softmax_scale=None,
        causal=False, is_fp8_kvcache=False, indices=None, attn_sink=None,
        extra_k_cache=None, extra_indices_in_kvcache=None, topk_length=None,
        extra_topk_length=None):
    return None, None
'''


def build_module(source=_MODULE_SOURCE, name='fake_rocm'):
    module = types.ModuleType(name)
    exec(compile(source, '<' + name + '>', 'exec'), module.__dict__)
    module.current_platform = types.SimpleNamespace(is_fp8_fnuz=lambda: False)
    return module


@pytest.fixture
def setup(monkeypatch):
    from hcu_adapter.bridges import vllm_flashmla_decode
    bridge = importlib.reload(vllm_flashmla_decode)
    monkeypatch.delenv(bridge.MODE_VARIABLE, raising=False)
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    monkeypatch.setattr(hcu_config, '_STATE', {'loaded': True, 'path': None,
                         'ops': {}, 'overrides': {}, 'hosts': {}})
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, 'gfx938', ''))
    monkeypatch.setattr(host, '_installed_vllm_version', lambda: '0.33.0')
    target = build_module()
    vendor = build_module(PROVIDER_SOURCE + FACTORY_SOURCE, 'fake_flash_mla')
    calls = []
    def load(name):
        calls.append(name)
        return {bridge.TARGET_MODULE: target, 'flash_mla.flash_mla_interface': vendor}[name]
    monkeypatch.setattr(bridge, 'import_module', load)
    yield bridge, target, vendor, calls
    importlib.reload(bridge)


def test_stock_does_not_import_host_or_vendor(setup, monkeypatch):
    bridge, target, _, calls = setup
    original = target.rocm_sparse_attn_decode
    # Default is now native, so an explicit stock is what must stay inert.
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'stock')
    assert hcu_env.resolve(bridge.MODE_VARIABLE) == ('stock', 'explicit')
    assert bridge.install() is False
    assert calls == [] and target.rocm_sparse_attn_decode is original
    assert registry.get(bridge.OP) is None


@pytest.mark.parametrize('choice', ['native', 'invalid'])
def test_kill_vetoes_before_import(setup, monkeypatch, choice):
    bridge, target, _, calls = setup
    original = target.rocm_sparse_attn_decode
    monkeypatch.setenv(bridge.MODE_VARIABLE, choice)
    monkeypatch.setenv('HCU_DISABLE_OPS', bridge.OP)
    assert bridge.install() is False
    assert calls == [] and target.rocm_sparse_attn_decode is original
    assert registry.get(bridge.OP) is None


def test_native_preserves_original_and_method(setup, monkeypatch):
    bridge, target, vendor, _ = setup
    original = target.rocm_sparse_attn_decode
    method = target.DeepseekV41ROCMAiterMLAAttention._forward_decode
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    from hcu_adapter.kernels import mla_sparse_decode_dtk
    seen = []
    monkeypatch.setattr(mla_sparse_decode_dtk, 'try_decode',
                        lambda **kw: seen.append(kw) or None)
    assert bridge.install()
    routed = target.rocm_sparse_attn_decode
    assert routed is not original and routed.__wrapped__ is original
    assert inspect.signature(routed) == inspect.signature(original)
    assert target.DeepseekV41ROCMAiterMLAAttention._forward_decode is method
    assert bridge._publication.original is original
    assert bridge._publication.method is method
    record = registry.get(bridge.OP)
    assert record['mechanism'] == registry.MECH_REBIND
    assert record['target'] == bridge.TARGET and record['implementation'] == 'native'
    assert bridge.install()
    assert target.rocm_sparse_attn_decode is routed
    # The real call path reads the module global, so the rebind is intercepted.
    assert target.DeepseekV41ROCMAiterMLAAttention()._forward_decode(
        None, None, None, None, None, False, None) == 'stock'
    assert seen and seen[0]['provider'] is vendor.flash_mla_with_kvcache
    assert seen[0]['metadata_factory'] is vendor.get_mla_metadata
    assert seen[0]['epilogue'] is target.rocm_inverse_rope_mxfp8_rows


@pytest.mark.parametrize('value', [0, 7])
def test_selected_rotation_is_returned_without_fallback(setup, monkeypatch, value):
    bridge, target, _, _ = setup
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    from hcu_adapter.kernels import mla_sparse_decode_dtk
    # The bridge imports the kernel entry during install, so patch it first.
    # Zero is a valid BF16 result; only None may fall through to stock.
    monkeypatch.setattr(mla_sparse_decode_dtk, 'try_decode', lambda **kw: value)
    bridge.install()
    assert target.DeepseekV41ROCMAiterMLAAttention()._forward_decode(
        None, None, None, None, None, False, None) == value


@pytest.mark.parametrize('kind', ['target_missing', 'target_default', 'target_kwonly',
                                  'target_return', 'method_missing', 'method_return',
                                  'epilogue', 'wrapped', 'callsite'])
def test_abi_drift_rejected_before_publication(setup, monkeypatch, kind):
    bridge, target, _, _ = setup
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    original = target.rocm_sparse_attn_decode
    cls = target.DeepseekV41ROCMAiterMLAAttention
    if kind.startswith('target_') or kind == 'wrapped':
        source = _MODULE_SOURCE
        if kind == 'target_missing':
            source = source.replace('output_mxfp8=None) -> int', ') -> int')
        if kind == 'target_default':
            source = source.replace('adaptive_splits=False', 'adaptive_splits=None')
        if kind == 'target_kwonly':
            source = source.replace('output_mxfp8=None) -> int', '*, output_mxfp8=None) -> int')
        if kind == 'target_return':
            source = source.replace('output_mxfp8=None) -> int:', 'output_mxfp8=None):')
        if kind == 'wrapped':
            source = source.replace("    return 'stock'",
                                    "    return 'stock'\nrocm_sparse_attn_decode.__wrapped__ = rocm_sparse_attn_decode")
        drifted = build_module(source)
        target.rocm_sparse_attn_decode = drifted.rocm_sparse_attn_decode
        target.DeepseekV41ROCMAiterMLAAttention = drifted.DeepseekV41ROCMAiterMLAAttention
    if kind == 'method_missing':
        source = _MODULE_SOURCE.replace('output, output_mxfp8=None) -> int',
                                        'output) -> int')
        drifted = build_module(source)
        target.rocm_sparse_attn_decode = drifted.rocm_sparse_attn_decode
        target.DeepseekV41ROCMAiterMLAAttention = drifted.DeepseekV41ROCMAiterMLAAttention
    if kind == 'method_return':
        source = _MODULE_SOURCE.replace(
            'attn_metadata, swa_only, output, output_mxfp8=None) -> int:',
            'attn_metadata, swa_only, output, output_mxfp8=None):')
        drifted = build_module(source)
        target.rocm_sparse_attn_decode = drifted.rocm_sparse_attn_decode
        target.DeepseekV41ROCMAiterMLAAttention = drifted.DeepseekV41ROCMAiterMLAAttention
    if kind == 'epilogue':
        target.rocm_inverse_rope_mxfp8_rows = lambda o, positions: None
    if kind == 'callsite':
        # Same 8-parameter shape but no module-global read: the method was
        # rewritten to call something else, so the rebind could never intercept.
        source = _MODULE_SOURCE.replace(
            'return rocm_sparse_attn_decode(q, kv_cache, None, swa_only, None, None,\n'
            '            None, None, None, None, None, None, None, 1.0, 512, 448, 64,\n'
            '            output, output_mxfp8=output_mxfp8)',
            'return 0')
        drifted = build_module(source)
        target.rocm_sparse_attn_decode = drifted.rocm_sparse_attn_decode
        target.DeepseekV41ROCMAiterMLAAttention = drifted.DeepseekV41ROCMAiterMLAAttention
    with pytest.raises(bridge.FlashMLADecodeCompatibilityError):
        bridge.install()
    assert registry.get(bridge.OP) is None
    # epilogue/callsite drift leaves the global itself intact; the others replace it.
    if kind in ('target_missing', 'target_default', 'target_kwonly', 'target_return', 'wrapped'):
        assert target.rocm_sparse_attn_decode is not original


@pytest.mark.parametrize('reason', ['version', 'arch', 'fnuz', 'vendor_missing', 'vendor_abi'])
def test_explicit_native_unavailable_fails_closed(setup, monkeypatch, reason):
    bridge, target, vendor, _ = setup
    original = target.rocm_sparse_attn_decode
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    if reason == 'version': monkeypatch.setattr(host, '_installed_vllm_version', lambda: '0.32.0')
    if reason == 'arch': monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, 'gfx936', ''))
    if reason == 'fnuz': target.current_platform = types.SimpleNamespace(is_fp8_fnuz=lambda: True)
    if reason == 'vendor_missing': del vendor.flash_mla_with_kvcache
    if reason == 'vendor_abi':
        vendor.flash_mla_with_kvcache = lambda q: None
    with pytest.raises(bridge.FlashMLADecodeCompatibilityError):
        bridge.install()
    assert target.rocm_sparse_attn_decode is original
    assert registry.get(bridge.OP) is None


@pytest.mark.parametrize('change', ['stock', 'kill', 'invalid', 'global', 'method', 'epilogue'])
def test_late_drift_latches_failure(setup, monkeypatch, change):
    bridge, target, _, _ = setup
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    bridge.install()
    cls = target.DeepseekV41ROCMAiterMLAAttention
    saved_global, saved_method, saved_epilogue = (
        target.rocm_sparse_attn_decode, cls._forward_decode, target.rocm_inverse_rope_mxfp8_rows)
    if change == 'stock': monkeypatch.setenv(bridge.MODE_VARIABLE, 'stock')
    if change == 'kill': monkeypatch.setenv('HCU_DISABLE_OPS', bridge.OP)
    if change == 'invalid': monkeypatch.setenv(bridge.MODE_VARIABLE, 'bad')
    if change == 'global': target.rocm_sparse_attn_decode = build_module().rocm_sparse_attn_decode
    if change == 'method': cls._forward_decode = lambda self: 0
    if change == 'epilogue': target.rocm_inverse_rope_mxfp8_rows = lambda *a: None
    with pytest.raises(bridge.FlashMLADecodeCompatibilityError):
        bridge.require_healthy()
    # Restoring inputs must not re-admit the process.
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    target.rocm_sparse_attn_decode, cls._forward_decode = saved_global, saved_method
    target.rocm_inverse_rope_mxfp8_rows = saved_epilogue
    with pytest.raises(bridge.FlashMLADecodeCompatibilityError):
        bridge.install()


def test_host_orders_prefill_then_decode_then_extensions(setup, monkeypatch):
    bridge, target, _, _ = setup
    from hcu_adapter.bridges import vllm_flashmla
    prefill = importlib.reload(vllm_flashmla)
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    monkeypatch.setenv(prefill.MODE_VARIABLE, 'native')
    monkeypatch.setattr(host, '_RUNTIME_ACTIONS', host.ProcessLocalActions())
    monkeypatch.setattr(host, '_ensure_vllm_compatible', lambda: '0.33.0')
    monkeypatch.setattr(host, '_ensure_hcu_platform_selected', lambda: None)
    monkeypatch.setattr(host, 'configure_vllm_hip_runtime_path', lambda: None)
    monkeypatch.setattr(host, '_require_vllm_hip_runtime_abi', lambda: None)
    monkeypatch.setattr(host, '_installed_adapter_version', lambda: 'test')
    from hcu_adapter.bridges import vllm_fp8, vllm_scaled_mm, vllm_allreduce
    from hcu_adapter.bridges import vllm_mhc, vllm_sparse_indexer
    for owner, name in ((vllm_fp8, 'install'), (vllm_scaled_mm, 'install'),
                        (vllm_allreduce, 'prepare_runtime'), (vllm_mhc, 'install'),
                        (vllm_sparse_indexer, 'install')):
        monkeypatch.setattr(owner, name, lambda: None)
    monkeypatch.setattr(vllm_mhc, 'require_healthy', lambda: None)
    monkeypatch.setattr(vllm_sparse_indexer, 'require_healthy', lambda: None)
    order = []
    monkeypatch.setattr(prefill, 'install', lambda: order.append('prefill'))
    monkeypatch.setattr(bridge, 'install', lambda: order.append('decode'))
    def extensions(context, *, health_check_sink):
        order.append('extensions')
        return ()
    monkeypatch.setattr(host, 'activate_extensions', extensions)
    host.hcu_runtime_plugin()
    assert order == ['prefill', 'decode', 'extensions']


@pytest.mark.parametrize('failed', [False, True])
def test_fork_adopts_healthy_binding_but_preserves_failure(setup, monkeypatch, failed):
    bridge, target, _, _ = setup
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    bridge.install()
    if failed:
        monkeypatch.setenv(bridge.MODE_VARIABLE, 'stock')
        with pytest.raises(bridge.FlashMLADecodeCompatibilityError):
            bridge.require_healthy()
        monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    context = multiprocessing.get_context('fork')
    receive, send = context.Pipe(duplex=False)
    def child():
        try:
            if not failed:
                with pytest.raises(bridge.FlashMLADecodeCompatibilityError):
                    bridge.require_healthy()
            bridge.install()
            bridge.require_healthy()
            send.send('healthy')
        except bridge.FlashMLADecodeCompatibilityError:
            send.send('refused')
        finally:
            send.close()
    process = context.Process(target=child)
    process.start()
    send.close()
    try:
        assert receive.poll(10), 'child did not report health'
        assert receive.recv() == ('refused' if failed else 'healthy')
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        receive.close()


def test_fresh_interpreter_has_stock_policy_and_no_host_import():
    root = str(Path(__file__).resolve().parents[2])
    paths = [root] + [p for p in sys.path if p.endswith(('site-packages', 'dist-packages'))]
    code = f'''import sys
sys.path[:0] = {paths!r}
from hcu_adapter.bridges import vllm_flashmla_decode as b
assert b.install() is False
assert b._publication is None
assert 'vllm' not in sys.modules
assert 'flash_mla' not in sys.modules
print('fresh-stock-ok')
'''
    # The default is native, so drive the stock policy explicitly here.
    environment = dict(os.environ, HCU_ADAPTER_HOST='none',
                       HCU_VLLM_FLASHMLA_DECODE='stock')
    environment.pop('HCU_DISABLE_OPS', None)
    result = subprocess.run([sys.executable, '-I', '-S', '-c', code],
                            env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'fresh-stock-ok'
