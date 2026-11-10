# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU startup/rebind/gate contracts; framework/vendor doubles, no device proof.
Run: HCU_ADAPTER_HOST=none pytest tests/adapter/test_vllm_flashmla.py -v
"""
import importlib
import inspect
import multiprocessing
from types import SimpleNamespace

import pytest

from hcu_adapter import hcu_config, hcu_env, registry
from hcu_adapter.bridges import vllm as host


def official(q, kv, indices, topk_length, scale, head_dim, nope_head_dim,
             rope_head_dim, attn_sink, output, ragged_indices=None, ragged_indptr=None):
    return 'stock'


def provider(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None):
    raise AssertionError('CPU bridge tests must not launch vendor')


@pytest.fixture
def setup(monkeypatch):
    from hcu_adapter.bridges import vllm_flashmla
    bridge = importlib.reload(vllm_flashmla)
    monkeypatch.delenv(bridge.MODE_VARIABLE, raising=False)
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    monkeypatch.setattr(hcu_config, '_STATE', {'loaded': True, 'path': None,
                         'ops': {}, 'overrides': {}, 'hosts': {}})
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, 'gfx938', ''))
    monkeypatch.setattr(host, '_installed_vllm_version', lambda: '0.33.0')
    target = SimpleNamespace(rocm_sparse_attn_prefill=official)
    vendor = SimpleNamespace(flash_mla_sparse_fwd=provider)
    calls = []
    def load(name):
        calls.append(name)
        return {bridge.TARGET_MODULE: target, 'flash_mla.flash_mla_interface': vendor}[name]
    monkeypatch.setattr(bridge, 'import_module', load)
    yield bridge, target, vendor, calls
    # Do not leak activated owner state into other host lifecycle tests.
    importlib.reload(bridge)


def test_stock_does_not_import_host_or_vendor(setup, monkeypatch):
    bridge, target, _, calls = setup
    # Default is now native, so an explicit stock is what must stay inert.
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'stock')
    assert hcu_env.resolve(bridge.MODE_VARIABLE) == ('stock', 'explicit')
    assert bridge.install() is False
    assert calls == []
    assert target.rocm_sparse_attn_prefill is official
    assert registry.get(bridge.OP) is None


@pytest.mark.parametrize('choice', ['native', 'invalid'])
def test_kill_vetoes_before_import_or_signature(setup, monkeypatch, choice):
    bridge, target, _, calls = setup
    monkeypatch.setenv(bridge.MODE_VARIABLE, choice)
    monkeypatch.setenv('HCU_DISABLE_OPS', bridge.OP)
    target.rocm_sparse_attn_prefill = lambda: None
    assert bridge.install() is False
    assert calls == []
    assert registry.get(bridge.OP) is None


def test_native_preserves_original_and_registers(setup, monkeypatch):
    bridge, target, _, _ = setup
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    from hcu_adapter.kernels import mla_sparse_prefill_dtk
    seen = []
    monkeypatch.setattr(mla_sparse_prefill_dtk, 'try_prefill', lambda **kw: seen.append(kw) or False)
    assert bridge.install()
    routed = target.rocm_sparse_attn_prefill
    assert routed is not official
    assert routed.__wrapped__ is official
    assert inspect.signature(routed) == inspect.signature(official)
    assert bridge._publication.original is official
    record = registry.get(bridge.OP)
    assert record['mechanism'] == registry.MECH_REBIND
    assert record['target'] == bridge.TARGET
    assert record['implementation'] == 'native'
    assert bridge.install()  # Idempotent, no second wrapper.
    assert target.rocm_sparse_attn_prefill is routed
    # Lookup through the module global is the real model's calling convention.
    assert target.rocm_sparse_attn_prefill(*range(10)) == 'stock'
    assert seen[0]['provider'] is provider


@pytest.mark.parametrize('kind', ['missing', 'extra', 'default', 'keyword_only'])
def test_signature_drift_rejected_before_publication(setup, monkeypatch, kind):
    bridge, target, _, _ = setup
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    signature = inspect.signature(official)
    params = list(signature.parameters.values())
    if kind == 'missing': params.pop()
    if kind == 'extra': params.append(inspect.Parameter('new', inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None))
    if kind == 'default': params[-1] = params[-1].replace(default=0)
    if kind == 'keyword_only': params[-1] = params[-1].replace(kind=inspect.Parameter.KEYWORD_ONLY)
    def changed(*args, **kwargs): pass
    changed.__signature__ = signature.replace(parameters=params)
    target.rocm_sparse_attn_prefill = changed
    with pytest.raises(bridge.FlashMLACompatibilityError): bridge.install()
    assert target.rocm_sparse_attn_prefill is changed
    assert registry.get(bridge.OP) is None
    target.rocm_sparse_attn_prefill = official
    with pytest.raises(bridge.FlashMLACompatibilityError): bridge.install()


@pytest.mark.parametrize('change', ['stock', 'kill', 'invalid', 'binding'])
def test_late_policy_or_binding_drift_latches_failure(setup, monkeypatch, change):
    bridge, target, _, _ = setup
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    bridge.install()
    saved = target.rocm_sparse_attn_prefill
    if change == 'stock': monkeypatch.setenv(bridge.MODE_VARIABLE, 'stock')
    if change == 'kill': monkeypatch.setenv('HCU_DISABLE_OPS', bridge.OP)
    if change == 'invalid': monkeypatch.setenv(bridge.MODE_VARIABLE, 'bad')
    if change == 'binding': target.rocm_sparse_attn_prefill = official
    with pytest.raises(bridge.FlashMLACompatibilityError): bridge.require_healthy()
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    target.rocm_sparse_attn_prefill = saved
    with pytest.raises(bridge.FlashMLACompatibilityError): bridge.install()


@pytest.mark.parametrize('reason', ['version', 'arch', 'missing_vendor', 'vendor_signature'])
def test_explicit_native_unavailable_fails_closed(setup, monkeypatch, reason):
    bridge, target, vendor, _ = setup
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    if reason == 'version': monkeypatch.setattr(host, '_installed_vllm_version', lambda: '0.32.0')
    if reason == 'arch': monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, 'gfx950', ''))
    if reason == 'missing_vendor': del vendor.flash_mla_sparse_fwd
    if reason == 'vendor_signature': vendor.flash_mla_sparse_fwd = lambda: None
    with pytest.raises(bridge.FlashMLACompatibilityError): bridge.install()
    assert target.rocm_sparse_attn_prefill is official
    assert registry.get(bridge.OP) is None


def test_host_installs_prefill_before_extensions_and_checks_owner(setup, monkeypatch):
    bridge, target, _, calls = setup
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    monkeypatch.setattr(host, '_RUNTIME_ACTIONS', host.ProcessLocalActions())
    monkeypatch.setattr(host, '_ensure_vllm_compatible', lambda: '0.33.0')
    monkeypatch.setattr(host, '_ensure_hcu_platform_selected', lambda: None)
    monkeypatch.setattr(host, 'configure_vllm_hip_runtime_path', lambda: None)
    monkeypatch.setattr(host, '_require_vllm_hip_runtime_abi', lambda: None)
    monkeypatch.setattr(host, '_installed_adapter_version', lambda: 'test')
    from hcu_adapter.bridges import vllm_fp8, vllm_scaled_mm, vllm_allreduce
    from hcu_adapter.bridges import vllm_mhc, vllm_sparse_indexer
    for owner, method in ((vllm_fp8, 'install'), (vllm_scaled_mm, 'install'),
                          (vllm_allreduce, 'prepare_runtime'), (vllm_mhc, 'install'),
                          (vllm_sparse_indexer, 'install')):
        monkeypatch.setattr(owner, method, lambda: None)
    monkeypatch.setattr(vllm_mhc, 'require_healthy', lambda: calls.append('mhc-health'))
    monkeypatch.setattr(vllm_sparse_indexer, 'require_healthy', lambda: calls.append('indexer-health'))
    # Both FlashMLA tiers default to native now; stub the decode owner so this
    # ordering test stays a wiring check, not a real vLLM import.
    from hcu_adapter.bridges import vllm_flashmla_decode
    decode_bridge = importlib.reload(vllm_flashmla_decode)
    monkeypatch.setattr(decode_bridge, 'install', lambda: None)
    monkeypatch.setattr(decode_bridge, 'require_healthy', lambda: None)
    def extensions(context, *, health_check_sink):
        assert target.rocm_sparse_attn_prefill is not official
        calls.append('extensions')
        return ()
    monkeypatch.setattr(host, 'activate_extensions', extensions)
    host.hcu_runtime_plugin()
    host.hcu_runtime_plugin()
    assert calls.count('extensions') == 1
    assert calls.count('mhc-health') == 2
    assert calls.count('indexer-health') == 2
    target.rocm_sparse_attn_prefill = official
    with pytest.raises(bridge.FlashMLACompatibilityError):
        host.require_runtime_healthy()

@pytest.mark.parametrize('failed', [False, True])
def test_fork_adopts_healthy_binding_but_preserves_failure(setup, monkeypatch, failed):
    bridge, target, _, _ = setup
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    bridge.install()
    if failed:
        monkeypatch.setenv(bridge.MODE_VARIABLE, 'stock')
        with pytest.raises(bridge.FlashMLACompatibilityError): bridge.require_healthy()
        monkeypatch.setenv(bridge.MODE_VARIABLE, 'native')
    context = multiprocessing.get_context('fork')
    receive, send = context.Pipe(duplex=False)
    def child():
        try:
            if not failed:
                with pytest.raises(bridge.FlashMLACompatibilityError): bridge.require_healthy()
            bridge.install()
            bridge.require_healthy()
            send.send('healthy')
        except bridge.FlashMLACompatibilityError:
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
    import os
    from pathlib import Path
    import subprocess
    import sys
    root = str(Path(__file__).resolve().parents[2])
    paths = [root] + [p for p in sys.path if p.endswith(('site-packages', 'dist-packages'))]
    code = f'''import sys
sys.path[:0] = {paths!r}
from hcu_adapter.bridges import vllm_flashmla as b
assert b.install() is False
assert b._publication is None
assert 'vllm' not in sys.modules
assert 'flash_mla' not in sys.modules
print('fresh-stock-ok')
'''
    # The default is native, so drive the stock policy explicitly here.
    environment = dict(os.environ, HCU_ADAPTER_HOST='none',
                       HCU_VLLM_FLASHMLA_PREFILL='stock')
    environment.pop('HCU_DISABLE_OPS', None)
    result = subprocess.run([sys.executable, '-I', '-S', '-c', code],
                            env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'fresh-stock-ok'


def test_unset_environment_defaults_the_flashmla_tiers_to_native(setup, monkeypatch):
    bridge, _, _, _ = setup
    from hcu_adapter.bridges import vllm_flashmla_decode
    decode = importlib.reload(vllm_flashmla_decode)
    monkeypatch.delenv('HCU_VLLM_FLASHMLA_PREFILL', raising=False)
    monkeypatch.delenv('HCU_VLLM_FLASHMLA_DECODE', raising=False)
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    # Nothing configured anywhere: both tiers must answer native by default.
    assert hcu_env.resolve('HCU_VLLM_FLASHMLA_PREFILL', host='vllm') == ('native', 'default')
    assert hcu_env.resolve('HCU_VLLM_FLASHMLA_DECODE', host='vllm') == ('native', 'default')
    assert bridge.selected_mode() == 'native'
    assert decode.selected_mode() == 'native'
    # Explicit stock still disables either tier, and kill switch still wins.
    monkeypatch.setenv('HCU_VLLM_FLASHMLA_PREFILL', 'stock')
    monkeypatch.setenv('HCU_VLLM_FLASHMLA_DECODE', 'stock')
    assert bridge.selected_mode() == 'stock'
    assert decode.selected_mode() == 'stock'
    monkeypatch.setenv('HCU_VLLM_FLASHMLA_PREFILL', 'native')
    monkeypatch.setenv('HCU_VLLM_FLASHMLA_DECODE', 'native')
    monkeypatch.setenv('HCU_DISABLE_OPS', 'vllm_flashmla_prefill,vllm_flashmla_decode')
    assert bridge.selected_mode() == 'stock'
    assert decode.selected_mode() == 'stock'


def test_config_and_explicit_env_precedence(setup, monkeypatch):
    bridge, _, _, _ = setup
    hcu_config._STATE['overrides'][bridge.MODE_VARIABLE] = 'native'
    assert hcu_env.resolve(bridge.MODE_VARIABLE) == ('native', 'config')
    monkeypatch.setenv(bridge.MODE_VARIABLE, 'stock')
    assert hcu_env.resolve(bridge.MODE_VARIABLE) == ('stock', 'explicit')
    assert bridge.install() is False
