# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""PR review regressions for default admission and irreversible publication.

CPU only: platform/version and the public quantization API are explicit doubles.
"""
import importlib
from types import SimpleNamespace as NS

import pytest

from hcu_adapter import ct_w4a8_registration as ct, hcu_config, registry
from hcu_adapter.bridges import vllm as host, vllm_pcp as pcp
from hcu_adapter.extensions import ExtensionCommitError, ExtensionContext
from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle
from test_vllm_flashmla import setup as prefill_setup
from test_vllm_mhc_bridge import seams
from test_vllm_flashmla_decode import setup as decode_setup


@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    monkeypatch.setattr(hcu_config, '_STATE', {
        'loaded': True, 'path': None, 'ops': {}, 'overrides': {}, 'hosts': {}})
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)


@pytest.mark.parametrize('arch,version', [
    ('gfx928', '0.33.0'), ('gfx92a', '0.33.0'),
    ('gfx936', '0.32.1'), ('gfx938', '0.32.1')])
def test_default_mhc_outside_scope_is_inert_and_healthy(seams, monkeypatch, arch, version):
    monkeypatch.delenv('HCU_VLLM_MHC_IMPL', raising=False)
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, arch, 'test'))
    monkeypatch.setattr(host, '_installed_vllm_version', lambda: version)
    assert seams.m.install() is False
    seams.m.require_healthy()
    assert not seams.calls and not seams.records
    assert seams.m._publication is None and seams.m._failure is None


@pytest.mark.parametrize('arch,version', [
    ('gfx928', '0.33.0'), ('gfx92a', '0.33.0'),
    ('gfx936', '0.32.1'), ('gfx938', '0.32.1')])
def test_default_decode_outside_scope_is_inert(decode_setup, monkeypatch, arch, version):
    bridge, target, _, calls = decode_setup
    original = target.rocm_sparse_attn_decode
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, arch, 'test'))
    monkeypatch.setattr(host, '_installed_vllm_version', lambda: version)
    assert bridge.install() is False
    bridge.require_healthy()
    assert not calls and target.rocm_sparse_attn_decode is original
    assert bridge._publication is None and bridge._failure is None


@pytest.mark.parametrize('arch,version', [
    ('gfx928', '0.33.0'), ('gfx92a', '0.33.0'),
    ('gfx936', '0.32.1'), ('gfx938', '0.32.1')])
def test_default_prefill_outside_scope_is_inert(prefill_setup, monkeypatch, arch, version):
    bridge, target, _, calls = prefill_setup
    original = target.rocm_sparse_attn_prefill
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, arch, 'test'))
    monkeypatch.setattr(host, '_installed_vllm_version', lambda: version)
    assert bridge.install() is False
    bridge.require_healthy()
    assert not calls and target.rocm_sparse_attn_prefill is original
    assert bridge._publication is None and bridge._failure is None


@pytest.mark.parametrize('arch', ['gfx936', 'gfx938'])
def test_default_prefill_supported_platform_still_installs(prefill_setup, monkeypatch, arch):
    bridge, _, _, calls = prefill_setup
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, arch, 'test'))
    bridge.install()
    bridge.require_healthy()
    assert calls and registry.get(bridge.OP)['implementation'] == 'native'


@pytest.mark.parametrize('source', ['explicit', 'config'])
def test_explicit_unsupported_prefill_still_refuses(monkeypatch, source):
    bridge = importlib.reload(
        importlib.import_module('hcu_adapter.bridges.vllm_flashmla'))
    monkeypatch.setattr(bridge.hcu_env, 'resolve', lambda *args, **kwargs: ('native', source))
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, 'gfx928', 'test'))
    try:
        with pytest.raises(bridge.FlashMLACompatibilityError):
            bridge.install()
    finally:
        importlib.reload(bridge)


@pytest.mark.parametrize('source', ['explicit', 'config'])
@pytest.mark.parametrize('owner', ['mhc', 'decode'])
def test_explicit_unsupported_request_still_refuses(monkeypatch, source, owner):
    name = 'vllm_mhc' if owner == 'mhc' else 'vllm_flashmla_decode'
    bridge = importlib.reload(importlib.import_module('hcu_adapter.bridges.' + name))
    mode = 'aiter' if owner == 'mhc' else 'native'
    monkeypatch.setattr(bridge.hcu_env, 'resolve', lambda *args, **kwargs: (mode, source))
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, 'gfx928', 'test'))
    error = bridge.MhcCompatibilityError if owner == 'mhc' else bridge.FlashMLADecodeCompatibilityError
    try:
        with pytest.raises(error):
            bridge.install()
    finally:
        importlib.reload(bridge)


@pytest.mark.parametrize('arch', ['gfx936', 'gfx938'])
def test_default_decode_supported_platform_still_installs(decode_setup, monkeypatch, arch):
    bridge, _, _, calls = decode_setup
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, arch, 'test'))
    assert bridge.install()
    bridge.require_healthy()
    assert calls and registry.get(bridge.OP)['implementation'] == 'native'


@pytest.mark.parametrize('arch,version', [
    ('gfx928', '0.32.1'), ('gfx92a', '0.32.1'),
    ('gfx936', '0.32.1'), ('gfx938', '0.32.1'),
    ('gfx928', '0.33.0'), ('gfx92a', '0.33.0')])
def test_general_plugin_does_not_require_v41_features(monkeypatch, arch, version):
    """Real host callback + both real feature installers; no V4.1 construction."""
    from hcu_adapter.bridges import vllm_mhc, vllm_flashmla_decode
    mhc = importlib.reload(vllm_mhc)
    decode = importlib.reload(vllm_flashmla_decode)
    for owner in (mhc, decode):
        monkeypatch.delenv(owner.MODE_VARIABLE, raising=False)
        monkeypatch.setattr(owner, 'import_module',
                            lambda name: pytest.fail('out-of-scope feature imported ' + name))
    monkeypatch.setattr(host, '_RUNTIME_ACTIONS', host.ProcessLocalActions())
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, arch, 'test'))
    monkeypatch.setattr(host, '_installed_vllm_version', lambda: version)
    monkeypatch.setattr(host, '_installed_adapter_version', lambda: 'test')
    for name in ('_ensure_hcu_platform_selected', 'configure_vllm_hip_runtime_path',
                 '_require_vllm_hip_runtime_abi', '_log_post_install_bindings'):
        monkeypatch.setattr(host, name, lambda: None)
    # Other owners are outside this finding. Keep the two affected installers
    # and their health checks real throughout the public runtime callback.
    for name, action in [('vllm_fp8', 'install'), ('vllm_scaled_mm', 'install'),
                         ('vllm_allreduce', 'prepare_runtime'),
                         ('vllm_sparse_indexer', 'install'),
                         ('vllm_flashmla', 'install'), ('vllm_pcp', 'install')]:
        owner = importlib.import_module('hcu_adapter.bridges.' + name)
        monkeypatch.setattr(owner, action, lambda: None)
        if hasattr(owner, 'require_healthy'):
            monkeypatch.setattr(owner, 'require_healthy', lambda: None)
    monkeypatch.setattr(host, 'activate_extensions', lambda context, **kwargs: ())
    monkeypatch.delenv('HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS', raising=False)
    monkeypatch.delenv('HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES', raising=False)
    try:
        host.hcu_runtime_plugin()
        host.hcu_runtime_plugin()
        host.require_runtime_healthy()
        assert mhc._publication is None and decode._publication is None
    finally:
        importlib.reload(mhc)
        importlib.reload(decode)
