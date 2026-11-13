# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""PR review regressions for failures that follow an irreversible publication.

Both extensions publish process state and then run a step that can fail. Core has
no unregister, so a failure after that point must latch rather than be reported
as a recoverable error the host would swallow.

CPU only: the public quantization registry is an explicit double.
"""
import importlib
from types import SimpleNamespace as NS

import pytest

from hcu_adapter import ct_w4a8_registration as ct, hcu_config, registry
from hcu_adapter.bridges import vllm_pcp as pcp
from hcu_adapter.extensions import ExtensionCommitError, ExtensionContext
from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle


@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    monkeypatch.setattr(hcu_config, '_STATE', {
        'loaded': True, 'path': None, 'ops': {}, 'overrides': {}, 'hosts': {}})
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)


@pytest.fixture
def ct_state(monkeypatch):
    for name, value in [('_publication', None), ('_failure', None),
                        ('_publishing', False), ('_provider_binder', None)]:
        monkeypatch.setattr(ct, name, value)
    monkeypatch.setenv(ct.MODE_VARIABLE, 'native')
    configs = {}
    calls = []
    cls = type('Config', (), {})

    def register(quantization):
        def publish(config):
            calls.append(quantization)
            configs[quantization] = config
        return publish

    def resolve(quantization):
        return configs[quantization]

    quant = NS(register_quantization_config=register,
               get_quantization_config=resolve, QUANTIZATION_METHODS=configs)
    monkeypatch.setattr(ct, 'import_module', lambda name: quant)
    monkeypatch.setattr(ct, '_config_provider', lambda: cls)
    return NS(quant=quant, calls=calls, cls=cls)


CTX = ExtensionContext('0.1.0', '0.33.0')


def test_ct_registry_failure_after_publication_is_fatal_and_latched(ct_state, monkeypatch):
    original = registry.register

    def fail(**kwargs):
        raise RuntimeError('adapter record unavailable')

    monkeypatch.setattr(registry, 'register', fail)
    with pytest.raises(ExtensionCommitError):
        ct.activate_extension(CTX)
    assert ct_state.calls == [ct.NAME]
    assert ct_state.quant.get_quantization_config(ct.NAME) is ct_state.cls
    assert ct._failure is not None
    monkeypatch.setattr(registry, 'register', original)
    for action in (ct.check_health, lambda: ct.activate_extension(CTX)):
        with pytest.raises(ExtensionCommitError):
            action()
    assert ct_state.calls == [ct.NAME]


@pytest.mark.parametrize('drift', ['missing', 'implementation', 'target', 'source', 'unreadable'])
def test_ct_health_checks_adapter_record_and_latches(ct_state, monkeypatch, drift):
    ct.activate_extension(CTX)
    ct.check_health()
    original = dict(registry.get(ct.OP))
    get_record = registry.get
    if drift == 'unreadable':
        def unreadable(op):
            raise RuntimeError('record unreadable')
        monkeypatch.setattr(registry, 'get', unreadable)
    elif drift == 'missing':
        registry._RECORDS.pop(ct.OP)
    else:
        forged = dict(original)
        forged[drift] = 'foreign'
        registry._RECORDS[ct.OP] = forged
    with pytest.raises(ExtensionCommitError, match='registry record'):
        ct.check_health()
    monkeypatch.setattr(registry, 'get', get_record)
    registry._RECORDS[ct.OP] = original
    # Repairing external state must not clear the process failure latch.
    for action in (ct.check_health, lambda: ct.activate_extension(CTX)):
        with pytest.raises(ExtensionCommitError):
            action()


def test_ct_healthy_activation_publishes_once(ct_state):
    ct.activate_extension(CTX)
    ct.activate_extension(CTX)
    ct.check_health()
    assert ct_state.calls == [ct.NAME]
    assert registry.get(ct.OP)['implementation'] == 'native'


@pytest.fixture
def pcp_state(monkeypatch):
    monkeypatch.setattr(pcp, '_publication', None)
    monkeypatch.setattr(pcp, '_failure', None)
    monkeypatch.setattr(pcp, '_lifecycle', ProcessLocalLifecycle(
        runtime_action='vllm-pcp', error_type=pcp.PcpCompatibilityError))
    monkeypatch.setenv(pcp.MODE_VARIABLE, 'native')


def test_pcp_health_invalid_policy_stays_failed_after_restoring_mode(pcp_state, monkeypatch):
    pcp.install()
    monkeypatch.setenv(pcp.MODE_VARIABLE, 'invalid')
    with pytest.raises(pcp.PcpCompatibilityError, match='policy invalid'):
        pcp.require_healthy()
    monkeypatch.setenv(pcp.MODE_VARIABLE, 'native')
    for action in (pcp.require_healthy, pcp.install):
        with pytest.raises(pcp.PcpCompatibilityError):
            action()


def test_pcp_invalid_before_publication_can_be_corrected(pcp_state, monkeypatch):
    monkeypatch.setenv(pcp.MODE_VARIABLE, 'invalid')
    with pytest.raises(ValueError):
        pcp.require_healthy()
    assert pcp._failure is None
    monkeypatch.setenv(pcp.MODE_VARIABLE, 'native')
    assert pcp.install()
    pcp.require_healthy()


# ---------------------------------------------------------------- P2-1
# The published name cannot be removed, so a policy change after publication is
# irreversible: it must latch instead of returning early and reporting health.


def test_ct_published_then_invalid_policy_latches_and_stays_failed(ct_state, monkeypatch):
    ct.activate_extension(CTX)
    ct.check_health()
    monkeypatch.setenv(ct.MODE_VARIABLE, 'bogus')
    with pytest.raises(ExtensionCommitError, match='policy invalid after publication'):
        ct.activate_extension(CTX)
    assert ct._failure is not None
    # Restoring a legal value must not clear the latched failure.
    monkeypatch.setenv(ct.MODE_VARIABLE, 'native')
    for action in (ct.check_health, lambda: ct.activate_extension(CTX)):
        with pytest.raises(ExtensionCommitError):
            action()
    assert ct_state.calls == [ct.NAME]


@pytest.mark.parametrize('source', ['stock', 'kill'])
def test_ct_published_then_disabled_latches(ct_state, monkeypatch, source):
    ct.activate_extension(CTX)
    if source == 'stock':
        monkeypatch.setenv(ct.MODE_VARIABLE, 'stock')
    else:
        monkeypatch.setenv('HCU_DISABLE_OPS', ct.OP)
    with pytest.raises(ExtensionCommitError, match='policy changed to stock after publication'):
        ct.activate_extension(CTX)
    assert ct._failure is not None
    # The public selector still resolves the config, which is exactly why the
    # process must be refused rather than silently treated as stock.
    assert ct_state.quant.get_quantization_config(ct.NAME) is ct_state.cls
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    monkeypatch.setenv(ct.MODE_VARIABLE, 'native')
    for action in (ct.check_health, lambda: ct.activate_extension(CTX)):
        with pytest.raises(ExtensionCommitError):
            action()


@pytest.mark.parametrize('source', ['stock', 'kill', 'invalid'])
def test_ct_health_refuses_policy_change_after_publication(ct_state, monkeypatch, source):
    ct.activate_extension(CTX)
    ct.check_health()
    if source == 'stock':
        monkeypatch.setenv(ct.MODE_VARIABLE, 'stock')
    elif source == 'kill':
        monkeypatch.setenv('HCU_DISABLE_OPS', ct.OP)
    else:
        monkeypatch.setenv(ct.MODE_VARIABLE, 'bogus')
    with pytest.raises(ExtensionCommitError):
        ct.check_health()
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    monkeypatch.setenv(ct.MODE_VARIABLE, 'native')
    with pytest.raises(ExtensionCommitError):
        ct.check_health()


def test_ct_invalid_policy_before_publication_stays_correctable(ct_state, monkeypatch):
    """A pure configuration error before anything is published can be fixed."""
    monkeypatch.setenv(ct.MODE_VARIABLE, 'bogus')
    with pytest.raises(ValueError):
        ct.activate_extension(CTX)
    assert ct._failure is None and ct._publication is None
    assert ct_state.calls == []
    monkeypatch.setenv(ct.MODE_VARIABLE, 'native')
    ct.activate_extension(CTX)
    ct.check_health()
    assert ct_state.calls == [ct.NAME]
