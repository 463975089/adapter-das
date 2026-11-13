# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real quantization owners/loader; only framework providers are doubles."""
from types import SimpleNamespace as NS

import pytest

from hcu_adapter import ct_w4a8_registration as ct, dsv41_activation as channel
from hcu_adapter import hcu_config, registry
from hcu_adapter.extensions import (
    ExtensionCommitError, ExtensionContext, ExtensionStatus,
    RequiredExtensionError, activate_extensions,
)
from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle


@pytest.fixture(params=[ct, channel], ids=['ct', 'channel'])
def owner(request, monkeypatch):
    module = request.param
    monkeypatch.setattr(hcu_config, '_STATE', {
        'loaded': True, 'path': None, 'ops': {}, 'overrides': {}, 'hosts': {}})
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    for name, value in [('_publication', None), ('_failure', None),
                        ('_publishing', False), ('_provider_binder', None)]:
        monkeypatch.setattr(module, name, value)
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    monkeypatch.setenv(module.MODE_VARIABLE, 'native')
    configs, calls = {}, []
    methods = []
    cls = type('Config', (), {})

    def register(quantization):
        def publish(config):
            calls.append(quantization)
            configs[quantization] = config
            methods.append(quantization)
        return publish

    def resolve(quantization):
        return configs[quantization]

    quant = NS(register_quantization_config=register,
               get_quantization_config=resolve, QUANTIZATION_METHODS=methods)
    if module is ct:
        monkeypatch.setattr(module, 'import_module', lambda name: quant)
        monkeypatch.setattr(module, '_config_provider', lambda: cls)
        method = module.NAME
    else:
        candidate = NS(HcuDSV41ChannelFP8Config=cls, _is_hcu=lambda: False)
        monkeypatch.setattr(module, '_import_module', lambda name: (
            quant if name == 'vllm.model_executor.layers.quantization' else candidate))
        monkeypatch.setattr(module, '_engram_edp_hooks', (lambda: None, lambda: None))
        monkeypatch.setattr(module, '_dspark_graph_hooks', (lambda: None, lambda: None))
        method = module.QUANTIZATION_METHOD
    spec = module.extension_spec()
    point = NS(name=spec.name, load=lambda: module.extension_spec)
    return NS(module=module, spec=spec, point=point, calls=calls,
              quant=quant, cls=cls, method=method)


def context(owner, required='optional'):
    return ExtensionContext(
        '0.1.0', '0.33.0',
        requested_extensions={owner.spec.name} if required == 'extension' else (),
        requested_capabilities=owner.spec.capabilities if required == 'capability' else (),
    )


def load(owner, ctx=None):
    lifecycle = ProcessLocalLifecycle(runtime_action='review-test')
    reports = activate_extensions(
        ctx or context(owner), entry_points=[owner.point],
        health_check_sink=lifecycle.record_extension_health)
    lifecycle.record_extension_reports(reports)
    return reports, lifecycle


def disable(owner, monkeypatch, source):
    if source == 'kill':
        monkeypatch.setenv('HCU_DISABLE_OPS', owner.module.OP)
    else:
        monkeypatch.setenv(owner.module.MODE_VARIABLE, source)


def restore(owner, monkeypatch):
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    monkeypatch.setenv(owner.module.MODE_VARIABLE, 'native')


@pytest.mark.parametrize('source', ['stock', 'kill'])
@pytest.mark.parametrize('required', ['optional', 'extension', 'capability'])
def test_first_disabled_owner_never_reports_active(owner, monkeypatch, source, required):
    disable(owner, monkeypatch, source)
    if required == 'optional':
        reports, _ = load(owner)
    else:
        with pytest.raises(RequiredExtensionError) as refusal:
            load(owner, context(owner, required))
        reports = refusal.value.reports
    assert len(reports) == 1
    assert reports[0].status is ExtensionStatus.FAILED
    assert owner.calls == []
    assert owner.module._publication is None
    assert owner.module._failure is None
    restore(owner, monkeypatch)
    reports, lifecycle = load(owner, context(owner, required))
    assert reports[0].status is ExtensionStatus.ACTIVE
    lifecycle.require_extension_healthy(next(iter(owner.spec.capabilities)))
    assert owner.calls == [owner.method]


@pytest.mark.parametrize('source', ['invalid', 'stock', 'kill'])
@pytest.mark.parametrize('entry', ['activate', 'health'])
def test_published_policy_drift_is_fatal_even_after_restore(owner, monkeypatch, source, entry):
    _, lifecycle = load(owner)
    lifecycle.require_extension_healthy(next(iter(owner.spec.capabilities)))
    disable(owner, monkeypatch, source)
    action = (lambda: load(owner)) if entry == 'activate' else owner.module.check_health
    with pytest.raises(ExtensionCommitError):
        action()
    assert owner.module._failure is not None
    assert owner.quant.get_quantization_config(owner.method) is owner.cls
    restore(owner, monkeypatch)
    for action in (lambda: load(owner), owner.module.check_health):
        with pytest.raises(ExtensionCommitError):
            action()
    with pytest.raises(RuntimeError):
        lifecycle.require_extension_healthy(next(iter(owner.spec.capabilities)))
    assert owner.calls == [owner.method]


def test_invalid_before_publication_is_correctable(owner, monkeypatch):
    monkeypatch.setenv(owner.module.MODE_VARIABLE, 'invalid')
    reports, _ = load(owner)
    assert reports[0].status is ExtensionStatus.FAILED
    assert owner.module._failure is None and owner.module._publication is None
    restore(owner, monkeypatch)
    reports, lifecycle = load(owner)
    assert reports[0].status is ExtensionStatus.ACTIVE
    lifecycle.require_extension_healthy(next(iter(owner.spec.capabilities)))
    assert owner.calls == [owner.method]
