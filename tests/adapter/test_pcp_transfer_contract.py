"""Optional transfer admission is owner-bound and retains default refusal."""
import pytest
from types import SimpleNamespace
from hcu_adapter.kernels import pcp_fork


def test_transfer_validator_registration(monkeypatch):
    monkeypatch.setattr(pcp_fork, '_kv_transfer_validator', None)
    assert pcp_fork.bound_kv_transfer_validator() is None
    with pytest.raises(pcp_fork.PcpAttentionError, match='callable'):
        pcp_fork.set_kv_transfer_validator(None)
    validator = lambda cfg: None
    pcp_fork.set_kv_transfer_validator(validator)
    pcp_fork.set_kv_transfer_validator(validator)
    assert pcp_fork.bound_kv_transfer_validator() is validator
    pcp_fork.validate_kv_transfer_config(SimpleNamespace(kv_transfer_config=object()))
    with pytest.raises(pcp_fork.PcpAttentionError, match='already bound'):
        pcp_fork.set_kv_transfer_validator(lambda cfg: None)
    assert pcp_fork.bound_kv_transfer_validator() is validator


def test_transfer_admission_default_refusal_and_no_transfer(monkeypatch):
    monkeypatch.setattr(pcp_fork, '_kv_transfer_validator', None)
    pcp_fork.validate_kv_transfer_config(SimpleNamespace(kv_transfer_config=None))
    with pytest.raises(pcp_fork.PcpAttentionError):
        pcp_fork.validate_kv_transfer_config(SimpleNamespace(kv_transfer_config=object()))


@pytest.mark.parametrize('result', [False, True, 0, 'accepted'])
def test_transfer_admission_requires_explicit_none_contract(monkeypatch, result):
    cfg = SimpleNamespace(kv_transfer_config=object())
    calls = []
    def validator(value):
        calls.append(value)
        return result
    monkeypatch.setattr(pcp_fork, '_kv_transfer_validator', validator)
    with pytest.raises(pcp_fork.PcpAttentionError):
        pcp_fork.validate_kv_transfer_config(cfg)
    assert calls == [cfg]


def test_transfer_admission_passes_owner_rejection(monkeypatch):
    def validator(cfg):
        raise ValueError('owner rejected this connector')
    monkeypatch.setattr(pcp_fork, '_kv_transfer_validator', validator)
    with pytest.raises(ValueError, match='owner rejected'):
        pcp_fork.validate_kv_transfer_config(SimpleNamespace(kv_transfer_config=object()))
