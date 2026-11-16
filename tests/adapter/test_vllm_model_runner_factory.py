"""Public runner factory admission and preservation of the official default."""
import importlib
import pytest
import hcu_adapter.bridges.vllm_model_runner as registry


@pytest.fixture
def api():
    yield importlib.reload(registry)
    importlib.reload(registry)


def test_default_runner_is_preserved(api):
    expected = object()
    assert api.create_model_runner(object(), "device", default=lambda: expected) is expected


def test_explicit_factory_is_scoped_and_owns_its_registration(api):
    expected = object()
    target = object()
    record = api.register_model_runner_factory(
        "test.target", matches=lambda config: config is target,
        create=lambda config, device: expected if device == "device" else None)
    api.require_model_runner_factory(record)
    assert api.create_model_runner(target, "device", default=lambda: pytest.fail("default selected")) is expected


def test_foreign_configuration_uses_default(api):
    api.register_model_runner_factory("test.target", matches=lambda config: False,
                                      create=lambda config, device: pytest.fail("foreign runner"))
    assert api.create_model_runner(None, None, default=lambda: "official") == "official"


def test_ambiguous_factory_and_late_registration_fail(api):
    for owner in ("one", "two"):
        api.register_model_runner_factory(owner, matches=lambda config: True,
                                          create=lambda config, device: object())
    with pytest.raises(RuntimeError, match="multiple"):
        api.create_model_runner(None, None, default=lambda: None)
    with pytest.raises(RuntimeError, match="frozen"):
        api.register_model_runner_factory("late", matches=lambda config: False,
                                          create=lambda config, device: object())
