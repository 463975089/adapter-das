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


@pytest.mark.parametrize("active", [False, True])
def test_fork_does_not_acquire_lock_held_by_another_parent_thread(api, active):
    import os, select, signal, threading
    if not hasattr(os, "fork"):
        pytest.skip("requires fork")
    if active:
        api.register_model_runner_factory("fork", matches=lambda c: True,
                                          create=lambda c, d: "foreign")
    held, release = threading.Event(), threading.Event()
    def hold():
        with api._lock:
            held.set()
            release.wait(10)
    thread = threading.Thread(target=hold)
    thread.start()
    assert held.wait(3)
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read)
        try:
            result = api.create_model_runner(None, None, default=lambda: "official")
            message = result.encode()
        except RuntimeError as error:
            message = str(error).encode()
        except BaseException:
            message = b"unexpected failure"
        os.write(write, message)
        os._exit(0)
    os.close(write)
    try:
        ready, _, _ = select.select([read], [], [], 3)
        assert ready, "child deadlocked on inherited registry lock"
        result = os.read(read, 4096).decode()
        assert ("inherited across fork" in result) if active else result == "official"
    finally:
        release.set()
        thread.join(3)
        os.close(read)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)


@pytest.mark.parametrize("stage", ["false_predicate", "factory"])
def test_callback_cannot_remove_its_owner_and_return_success(api, stage):
    def matches(config):
        if stage == "false_predicate":
            api._factories.clear()
            return False
        return True
    def create(config, device):
        api._factories.clear()
        return object()
    api.register_model_runner_factory("callback", matches=matches, create=create)
    with pytest.raises(RuntimeError, match="ownership"):
        api.create_model_runner(None, None, default=lambda: "official")


@pytest.mark.parametrize("stage", ["false_predicate", "factory"])
def test_real_fork_inside_callback_cannot_return_runner(api, stage):
    import os, select
    if not hasattr(os, "fork"):
        pytest.skip("requires fork")
    state = {"pid": None, "child": False}
    read, write = os.pipe()
    def fork_callback():
        pid = os.fork()
        state["pid"] = pid
        state["child"] = pid == 0
    def matches(config):
        if stage == "false_predicate":
            fork_callback()
            return False
        return True
    def create(config, device):
        fork_callback()
        return "plugin"
    api.register_model_runner_factory("callback", matches=matches, create=create)
    try:
        try:
            result = api.create_model_runner(None, None, default=lambda: "official")
        except RuntimeError as error:
            result = str(error)
        if state["child"]:
            os.write(write, result.encode())
            os._exit(0)
        os.close(write)
        assert select.select([read], [], [], 3)[0]
        assert "inherited across fork" in os.read(read, 4096).decode()
        assert result == ("official" if stage == "false_predicate" else "plugin")
    finally:
        if not state["child"]:
            os.close(read)
            if state["pid"]:
                os.waitpid(state["pid"], 0)


@pytest.mark.parametrize("frozen", [False, True])
def test_duplicate_registration_returns_the_healthy_canonical_handle(api, frozen):
    matches = lambda config: False
    create = lambda config, device: object()
    first = api.register_model_runner_factory("retry", matches=matches, create=create)
    if frozen:
        assert api.create_model_runner(None, None, default=lambda: "stock") == "stock"
    second = api.register_model_runner_factory("retry", matches=matches, create=create)
    assert second is first
    api.require_model_runner_factory(first)
    api.require_model_runner_factory(second)
