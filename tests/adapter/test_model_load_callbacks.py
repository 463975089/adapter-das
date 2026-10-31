"""Process/owner contracts; no torch, vLLM, or device is needed."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hcu_adapter import model_load as api


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(api, "_callbacks", api._Callbacks())
    # Generic registry dispatch is independent of platform availability.
    monkeypatch.setattr(api, "require_post_model_load", lambda config: None)


def dispatch():
    api.dispatch_post_model_load(SimpleNamespace(vllm_config=object()), load_dummy_weights=False)


def test_import_is_framework_light():
    code = "import hcu_adapter.model_load; import sys; assert not {'torch', 'vllm', 'sgl_kernel'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-S", "-B", "-c", code], check=True,
                   cwd=Path(__file__).resolve().parents[2])


def test_named_owners_idempotency_and_repeat_successful_loads():
    calls = []
    callback = lambda context: calls.append(("b", context.load_dummy_weights))
    api.register_post_model_load("b", callback)
    api.register_post_model_load("b", callback)
    api.register_post_model_load("a", lambda context: calls.append(("a", False)))
    dispatch()
    dispatch()
    assert calls == [("a", False), ("b", False)] * 2
    with pytest.raises(api.PostModelLoadError, match="already registered"):
        api.register_post_model_load("b", lambda context: None)


@pytest.mark.parametrize("owner", [None, "", " ", 1])
def test_bad_owner(owner):
    with pytest.raises(ValueError):
        api.register_post_model_load(owner, lambda context: None)


def test_bad_callback():
    with pytest.raises(TypeError):
        api.register_post_model_load("owner", None)


def test_callback_failure_names_owner_stops_dispatch_and_keeps_cause():
    calls = []
    failure = ValueError("freeze failed")
    def fail(context):
        raise failure
    api.register_post_model_load("a", fail)
    api.register_post_model_load("b", lambda context: calls.append("wrong"))
    with pytest.raises(api.PostModelLoadError, match="owner 'a' failed") as caught:
        dispatch()
    assert caught.value.__cause__ is failure
    assert calls == []


def test_no_registration_during_dispatch():
    api.register_post_model_load("owner", lambda context: api.register_post_model_load("late", lambda c: None))
    with pytest.raises(api.PostModelLoadError, match="cannot register during"):
        dispatch()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires actual POSIX fork")
def test_fork_preserves_declarations_resets_dispatch_and_lock():
    reader, writer = os.pipe()
    calls = []
    callback = lambda context: calls.append("called")
    api.register_post_model_load("owner", callback)
    # Deliberately fork while the parent registry lock is held.
    with api._callbacks.lock:
        api._callbacks.dispatching = True
        pid = os.fork()
        if pid == 0:
            try:
                os.close(reader)
                dispatch()
                assert calls == ["called"]
                api.register_post_model_load("owner", callback)
                dispatch()
                assert calls == ["called", "called"]
                os.write(writer, b"ok")
                os._exit(0)
            except BaseException:
                os._exit(1)
        api._callbacks.dispatching = False
    os.close(writer)
    try:
        import select
        if not select.select([reader], [], [], 10)[0]:
            import signal
            os.kill(pid, signal.SIGKILL)
            pytest.fail("fork child timed out")
        assert os.read(reader, 16) == b"ok"
    finally:
        os.close(reader)
        waited, status = os.waitpid(pid, 0)
    assert waited == pid and os.waitstatus_to_exitcode(status) == 0
    dispatch()
    assert calls == ["called"]
