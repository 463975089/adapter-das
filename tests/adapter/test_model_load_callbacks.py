# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

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


def dispatch():
    api.dispatch_post_model_load(SimpleNamespace(vllm_config=object()), load_dummy_weights=False)


def test_import_is_framework_light(tmp_path):
    code = "import sys; sys.path.insert(0, sys.argv[1]); import hcu_adapter.model_load; assert not {'torch', 'vllm', 'sglang', 'sgl_kernel'} & sys.modules.keys()"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-S", "-B", "-c", code,
                    str(Path(api.__file__).resolve().parents[1])], check=True,
                   cwd=tmp_path, env=env, timeout=15)


@pytest.mark.parametrize("config_attribute", ["server_args", "vllm_config"])
def test_shared_dispatch_with_both_host_imports_blocked(config_attribute, tmp_path):
    """Registration and invocation must work without either framework installed."""
    code = r'''
import builtins
import sys
from types import SimpleNamespace as NS
sys.path.insert(0, sys.argv[2])
original_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'vllm', 'sglang', 'torch', 'sgl_kernel'}:
        raise AssertionError('shared dispatch imported framework: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked
from hcu_adapter import model_load as api
from hcu_adapter.bridges import sglang as sglang_bridge
from hcu_adapter.bridges import vllm_model_load as vllm_lifecycle
assert callable(sglang_bridge.install)
assert vllm_lifecycle.register_model_lifecycle_hooks is api.register_model_lifecycle_hooks
config = NS(parallel_config=NS(
    worker_cls='hcu_adapter.bridges.vllm_worker.Worker',
    distributed_executor_backend='uni', enable_elastic_ep=False))
worker = NS(**{sys.argv[1]: config})
events = []
def loaded(context):
    assert context.worker is worker and context.load_dummy_weights is False
    events.append('loaded')
def guarded(context):
    assert context.worker is worker
    events.append((context.operation, context.role))
hooks = dict(post_model_load_owner='shared', post_model_load_callback=loaded,
             model_mutation_owner='shared', model_mutation_guard=guarded)
api.preflight_model_lifecycle_hooks(**hooks)
api.register_model_lifecycle_hooks(**hooks)
api.dispatch_post_model_load(worker, load_dummy_weights=False)
api.dispatch_model_mutation(worker, operation='reload_weights', role='target')
assert events == ['loaded', ('reload_weights', 'target')]
assert not {'vllm', 'sglang', 'torch', 'sgl_kernel'} & sys.modules.keys()
print('shared lifecycle invocation: PASS')
'''
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-S", "-B", "-c", code, config_attribute,
         str(Path(api.__file__).resolve().parents[1])],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_host_admission_precedes_owners_and_failure_does_not_dispatch():
    calls = []
    api.register_post_model_load("shared", lambda context: calls.append("owner"))
    def admission():
        calls.append("admission")
        raise ValueError("host boundary unavailable")
    with pytest.raises(ValueError, match="host boundary unavailable"):
        api.dispatch_post_model_load(object(), load_dummy_weights=False,
                                     before_dispatch=admission)
    assert calls == ["admission"]
    api.dispatch_post_model_load(object(), load_dummy_weights=False,
                                 before_dispatch=lambda: calls.append("admitted"))
    assert calls == ["admission", "admitted", "owner"]


def test_no_owner_does_not_require_a_host_admission_or_worker_config():
    def forbidden():
        pytest.fail("empty shared registry must not inspect host state")
    api.dispatch_post_model_load(object(), load_dummy_weights=True,
                                 before_dispatch=forbidden)


def test_shared_mutation_accepts_host_owned_operation_identifiers(monkeypatch):
    monkeypatch.setattr(api, "_mutation_guards", api._Callbacks())
    seen = []
    api.register_model_mutation_guard("shared", lambda context: seen.append(
        (context.worker, context.operation, context.role)))
    worker = object()
    api.dispatch_model_mutation(worker, operation="update_weights_from_tensor", role="model")
    assert seen == [(worker, "update_weights_from_tensor", "model")]


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
