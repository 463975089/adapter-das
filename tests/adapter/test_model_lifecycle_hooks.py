"""Public two-hook transaction contracts, without importing torch or vLLM."""

from functools import partial
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

import pytest

from hcu_adapter import model_load as api


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(api, "_callbacks", api._Callbacks())
    monkeypatch.setattr(api, "_mutation_guards", api._Callbacks())


def hooks(events):
    def loaded(context):
        events.append(("loaded", context.load_dummy_weights))
    def guarded(context):
        events.append(("guarded", context.operation, context.role))
    return dict(post_model_load_owner="test.freeze", post_model_load_callback=loaded,
                model_mutation_owner="test.protection", model_mutation_guard=guarded)


def snapshot():
    return (dict(api._callbacks.callbacks), dict(api._mutation_guards.callbacks))


def dispatch(which):
    if which == "load":
        api.dispatch_post_model_load(NS(vllm_config=object()), load_dummy_weights=False)
    else:
        api.dispatch_model_mutation(object(), operation="start_weight_update", role="draft")


def test_import_and_both_public_apis_are_framework_light(tmp_path):
    source = """
import sys
sys.path.insert(0, sys.argv[1])
from hcu_adapter.model_load import preflight_model_lifecycle_hooks, register_model_lifecycle_hooks
def callback(context): pass
kwargs = dict(post_model_load_owner='one', post_model_load_callback=callback,
              model_mutation_owner='one', model_mutation_guard=callback)
assert preflight_model_lifecycle_hooks(**kwargs) is None
assert register_model_lifecycle_hooks(**kwargs) is None
assert register_model_lifecycle_hooks(**kwargs) is None
assert not {'torch', 'vllm', 'sgl_kernel'} & sys.modules.keys()
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-S", "-B", "-c", source,
                    str(Path(api.__file__).resolve().parents[1])], check=True,
                   cwd=tmp_path, env=env, timeout=15)


def test_preflight_has_no_registration_or_callback_effect_and_commit_dispatches():
    events = []
    values = hooks(events)
    before = snapshot()
    assert api.preflight_model_lifecycle_hooks(**values) is None
    assert snapshot() == before and events == []
    assert api.register_model_lifecycle_hooks(**values) is None
    dispatch("load")
    dispatch("mutation")
    assert events == [("loaded", False), ("guarded", "start_weight_update", "draft")]


@pytest.mark.parametrize("existing", ["post", "mutation", "both"])
def test_legacy_declarations_and_pair_idempotency(monkeypatch, existing):
    values = hooks([])
    if existing in ("post", "both"):
        api.register_post_model_load(values["post_model_load_owner"], values["post_model_load_callback"])
    if existing in ("mutation", "both"):
        api.register_model_mutation_guard(values["model_mutation_owner"], values["model_mutation_guard"])
    api.register_model_lifecycle_hooks(**values)
    before = snapshot()
    def forbidden(*args):
        pytest.fail("an unchanged pair must not attempt a registration")
    monkeypatch.setattr(api, "register_post_model_load", forbidden)
    monkeypatch.setattr(api, "register_model_mutation_guard", forbidden)
    api.preflight_model_lifecycle_hooks(**values)
    api.register_model_lifecycle_hooks(**values)
    assert snapshot() == before


@pytest.mark.parametrize("conflict", ["post", "mutation"])
def test_first_or_second_owner_conflict_prevents_all_writes(conflict):
    values = hooks([])
    if conflict == "post":
        api.register_post_model_load(values["post_model_load_owner"], lambda context: None)
        expected = api.PostModelLoadError
    else:
        api.register_model_mutation_guard(values["model_mutation_owner"], lambda context: None)
        expected = api.ModelMutationError
    before = snapshot()
    for method in (api.preflight_model_lifecycle_hooks, api.register_model_lifecycle_hooks):
        with pytest.raises(expected, match="already registered"):
            method(**values)
        assert snapshot() == before


def test_successful_preflight_is_not_a_reservation():
    values = hooks([])
    api.preflight_model_lifecycle_hooks(**values)
    api.register_model_mutation_guard(values["model_mutation_owner"], lambda context: None)
    before = snapshot()
    with pytest.raises(api.ModelMutationError, match="already registered"):
        api.register_model_lifecycle_hooks(**values)
    assert snapshot() == before


@pytest.mark.parametrize("key", ["post_model_load_owner", "model_mutation_owner"])
@pytest.mark.parametrize("bad", [None, "", " ", 1])
def test_invalid_named_owner_is_rejected_before_registration(key, bad):
    values = hooks([])
    values[key] = bad
    for method in (api.preflight_model_lifecycle_hooks, api.register_model_lifecycle_hooks):
        with pytest.raises(ValueError, match="non-empty string"):
            method(**values)
        assert snapshot() == ({}, {})


def callbacks(kind):
    async def coroutine(context): pass
    async def async_generator(context): yield None
    def generator(context): yield None
    class AsyncCallable:
        async def __call__(self, context): pass
    class GeneratorCallable:
        def __call__(self, context): yield None
    return {"coroutine": coroutine, "async-generator": async_generator,
            "generator": generator, "async-callable": AsyncCallable(),
            "generator-callable": GeneratorCallable(), "partial": partial(coroutine)}[kind]


@pytest.mark.parametrize("key", ["post_model_load_callback", "model_mutation_guard"])
@pytest.mark.parametrize("kind", ["coroutine", "async-generator", "generator", "async-callable", "generator-callable", "partial", "not-callable"])
def test_invalid_callbacks_and_async_forms_never_partially_register(key, kind):
    values = hooks([])
    values[key] = None if kind == "not-callable" else callbacks(kind)
    for method in (api.preflight_model_lifecycle_hooks, api.register_model_lifecycle_hooks):
        with pytest.raises(TypeError, match="callable|synchronous"):
            method(**values)
        assert snapshot() == ({}, {})


def test_legacy_mutation_guard_also_rejects_async_partial():
    with pytest.raises(TypeError, match="synchronous"):
        api.register_model_mutation_guard("partial", callbacks("partial"))


@pytest.mark.parametrize("when", ["before-second-insert", "after-second-insert"])
@pytest.mark.parametrize("existing", ["neither", "post", "mutation"])
@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_second_registration_failure_rolls_back_both_preserving_existing_owners(
    monkeypatch, when, existing, failure_type,
):
    values = hooks([])
    api.register_post_model_load("unrelated.load", lambda context: None)
    api.register_model_mutation_guard("unrelated.guard", lambda context: None)
    if existing == "post":
        api.register_post_model_load(values["post_model_load_owner"], values["post_model_load_callback"])
    if existing == "mutation":
        api.register_model_mutation_guard(values["model_mutation_owner"], values["model_mutation_guard"])
    before = snapshot()
    identities = (id(api._callbacks.callbacks), id(api._mutation_guards.callbacks))
    real = api.register_model_mutation_guard
    failure = failure_type("injected second registration failure")
    def second(owner, callback):
        # Prove failure occurs after the first real registration, not preflight.
        assert api._callbacks.callbacks[values["post_model_load_owner"]] is values["post_model_load_callback"]
        if when == "after-second-insert":
            real(owner, callback)
            assert api._mutation_guards.callbacks[owner] is callback
        raise failure
    monkeypatch.setattr(api, "register_model_mutation_guard", second)
    with pytest.raises(failure_type) as caught:
        api.register_model_lifecycle_hooks(**values)
    assert caught.value is failure
    assert snapshot() == before
    assert identities == (id(api._callbacks.callbacks), id(api._mutation_guards.callbacks))
    monkeypatch.setattr(api, "register_model_mutation_guard", real)
    api.register_model_lifecycle_hooks(**values)  # Rollback leaves a usable API.


@pytest.mark.parametrize("which", ["load", "mutation"])
@pytest.mark.parametrize("method_name", ["preflight_model_lifecycle_hooks", "register_model_lifecycle_hooks"])
def test_adding_pair_from_either_dispatch_is_refused_without_writes(which, method_name):
    values = hooks([])
    def callback(context):
        getattr(api, method_name)(**values)
    if which == "load":
        api.register_post_model_load("outer", callback)
        expected = api.PostModelLoadError
    else:
        api.register_model_mutation_guard("outer", callback)
        expected = api.ModelMutationError
    before = snapshot()
    with pytest.raises(expected, match="cannot register lifecycle hooks during"):
        dispatch(which)
    assert snapshot() == before
    assert not api._callbacks.dispatching and not api._mutation_guards.dispatching


@pytest.mark.parametrize("which", ["load", "mutation"])
def test_entirely_unchanged_pair_can_be_rechecked_during_dispatch(which):
    values = hooks([])
    def callback(context):
        api.preflight_model_lifecycle_hooks(**values)
        api.register_model_lifecycle_hooks(**values)
    values["post_model_load_callback" if which == "load" else "model_mutation_guard"] = callback
    api.register_model_lifecycle_hooks(**values)
    dispatch(which)


@pytest.mark.parametrize("which", ["load", "mutation"])
def test_recursive_dispatch_is_still_rejected(which):
    values = hooks([])
    values["post_model_load_callback" if which == "load" else "model_mutation_guard"] = lambda context: dispatch(which)
    api.register_model_lifecycle_hooks(**values)
    with pytest.raises(api.PostModelLoadError if which == "load" else api.ModelMutationError, match="recursive"):
        dispatch(which)


@pytest.mark.parametrize("fail", [False, True])
def test_other_thread_dispatch_cannot_observe_one_hook_transaction(monkeypatch, fail):
    events, errors = [], []
    values = hooks(events)
    at_second, release, observer_attempted, observer_done = [threading.Event() for _ in range(4)]
    real = api.register_model_mutation_guard
    def second(owner, callback):
        assert values["post_model_load_owner"] in api._callbacks.callbacks
        assert values["model_mutation_owner"] not in api._mutation_guards.callbacks
        at_second.set()
        assert release.wait(5)
        if fail:
            raise ValueError("second failed")
        real(owner, callback)
    monkeypatch.setattr(api, "register_model_mutation_guard", second)
    def registration():
        try:
            api.register_model_lifecycle_hooks(**values)
        except BaseException as error:
            errors.append(error)
    def observation():
        try:
            observer_attempted.set()
            dispatch("load")
            dispatch("mutation")
        except BaseException as error:
            errors.append(error)
        finally:
            observer_done.set()
    register_thread = threading.Thread(target=registration, daemon=True)
    observe_thread = threading.Thread(target=observation, daemon=True)
    register_thread.start()
    try:
        assert at_second.wait(5)
        observe_thread.start()
        assert observer_attempted.wait(5)
        assert not observer_done.wait(0.05)
        assert events == []
    finally:
        release.set()
        register_thread.join(5)
        if observe_thread.ident is not None:
            observe_thread.join(5)
    assert not register_thread.is_alive() and not observe_thread.is_alive()
    if fail:
        assert len(errors) == 1 and isinstance(errors[0], ValueError)
        assert events == [] and snapshot() == ({}, {})
    else:
        assert errors == []
        assert events == [("loaded", False), ("guarded", "start_weight_update", "draft")]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_with_all_parent_locks_held_keeps_pair_and_resets_sync_only():
    events = []
    values = hooks(events)
    api.register_model_lifecycle_hooks(**values)
    ctx = multiprocessing.get_context("fork")
    reader, writer = ctx.Pipe(False)
    def child():
        try:
            assert not api._callbacks.dispatching and not api._mutation_guards.dispatching
            api.preflight_model_lifecycle_hooks(**values)
            api.register_model_lifecycle_hooks(**values)
            dispatch("load")
            dispatch("mutation")
            writer.send(events)
        finally:
            writer.close()
    held, release = threading.Event(), threading.Event()
    def holder():
        # A *different* thread owns the parent locks at fork. A same-thread
        # RLock can remain reentrant in the child and give a false-positive test.
        with api._lifecycle_lock, api._callbacks.lock, api._mutation_guards.lock:
            api._callbacks.dispatching = api._mutation_guards.dispatching = True
            held.set()
            release.wait(15)
            api._callbacks.dispatching = api._mutation_guards.dispatching = False
    parent_thread = threading.Thread(target=holder, daemon=True)
    parent_thread.start()
    assert held.wait(5)
    process = ctx.Process(target=child)
    try:
        process.start()
        writer.close()
        assert reader.poll(10), "fork child blocked on inherited lifecycle lock"
        assert reader.recv() == [("loaded", False), ("guarded", "start_weight_update", "draft")]
        process.join(5)
        assert process.exitcode == 0
    finally:
        release.set()
        parent_thread.join(5)
        if process.pid is not None:
            if process.is_alive(): process.terminate()
            process.join(5)
        reader.close()
    assert not parent_thread.is_alive()
    assert events == []
    api.register_model_lifecycle_hooks(**values)
