"""Actual Worker mutation entry points; only loaders/transfer providers are doubled."""

import inspect
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from hcu_adapter import model_load as api

pytest.importorskip("vllm")
from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.v1.worker import gpu_worker
from vllm.v1.worker import gpu_model_runner as v1
from vllm.v1.worker.gpu import model_runner as v2
from hcu_adapter.bridges.vllm_worker import Worker
from test_vllm_post_model_load import config


@pytest.fixture(autouse=True)
def isolation(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setattr(api, "_mutation_guards", api._Callbacks())
    monkeypatch.delenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", raising=False)
    yield
    envs.disable_envs_cache()


def worker(monkeypatch, version, events):
    cfg = config()
    result = object.__new__(Worker)
    result.vllm_config = cfg
    runner_type = v1 if version == 1 else v2
    runner = object.__new__(runner_type.GPUModelRunner)
    runner.vllm_config = cfg
    runner.model = object()
    runner.reset_lora_state = lambda: events.append("reset-lora")
    runner.reset_encoder_cache = lambda: events.append("reset-encoder")
    runner.reset_mm_cache = lambda: events.append("reset-mm")

    def load(value, model, weights_iterator=None, weights_path=None, is_checkpoint_format=True):
        assert get_current_vllm_config() is cfg
        assert value is cfg and model is runner.model
        cfg.model_config = NS(model=weights_path)  # Simulated provider mutation.
        events.append(("reload", weights_iterator, weights_path, is_checkpoint_format))

    monkeypatch.setattr(runner_type, "reload_weights", load)
    result.model_runner = runner
    result.weight_transfer_engine = NS(
        supports_draft_weight_update=True,
        start_weight_update=lambda: events.append("engine-start"),
        reset_weight_update_target=lambda: events.append("engine-reset"),
    )
    result._weight_update_active = False
    result._weight_update_is_draft = None
    result._set_draft_weight_update_target = lambda: events.append("draft-retarget")
    return result


def invoke(value, operation):
    if operation == "reload":
        return value.reload_weights(iter(()), "/cpu-new-model", False)
    if operation == "target":
        return value.start_weight_update()
    return value.start_draft_weight_update()


def test_target_mutation_method_provenance_and_signatures():
    root = Path(os.environ["VLLM_TARGET_ROOT"])
    for module in (gpu_worker, v1, v2):
        assert Path(inspect.getfile(module)).read_bytes() == (
            root / Path(*module.__name__.split(".")).with_suffix(".py")
        ).read_bytes()
    for name in ("reload_weights", "_start_weight_update"):
        assert inspect.signature(getattr(Worker, name)) == inspect.signature(getattr(gpu_worker.Worker, name))
    assert Worker.start_weight_update is gpu_worker.Worker.start_weight_update
    assert Worker.start_draft_weight_update is gpu_worker.Worker.start_draft_weight_update


@pytest.mark.parametrize("backend", ["ray", "external_launcher", "contract.Executor", None])
def test_mutation_guard_query_rejects_uncovered_resolved_workers(backend):
    cfg = config(backend)
    assert cfg.parallel_config.worker_cls == api.OFFICIAL_MODEL_LOAD_WORKER
    envs.enable_envs_cache()
    with pytest.raises(api.ModelMutationError, match="does not provide the Adapter load seam"):
        api.require_model_mutation_guard(cfg)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("operation", ["reload", "target", "draft"])
@pytest.mark.parametrize("refuse", [False, True])
def test_real_worker_guards_before_mutation_or_delegates(monkeypatch, version, operation, refuse):
    events = []
    value = worker(monkeypatch, version, events)
    original_model_config = value.vllm_config.model_config
    def guard(context):
        assert context.worker is value and context.vllm_config is value.vllm_config
        assert get_current_vllm_config() is value.vllm_config
        assert context.role == ("draft" if operation == "draft" else "target")
        assert context.operation == ("reload_weights" if operation == "reload" else "start_weight_update")
        assert events == []
        events.append("guard")
        if refuse:
            raise ValueError("owned model")
    api.register_model_mutation_guard("cpu-owner", guard)
    envs.enable_envs_cache()  # Serving-time guard must not reuse postload's gate.
    api.require_model_mutation_guard(value.vllm_config)
    if refuse:
        with pytest.raises(api.ModelMutationError, match="cpu-owner.*owned model") as caught:
            invoke(value, operation)
        assert isinstance(caught.value.__cause__, ValueError)
        assert events == ["guard"]
        assert value.vllm_config.model_config is original_model_config
        assert value._weight_update_active is False
        assert value._weight_update_is_draft is None
    else:
        assert invoke(value, operation) is None
        if operation == "reload":
            assert events[1][0] == "reload" and events[1][2:] == ("/cpu-new-model", False)
            assert events[2:] == ["reset-lora", "reset-encoder", "reset-mm"]
        else:
            assert events == (["guard", "draft-retarget", "engine-start"] if operation == "draft" else ["guard", "engine-start"])
            assert value._weight_update_active is True
            assert value._weight_update_is_draft is (operation == "draft")


def test_guard_owners_order_conflict_and_non_none_result():
    calls = []
    first = lambda context: calls.append("a")
    api.register_model_mutation_guard("a", first)
    api.register_model_mutation_guard("a", first)
    api.register_model_mutation_guard("b", lambda context: False)
    with pytest.raises(api.ModelMutationError, match="already registered"):
        api.register_model_mutation_guard("a", lambda context: None)
    with pytest.raises(api.ModelMutationError, match="owner 'b'.*return None"):
        api.dispatch_model_mutation(object(), operation="reload_weights", role="target")
    assert calls == ["a"]


@pytest.mark.parametrize("recursive", [False, True])
def test_mutation_dispatch_rejects_registration_and_recursion(recursive):
    def callback(context):
        if recursive:
            api.dispatch_model_mutation(context.worker, operation="reload_weights", role="target")
        else:
            api.register_model_mutation_guard("late", lambda value: None)
    api.register_model_mutation_guard("owner", callback)
    with pytest.raises(api.ModelMutationError, match="recursive|cannot register during"):
        api.dispatch_model_mutation(object(), operation="reload_weights", role="target")


@pytest.mark.parametrize("async_guard", [False, True])
def test_mutation_guard_requires_synchronous_callback(async_guard):
    async def asynchronous(context): pass
    def generator(context): yield None
    with pytest.raises(TypeError, match="synchronous"):
        api.register_model_mutation_guard("owner", asynchronous if async_guard else generator)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_mutation_guards_keep_declarations_and_reset_locks_after_fork():
    import multiprocessing
    api.register_model_mutation_guard("owner", lambda context: None)
    ctx = multiprocessing.get_context("fork")
    reader, writer = ctx.Pipe(False)
    def child():
        try:
            api.dispatch_model_mutation(object(), operation="reload_weights", role="target")
            writer.send(list(api._mutation_guards.callbacks))
        finally:
            writer.close()
    with api._mutation_guards.lock:
        api._mutation_guards.dispatching = True
        process = ctx.Process(target=child)
        process.start()
        api._mutation_guards.dispatching = False
    writer.close()
    try:
        assert reader.poll(15), "inherited mutation lock blocked child"
        assert reader.recv() == ["owner"]
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive(): process.terminate()
        process.join(5)
        reader.close()
