"""New public pair at actual Worker/runner boundaries; CPU provider doubles."""

from types import SimpleNamespace as NS

import pytest

from hcu_adapter import model_load as api

pytest.importorskip("vllm")
from vllm import envs
from vllm.v1.worker import gpu_model_runner as v1, worker_base
from vllm.v1.worker.gpu import model_runner as v2
import vllm.plugins
import test_vllm_post_model_load as helpers


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    import torch
    assert not torch.cuda.is_initialized()
    envs.disable_envs_cache()
    monkeypatch.setattr(api, "_callbacks", api._Callbacks())
    monkeypatch.setattr(api, "_mutation_guards", api._Callbacks())
    monkeypatch.delenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", raising=False)
    def no_gpu(*args, **kwargs):
        pytest.fail("CPU lifecycle pair contract must not query/initialize GPU")
    for name in ("current_device", "get_device_properties", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, no_gpu)
    yield
    envs.disable_envs_cache()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("operation", ["reload", "target", "draft"])
@pytest.mark.parametrize("refuse", [False, True])
def test_public_pair_real_load_then_real_mutation_entry(monkeypatch, version, operation, refuse):
    cfg = helpers.config()
    events, state = [], {"frozen": False}
    runner = helpers.make_runner(monkeypatch, cfg, version, events)
    def loaded(context):
        assert context.worker.model_runner is runner
        assert events[-1] == "official-worker-end"
        assert "target-pack" in events and "draft-pack" in events
        assert not envs._is_envs_cache_enabled()
        state["frozen"] = True
        events.append("pair-freeze")
    def guard(context):
        assert state["frozen"] and envs._is_envs_cache_enabled()
        assert context.operation == ("reload_weights" if operation == "reload" else "start_weight_update")
        assert context.role == ("draft" if operation == "draft" else "target")
        if refuse:
            raise ValueError("CPU owner refuses before mutation")
        events.append("pair-guard-pass")
    values = dict(post_model_load_owner="cpu.freeze", post_model_load_callback=loaded,
                  model_mutation_owner="cpu.guard", model_mutation_guard=guard)
    helpers.prepare_wrapper(monkeypatch, cfg, runner, events, loaded)
    def activate():
        api.preflight_model_lifecycle_hooks(**values)
        api.register_model_lifecycle_hooks(**values)
    monkeypatch.setattr(vllm.plugins, "load_general_plugins", activate)
    wrapper = worker_base.WorkerWrapperBase(0)
    wrapper.init_worker([{"vllm_config": cfg}])
    wrapper.load_model()
    assert state["frozen"] and events[-1] == "pair-freeze"
    worker = wrapper.worker
    events.clear()
    module = v1 if version == 1 else v2
    monkeypatch.setattr(module, "reload_weights", lambda *args, **kwargs: events.append("CPU-reload-provider"))
    runner.reset_lora_state = lambda: events.append("reset-lora")
    runner.reset_encoder_cache = lambda: events.append("reset-encoder")
    runner.reset_mm_cache = lambda: events.append("reset-mm")
    worker.weight_transfer_engine = NS(
        supports_draft_weight_update=True,
        start_weight_update=lambda: events.append("CPU-transfer-start"),
        reset_weight_update_target=lambda: events.append("CPU-transfer-reset"),
    )
    worker._set_draft_weight_update_target = lambda: events.append("CPU-draft-retarget")
    worker._weight_update_active, worker._weight_update_is_draft = False, None
    envs.enable_envs_cache()
    def invoke():
        if operation == "reload":
            worker.reload_weights(weights_iterator=iter(()), is_checkpoint_format=False)
        elif operation == "target":
            worker.start_weight_update()
        else:
            worker.start_draft_weight_update()
    if refuse:
        with pytest.raises(api.ModelMutationError, match="cpu.guard.*CPU owner refuses"):
            invoke()
        assert events == [] and not worker._weight_update_active
    else:
        invoke()
        assert events == ["pair-guard-pass"] + (
            ["CPU-reload-provider", "reset-lora", "reset-encoder", "reset-mm"] if operation == "reload" else
            ["CPU-draft-retarget", "CPU-transfer-start"] if operation == "draft" else ["CPU-transfer-start"])
