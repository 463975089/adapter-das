"""Real v0.32/v0.33 Worker/Wrapper/executor/runner APIs, CPU provider doubles.

No copied method bodies, GPU initialization, checkpoints, or engine compilation.
VLLM_TARGET_ROOT optionally pins installed method bytes to the target checkout.
"""

from contextlib import nullcontext
import inspect
from pathlib import Path
import os
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm")
from vllm import envs
from vllm.config import DeviceConfig, SchedulerConfig, VllmConfig, get_current_vllm_config
from vllm.config.compilation import CUDAGraphMode, CompilationMode
from vllm.config.utils import hash_factors
from vllm.compilation.caching import aot_compile_hash_factors
from vllm.platforms.rocm import RocmPlatform
from vllm.v1.worker import gpu_worker, worker_base
from vllm.v1.worker import gpu_model_runner as v1
from vllm.v1.worker.gpu import model_runner as v2
from vllm.v1.worker.gpu.spec_decode import speculator as spec
from vllm.v1.executor import uniproc_executor as uni, multiproc_executor as mp

from hcu_adapter import model_load as api
from hcu_adapter.bridges import vllm_model_load as host
from hcu_adapter.bridges import vllm as bridge
from hcu_adapter.bridges.vllm_worker import Worker
from hcu_adapter.extensions import ExtensionContext, ExtensionSpec, activate_extensions


@pytest.fixture(autouse=True)
def isolation(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.setattr(api, "_callbacks", api._Callbacks())
    monkeypatch.delenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", raising=False)
    yield
    envs.disable_envs_cache()


def config(backend="uni"):
    cfg = VllmConfig(device_config=DeviceConfig("cpu"))
    cfg.parallel_config.distributed_executor_backend = backend
    cfg.parallel_config.worker_cls = "auto"
    cfg.scheduler_config = SchedulerConfig(max_model_len=16, is_encoder_decoder=False, async_scheduling=False)
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    cfg.compilation_config.mode = CompilationMode.NONE
    bridge.HcuPlatform.check_and_update_config(cfg)
    return cfg


def test_installed_target_methods_match_checkout():
    root = os.environ.get("VLLM_TARGET_ROOT")
    if not root:
        pytest.skip("set VLLM_TARGET_ROOT for exact source provenance")
    for module in [gpu_worker, worker_base, v1, v2, spec, uni, mp]:
        installed = Path(inspect.getfile(module))
        relative = Path(*module.__name__.split(".")).with_suffix(".py")
        assert installed.read_bytes() == (Path(root) / relative).read_bytes(), module.__name__
    assert Worker.__bases__ == (gpu_worker.Worker,)
    assert inspect.signature(Worker.load_model) == inspect.signature(gpu_worker.Worker.load_model)


@pytest.mark.parametrize("choice", ["auto", host.OFFICIAL_MODEL_LOAD_WORKER, host.POST_MODEL_LOAD_WORKER, "foreign.Worker"])
@pytest.mark.parametrize("backend", ["uni", "mp", "ray", "external_launcher", "contract.Executor", None])
def test_platform_super_first_known_only_and_idempotent(monkeypatch, choice, backend):
    cfg = config(backend)
    cfg.parallel_config.worker_cls = choice
    seen = []
    original = RocmPlatform.check_and_update_config.__func__
    def official(cls, value):
        seen.append(value.parallel_config.worker_cls)
        return original(cls, value)
    monkeypatch.setattr(RocmPlatform, "check_and_update_config", classmethod(official))
    bridge.HcuPlatform.check_and_update_config(cfg)
    if choice in ("auto", host.OFFICIAL_MODEL_LOAD_WORKER):
        expected = host.POST_MODEL_LOAD_WORKER if backend in ("uni", "mp") else host.OFFICIAL_MODEL_LOAD_WORKER
    else:
        expected = choice
    assert seen == [choice]
    assert cfg.parallel_config.worker_cls == expected
    bridge.HcuPlatform.check_and_update_config(cfg)
    assert seen == [choice, expected]
    assert cfg.parallel_config.worker_cls == expected


def test_super_failure_does_not_rewrite_worker(monkeypatch):
    cfg = config()
    cfg.parallel_config.worker_cls = host.OFFICIAL_MODEL_LOAD_WORKER
    def fail(cls, cfg):
        raise ValueError("official config rejected")
    monkeypatch.setattr(RocmPlatform, "check_and_update_config", classmethod(fail))
    with pytest.raises(ValueError, match="official config"):
        bridge.HcuPlatform.check_and_update_config(cfg)
    assert cfg.parallel_config.worker_cls == host.OFFICIAL_MODEL_LOAD_WORKER


@pytest.mark.parametrize("backend,worker,elastic", [
    ("uni", "foreign.Worker", False), ("mp", "auto", False),
    ("ray", host.POST_MODEL_LOAD_WORKER, False), (None, host.POST_MODEL_LOAD_WORKER, False),
    ("mp", host.POST_MODEL_LOAD_WORKER, True),
])
def test_unavailable_requires_fail_closed(backend, worker, elastic):
    cfg = config()
    cfg.parallel_config.worker_cls = worker
    cfg.parallel_config.distributed_executor_backend = backend
    cfg.parallel_config.enable_elastic_ep = elastic
    assert host.post_model_load_unavailable_reason(cfg)
    with pytest.raises(api.PostModelLoadError):
        host.require_post_model_load(cfg)


def test_elastic_launch_env_is_not_silently_covered(monkeypatch):
    cfg = config()
    monkeypatch.setenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", "1")
    with pytest.raises(api.PostModelLoadError, match="scale-up"):
        host.require_post_model_load(cfg)


def make_runner(monkeypatch, cfg, version, events, fail_at=None):
    """Execute the complete real runner load_model with CPU provider doubles."""
    module = v1 if version == 1 else v2
    runner = object.__new__(module.GPUModelRunner)
    model = NS()  # No torch model or checkpoint is created/loaded.
    def load_target(**kwargs):
        assert kwargs["vllm_config"] is cfg
        events.append("target-pack")
        if fail_at == "target":
            raise ValueError("target provider failed")
        return model
    monkeypatch.setattr(module, "get_model_loader", lambda _: NS(load_model=load_target))
    monkeypatch.setattr(module, "DeviceMemoryProfiler", lambda: _Memory())
    monkeypatch.setattr(module, "get_offloader", lambda: NS(post_init=lambda: events.append("runner-end")))
    runner.vllm_config = cfg
    runner.load_config = cfg.load_config
    runner.parallel_config = cfg.parallel_config
    runner.compilation_config = cfg.compilation_config
    runner.model_config = NS(model="CPU-provider-contract", multimodal_config=None, logits_processors=())
    runner.lora_config = None
    runner.jit_warmup_registry = NS(activate=nullcontext)
    runner.device = "cpu"
    def draft_load(target):
        assert target is model
        events.append("draft-pack")
        if fail_at == "draft":
            raise ValueError("draft provider failed")
    if version == 1:
        runner.drafter = NS(load_model=draft_load)
        runner._setup_eagle3_aux_hidden_state_outputs = lambda: None
        monkeypatch.setattr(module, "get_mixture_of_experts_model", lambda _: None)
        monkeypatch.setattr(module, "supports_multimodal_pruning", lambda _: False)
        monkeypatch.setattr(module, "is_breakable_cudagraph_enabled", lambda: False)
    else:
        class Draft(spec.DraftModelSpeculator):
            def load_draft_model(self, target_model, target_attn_layer_names):
                draft_load(target_model)
                # v0.33's real draft update traverses modules to restore the
                # source LM head. Use an empty CPU module rather than skipping
                # that official path or replacing its traversal helper.
                from torch.nn import Module
                return Module()
            def init_cudagraph_manager(self, *args): pass
            def capture(self): pass
            def propose(self, *args, **kwargs): pass
        drafter = object.__new__(Draft)
        drafter.vllm_config = cfg
        drafter.use_acceptance_estimator = False
        drafter._validate_local_argmax_reduction = lambda: None
        monkeypatch.setattr(spec, "get_layers_from_vllm_config", lambda *args: {})
        # Real DraftModelSpeculator.load_model reads target MM capability.
        # Avoid creating a real ModelConfig (which would access model metadata).
        monkeypatch.setattr(spec, "supports_multimodal_embeddings", lambda _: False)
        draft_cfg = NS(model_config=NS(supports_multimodal_inputs=False))
        drafter.vllm_config = draft_cfg
        runner.speculator = drafter
        runner.speculative_config = NS()
        runner.eplb = NS(prepare_load=lambda: None, maybe_register_speculator=lambda *args: False,
                         maybe_register_model=lambda *args: False, maybe_start_async_loop=lambda *args: None)
        runner.use_aux_hidden_state_outputs = False
        runner._draft_workspace_lane = 1
        runner.num_speculative_steps = 1
        runner.encoder_cache = None
        runner.req_states = None
        runner.is_pooling_model = False
        runner.is_last_pp_rank = False
        runner.is_first_pp_rank = True
        monkeypatch.setattr(module, "use_workspace_lane", lambda _: nullcontext())
        monkeypatch.setattr(module, "init_model_state", lambda *args: NS(num_new_sampled_tokens_per_step=1))
        monkeypatch.setattr(module, "build_custom_logits_processors", lambda *args: None)
    return runner


class _Memory:
    consumed_memory = 0
    def __enter__(self): return self
    def __exit__(self, *args): return False


def prepare_wrapper(monkeypatch, cfg, runner, events, callback, *, real_plugins=False):
    """Keep official WorkerWrapper.init_worker + Worker.load_model intact."""
    import vllm.plugins
    def activate(context):
        events.append("activate")
        api.register_post_model_load("test-owner", callback)
        api.register_post_model_load("test-owner", callback)
    point = NS(name="contract", load=lambda: lambda: ExtensionSpec("contract", 1, 0, frozenset(), activate))
    def initialize_extensions():
        return activate_extensions(ExtensionContext("test", "0.32.0"), [point])
    if real_plugins:
        monkeypatch.setattr(vllm.plugins, "plugins_loaded", False)
        monkeypatch.setattr(vllm.plugins, "load_plugins_by_group", lambda group: {"contract": initialize_extensions})
    else:
        monkeypatch.setattr(vllm.plugins, "load_general_plugins", initialize_extensions)
    monkeypatch.setattr(worker_base, "worker_receiver_cache_from_config", lambda *args: None)
    def initialize(self, **kwargs):
        self.vllm_config = kwargs["vllm_config"]
        self.model_runner = runner
    monkeypatch.setattr(gpu_worker.Worker, "__init__", initialize)
    monkeypatch.setattr(gpu_worker.Worker, "init_device", lambda self: events.append("init-device"))
    monkeypatch.setattr(gpu_worker.Worker, "_maybe_get_memory_pool_context", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(gpu_worker.Worker, "_scoped_allocator_max_split", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(gpu_worker, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(gpu_worker, "set_torch_threads_for_runtime", lambda: events.append("official-worker-end"))


@pytest.mark.parametrize("backend", ["ray", "external_launcher"])
@pytest.mark.parametrize("version", [1, 2])
def test_default_ct_unsupported_executor_keeps_official_load_without_callbacks(monkeypatch, backend, version):
    cfg = config(backend)
    cfg.model_config = NS(quantization="compressed-tensors")
    events = []
    runner = make_runner(monkeypatch, cfg, version, events)

    def must_not_dispatch(context):
        pytest.fail("default CT with an unsupported executor must keep official loading")

    prepare_wrapper(monkeypatch, cfg, runner, events, must_not_dispatch)
    wrapper = worker_base.WorkerWrapperBase(0)
    wrapper.init_worker([{"vllm_config": cfg}])
    assert cfg.parallel_config.worker_cls == host.OFFICIAL_MODEL_LOAD_WORKER
    assert type(wrapper.worker) is gpu_worker.Worker
    assert api._callbacks.callbacks  # A registered policy must not restrict CT.
    wrapper.load_model()
    assert events == ["activate", "target-pack", "draft-pack", "runner-end", "official-worker-end"]
    assert not envs._is_envs_cache_enabled()


@pytest.mark.parametrize("backend", ["ray", "external_launcher", "contract.Executor"])
def test_explicit_quant_policy_requires_covered_executor_before_loading(monkeypatch, backend):
    cfg = config(backend)
    cfg.model_config = NS(quantization="slimquant_marlin")
    events = []
    runner = make_runner(monkeypatch, cfg, 2, events)
    prepare_wrapper(monkeypatch, cfg, runner, events, lambda context: pytest.fail("no dispatch"))
    wrapper = worker_base.WorkerWrapperBase(0)
    wrapper.init_worker([{"vllm_config": cfg}])
    assert type(wrapper.worker) is gpu_worker.Worker
    # This generic Adapter barrier is required by the Plugin selector. It must
    # inspect the actual resolved config, before a caller starts loading/packing.
    with pytest.raises(api.PostModelLoadError, match="does not provide the Adapter load seam"):
        host.require_post_model_load(cfg)
    assert events == ["activate"]


@pytest.mark.parametrize("backend", ["uni", "mp"])
@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("fail_at", [None, "target", "draft"])
def test_real_startup_load_and_freeze_before_cache(monkeypatch, backend, version, fail_at):
    cfg = config(backend)
    events = []
    runner = make_runner(monkeypatch, cfg, version, events, fail_at)
    state = {"selected": [], "frozen": None}
    factor = "HCU_ADAPTER_TEST_POST_LOAD_FACTOR"
    # Stand-in extension policy, not a replacement for the real Plugin policy.
    def policy():
        if state["frozen"] is None:
            return "pending"
        return state["frozen"]
    monkeypatch.setitem(envs.environment_variables, factor, policy)
    initial_config_hash = cfg.compute_hash()  # Intentionally before model load.
    pending_aot = aot_compile_hash_factors(cfg)
    def freeze(context):
        assert get_current_vllm_config() is cfg
        assert context.vllm_config is cfg and context.worker.model_runner is runner
        assert events[-1] == "official-worker-end"
        assert "target-pack" in events and "draft-pack" in events
        assert not envs._is_envs_cache_enabled()
        assert context.load_dummy_weights is False
        state["frozen"] = f"v{version}:target+draft"
        events.append("freeze")
    prepare_wrapper(monkeypatch, cfg, runner, events, freeze)
    if backend == "uni":
        executor = object.__new__(uni.UniProcExecutor)
        executor.vllm_config = cfg
        executor._distributed_args = lambda: ("unused", 0, 0)
        monkeypatch.setattr(uni, "set_worker_net_device", lambda *args: None)
        invoke = executor._init_executor
    else:
        # Run the actual MP child startup method without OS/GPU workers or MQs.
        monkeypatch.setattr(mp.WorkerProc, "setup_proc_title_and_log_prefix", lambda *args, **kwargs: None)
        monkeypatch.setattr(mp.WorkerProc, "_init_message_queues", lambda *args: None)
        invoke = lambda: mp.WorkerProc(cfg, 0, 0, "unused", None, None, True)
    if fail_at:
        with pytest.raises(ValueError, match=f"{fail_at} provider failed"):
            invoke()
        assert "freeze" not in events
        assert not envs._is_envs_cache_enabled()
        return
    invoke()
    assert events == ["activate", "init-device", "target-pack", "draft-pack", "runner-end", "official-worker-end", "freeze"]
    # First ordinary/AOT env-factor reads see the frozen policy even if the
    # config's memoized hash was already read before loading. Real env cache API.
    assert cfg.compute_hash() == initial_config_hash
    ordinary = hash_factors(envs.compile_factors())
    aot = aot_compile_hash_factors(cfg)
    assert aot[0] == ordinary and aot[0] != pending_aot[0]
    envs.enable_envs_cache()
    assert getattr(envs, factor) == state["frozen"]
    assert aot_compile_hash_factors(cfg) == aot
    assert hash_factors(envs.compile_factors()) == ordinary


def test_worker_extension_cannot_override_load_model(monkeypatch):
    cfg = config()
    cfg.parallel_config.worker_extension_cls = "contract.BadExtension"
    class BadExtension:
        def load_model(self): pass
    original = worker_base.resolve_obj_by_qualname
    monkeypatch.setattr(worker_base, "resolve_obj_by_qualname", lambda name: BadExtension if name == "contract.BadExtension" else original(name))
    import vllm.plugins
    monkeypatch.setattr(vllm.plugins, "load_general_plugins", lambda: None)
    wrapper = worker_base.WorkerWrapperBase(0)
    with pytest.raises(AssertionError, match="conflicts"):
        wrapper.init_worker([{"vllm_config": cfg}])


def test_callback_failure_fails_real_worker_load(monkeypatch):
    cfg = config()
    events = []
    runner = make_runner(monkeypatch, cfg, 2, events)
    def fail(context):
        raise RuntimeError("policy refused to freeze")
    prepare_wrapper(monkeypatch, cfg, runner, events, fail)
    wrapper = worker_base.WorkerWrapperBase(0)
    wrapper.init_worker([{"vllm_config": cfg}])
    with pytest.raises(api.PostModelLoadError, match="test-owner.*policy refused"):
        wrapper.load_model()
    assert not envs._is_envs_cache_enabled()


def test_late_env_cache_rejected_at_real_worker_boundary(monkeypatch):
    cfg = config()
    events = []
    runner = make_runner(monkeypatch, cfg, 1, events)
    prepare_wrapper(monkeypatch, cfg, runner, events, lambda context: events.append("wrong"))
    wrapper = worker_base.WorkerWrapperBase(0)
    wrapper.init_worker([{"vllm_config": cfg}])
    envs.enable_envs_cache()
    with pytest.raises(api.PostModelLoadError, match="after env cache"):
        wrapper.load_model()
    assert "wrong" not in events
    assert "after env cache" in host.post_model_load_unavailable_reason(cfg)


def test_empty_vllm_registry_does_not_impose_a_late_cache_or_worker_gate():
    envs.enable_envs_cache()
    # Preserve the old no-owner short circuit, including opaque worker objects.
    host.dispatch_post_model_load(object(), load_dummy_weights=False)


def test_public_facade_shares_the_neutral_registry_contract():
    for name in ("PostModelLoadContext", "ModelMutationContext", "PostModelLoadError",
                 "ModelMutationError", "preflight_model_lifecycle_hooks",
                 "register_model_lifecycle_hooks", "register_post_model_load",
                 "register_model_mutation_guard"):
        assert getattr(host, name) is getattr(api, name)
    for name in ("require_post_model_load", "require_model_mutation_guard"):
        inspect.signature(getattr(host, name)).bind(vllm_config=object())


@pytest.mark.parametrize("version", [1, 2])
def test_dummy_flag_forwarded_by_real_worker(monkeypatch, version):
    cfg = config()
    events = []
    runner = make_runner(monkeypatch, cfg, version, events)
    seen = []
    prepare_wrapper(monkeypatch, cfg, runner, events, lambda context: seen.append(context.load_dummy_weights))
    wrapper = worker_base.WorkerWrapperBase(0)
    wrapper.init_worker([{"vllm_config": cfg}])
    wrapper.load_model(load_dummy_weights=True)
    assert cfg.load_config.load_format == "dummy"
    assert seen == [True]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires actual POSIX fork")
def test_real_plugin_once_flag_fork_keeps_callback_for_worker(monkeypatch):
    import multiprocessing
    import vllm.plugins
    cfg = config("mp")
    events = []
    runner = make_runner(monkeypatch, cfg, 2, events)
    def freeze(context):
        assert events[-1] == "official-worker-end"
        events.append("freeze")
    prepare_wrapper(monkeypatch, cfg, runner, events, freeze, real_plugins=True)
    vllm.plugins.load_general_plugins()
    assert vllm.plugins.plugins_loaded and events == ["activate"]
    ctx = multiprocessing.get_context("fork")
    reader, writer = ctx.Pipe(duplex=False)
    def child():
        try:
            # Official init_worker calls load_general_plugins, which must skip
            # activation because the parent's plugins_loaded=True was inherited.
            wrapper = worker_base.WorkerWrapperBase(0)
            wrapper.init_worker([{"vllm_config": cfg}])
            wrapper.load_model()
            writer.send(events)
        finally:
            writer.close()
    process = ctx.Process(target=child)
    process.start()
    writer.close()
    try:
        assert reader.poll(20), "forked real wrapper did not finish"
        assert reader.recv() == ["activate", "target-pack", "draft-pack", "runner-end", "official-worker-end", "freeze"]
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)
        reader.close()
    assert events == ["activate"]
