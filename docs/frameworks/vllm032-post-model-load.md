# vLLM 0.32 post-model-load extension seam

This is an additive Adapter interface, not an optimization activation. The
existing Extension API version/entry points do not change. No Plugin package,
runner, loader, worker body, platform identity, or vLLM env getter is replaced.

## Registration and use

An extension registers a module-level callback, once its own activation checks
have succeeded, from its existing `ExtensionSpec.activate` function:

```python
from hcu_adapter.model_load import (
    register_post_model_load,
    require_post_model_load,
    post_model_load_unavailable_reason,
)

def after_model_load(context):
    # context.worker, context.vllm_config, context.load_dummy_weights
    # Freeze this owner's actual target AND draft selections/packing here.
    # The owner implements freeze_policy; Adapter knows no optimization policy.
    freeze_policy(context.vllm_config)

def activate(extension_context):
    register_post_model_load("my-extension.compile-policy", after_model_load)

def select_optimization(vllm_config):
    # REQUIRED before committing to a route which needs this freeze boundary.
    require_post_model_load(vllm_config)
    # Now validate/select the extension's own optimization.
```

`post_model_load_unavailable_reason(config)` returns `None` only for a resolved,
supported configuration; otherwise it returns a diagnostic string.
`require_post_model_load(config)` raises `PostModelLoadError` with that reason.
Callers must not catch that error and continue selecting a freeze-dependent
route. The query describes the load seam, not the presence of any particular
extension, callback, selected method, or working GPU kernel.

Only the exact `hcu_adapter.bridges.vllm_worker.Worker` path with resolved `uni`
or `mp`, no elastic EP/scale-up, and env caching still disabled is supported.
Unresolved `auto`, `None` executor, explicit foreign workers/subclasses, Ray,
custom executors, elastic launch and late env-cache states fail the query.

`HcuPlatform.check_and_update_config` calls the official ROCm method first and,
only for resolved `uni`/`mp`, replaces its known
`vllm.v1.worker.gpu_worker.Worker` result. Ray, external launchers, custom and
unresolved executors retain the default official worker, so registering a policy callback
does not restrict default CT loading on those executors. A freeze-dependent
Plugin selector must still call `require_post_model_load` before loading/packing;
the unsupported executor's actual resolved config fails that query. The same
explicit official worker path is supported for `uni`/`mp`. Repeated config checks
preserve the Adapter path. Foreign explicit worker paths remain untouched, including
their own extension configuration; they must use a different supported route
or an independently implemented seam. `worker_extension_cls` cannot override
`load_model`: the official wrapper rejects that name conflict.

## Load boundary and ownership

The Adapter Worker inherits the official Worker, calls `super().load_model`
with the original keyword argument, then dispatches in the current VllmConfig
context. The official Worker loads both target and synchronous draft before
returning, irrespective of its V1/V2 runner selection:

- V1 `vllm/v1/worker/gpu_model_runner.py::GPUModelRunner.load_model` loads the
  target and then `drafter.load_model(self.model)`.
- V2 `vllm/v1/worker/gpu/model_runner.py::GPUModelRunner.load_model` loads the
  target and then `DraftModelSpeculator.load_model(self.model)`; the latter
  synchronously calls `load_draft_model`.
- Both runners return through `vllm/v1/worker/gpu_worker.py::Worker.load_model`.
  The callback runs after its remaining official post-load work succeeds.
- `UniProcExecutor._init_executor` returns after worker loading, before the
  EngineCore profiles KV caches/executes its first forward or enables env cache.
  `WorkerProc.__init__` loads before `enable_envs_cache` in the MP child.

One owner has one callable. Re-registering the same callable is a no-op; another
callable for the same owner raises. Use a stable module-level function, not a
fresh lambda/bound-method object at each activation. Owners dispatch in name
order, once per successful Worker `load_model` invocation, not once per process.
There is no unregister/replacement, implicit retry or exception swallowing.
An official load failure runs no callback. A callback failure stops subsequent
callbacks, raises with its owner and original cause, and fails startup.
Registration during dispatch and recursive dispatch are rejected.

Spawn activates extensions through the official WorkerWrapper plugin lifecycle.
Fork **retains callback declarations**, rebuilding only the Adapter's lock/PID
and dispatch state: the target's `vllm.plugins.plugins_loaded=True` can be
inherited, so its wrapper may skip activation in the child. The Adapter neither
resets that official flag nor copies plugin discovery. Each callback owner is
responsible for its own captured state, locks, selection records and process
semantics. In particular, this seam does not make a policy frozen in a parent
safe for a new child model. Runtime weight reloads, asynchronous loading, elastic
loading and arbitrary worker implementations are outside this startup contract.

Runtime weight-update rejection is an open integration issue. The startup seam
does not guard official `Worker.reload_weights`, `start_weight_update` or
`start_draft_weight_update`; these can bypass the successful-load callback.
The Plugin's frozen-selection guard does not by itself reject those entry points.
Do not infer hot-update support or an enforced prohibition from startup tests.

## Compile policy and cache semantics

Register effective compile factors before model construction, select and pack
all target/draft weights, then freeze immutable effective selections in this
callback, before the first ordinary/AOT forward and env cache prewarming.
Adapter does not inspect or freeze any extension's state for it. The callback
does not clear `VllmConfig.compute_hash`, rewrite global env getters, or replace
vLLM's cache. A callback cannot repair an artifact already persisted too early.

In target a198, ordinary backend hashing freshly calls `envs.compile_factors()`
and `hash_factors`; `aot_compile_hash_factors` also hashes that registry afresh,
separately from the config hash, which itself recomputes on this target.
Registry getters bypass cached
`envs.__getattr__`. A policy's selecting/empty phase must be distinct from frozen
selections and its getter must return the actual effective immutable policy.
The owner must guard against late registration, mutation after freeze, invalid
settings, and inappropriate fork/reload state. Getter exceptions alone do not
fail closed: target `compile_factors` logs and omits failing getters. These are
Plugin integration responsibilities, not evidence that its policy already works.

## Validation and remaining integration gates

`tests/adapter/test_model_load_callbacks.py` checks ownership, idempotency,
failures, framework-light import and real POSIX fork state/lock behavior.
`tests/adapter/test_vllm_post_model_load.py` executes installed official APIs:
WorkerWrapper resolution, uni startup, MP child startup method, both complete
runner load methods, V2 DraftModelSpeculator load, and official Worker load.
Only device contexts, checkpoint providers, peripheral model state/samplers,
thread tuning and IPC setup are replaced with CPU doubles. No model checkpoint,
GPU service, tensor kernel or distributed GPU worker is launched. The MP startup
method test runs in-process; a separate real fork test covers inherited official
plugin-once state and a real child WorkerWrapper/Worker/runner call chain.
Executor contracts additionally check super-first worker resolution across
`uni`, `mp`, Ray, external/custom and unresolved executors. Default CT with Ray
or an external launcher executes the actual official WorkerWrapper/Worker and
V1/V2 runner load methods with CPU providers, while a registered failing callback
remains uncalled. These are worker API contracts, not running Ray clusters or
distributed/GPU model validation. Explicit quant-policy availability is checked
on the actual resolved VllmConfig before worker loading.

The tests assert callback-after-target/draft/official-success and before real
env-cache enablement; they use real `VllmConfig.compute_hash`, ordinary env hash
helpers and AOT hash factors, including a config hash read before loading. They
do not extract/execute copies of upstream method bodies. `VLLM_TARGET_ROOT` pins
installed official module bytes to the selected checkout, including V2.

Run with the exact target Python, explicit Adapter source path (source phase),
`HCU_ADAPTER_HOST=none`, `VLLM_PLUGINS=__disabled__`, all GPU visibility variables
empty, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and TMPDIR/XDG/TRITON/TORCHINDUCTOR/
VLLM caches rooted in `/dev/shm`. Do not modify the running service's venv.

Remaining gates: build/install the integrated wheel in a separate environment;
wire the actual Plugin callback and fail-closed selection query; run real uni/mp
target/draft model startup and first forward; persist and reload engine AOT
artifacts with the actual frozen policy; verify distinct selections cannot reuse
stale artifacts. CPU API/hash tests are not model, numerical, Graph or engine
AOT artifact passes. No quantization capability is added by this change.
