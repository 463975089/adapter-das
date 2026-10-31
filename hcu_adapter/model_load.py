"""Framework-light, process-local callbacks for successful worker model loads.

Extensions register during activation (spawn) or inherit declarations (fork).
Importing this module imports no vLLM and selects no optimization. Availability
queries lazily inspect vLLM's cache/launch state. Callback failures fail the load.
"""

from __future__ import annotations

import os
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock


POST_MODEL_LOAD_WORKER = "hcu_adapter.bridges.vllm_worker.Worker"
OFFICIAL_MODEL_LOAD_WORKER = "vllm.v1.worker.gpu_worker.Worker"


class PostModelLoadError(RuntimeError):
    """A callback conflicts, fails, or lacks a supported load boundary."""


class ModelMutationError(RuntimeError):
    """A mutation guard conflicts, fails, or lacks a supported worker boundary."""


@dataclass(frozen=True)
class ModelMutationContext:
    worker: object
    operation: str
    role: str

    @property
    def vllm_config(self):
        return self.worker.vllm_config


@dataclass(frozen=True)
class PostModelLoadContext:
    worker: object
    load_dummy_weights: bool

    @property
    def vllm_config(self):
        return self.worker.vllm_config


class _Callbacks:
    def __init__(self):
        self.callbacks: dict[str, Callable] = {}
        self.reset()
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self.reset)

    def reset(self):
        # Keep declarations: vLLM's plugins_loaded once flag survives fork, so
        # WorkerWrapper may not reactivate extensions. Only our synchronization
        # and invocation state reset. Owners manage their own captured state.
        self.pid = os.getpid()
        self.lock = RLock()
        self.dispatching = False

    def ensure_process(self):
        if self.pid != os.getpid():
            self.reset()


_lifecycle_pid = os.getpid()
_lifecycle_lock = RLock()


def _reset_lifecycle_lock() -> None:
    global _lifecycle_pid, _lifecycle_lock
    _lifecycle_pid = os.getpid()
    _lifecycle_lock = RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_lifecycle_lock)


_callbacks = _Callbacks()
_mutation_guards = _Callbacks()


def _ensure_lifecycle_process() -> None:
    if _lifecycle_pid != os.getpid():
        _reset_lifecycle_lock()
    _callbacks.ensure_process()
    _mutation_guards.ensure_process()


def _require_synchronous(callback, label: str) -> None:
    implementations = (callback,) if inspect.isroutine(callback) else (callback, callback.__call__)
    if any(check(implementation) for implementation in implementations for check in (
        inspect.iscoroutinefunction, inspect.isasyncgenfunction,
        inspect.isgeneratorfunction,
    )):
        raise TypeError(f"{label} must be synchronous")


def _preflight_lifecycle_hooks(
    post_model_load_owner, post_model_load_callback,
    model_mutation_owner, model_mutation_guard,
) -> bool:
    """Caller holds the lifecycle gate and both registries; no state is written."""
    unchanged = True
    for registry, owner, callback, label, error in (
        (_callbacks, post_model_load_owner, post_model_load_callback,
         "post-model-load", PostModelLoadError),
        (_mutation_guards, model_mutation_owner, model_mutation_guard,
         "model-mutation", ModelMutationError),
    ):
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError(f"{label} owner must be a non-empty string")
        if not callable(callback):
            raise TypeError(f"{label} callback must be callable")
        _require_synchronous(callback, f"{label} callback")
        previous = registry.callbacks.get(owner)
        if previous is not None and previous is not callback:
            raise error(f"{label} owner {owner!r} already registered")
        unchanged = unchanged and previous is callback
    if unchanged:
        return False
    if _callbacks.dispatching:
        raise PostModelLoadError("cannot register lifecycle hooks during post-model-load dispatch")
    if _mutation_guards.dispatching:
        raise ModelMutationError("cannot register lifecycle hooks during model-mutation dispatch")
    return True


def preflight_model_lifecycle_hooks(
    *, post_model_load_owner: str,
    post_model_load_callback: Callable[[PostModelLoadContext], None],
    model_mutation_owner: str,
    model_mutation_guard: Callable[[ModelMutationContext], None],
) -> None:
    """Check the two named hooks without registering or invoking either one.

    This is a current-state query, not a reservation. Atomic registration repeats
    these checks. Both callbacks must be synchronous; identical declarations are
    idempotent, including an entirely unchanged pair during dispatch.
    """
    _ensure_lifecycle_process()
    with _lifecycle_lock, _callbacks.lock, _mutation_guards.lock:
        _preflight_lifecycle_hooks(
            post_model_load_owner, post_model_load_callback,
            model_mutation_owner, model_mutation_guard,
        )


def register_model_lifecycle_hooks(
    *, post_model_load_owner: str,
    post_model_load_callback: Callable[[PostModelLoadContext], None],
    model_mutation_owner: str,
    model_mutation_guard: Callable[[ModelMutationContext], None],
) -> None:
    """Register this post-load/guard pair atomically, restoring both on failure.

    Legacy registration and dispatch share the gate, so other threads cannot
    dispatch a partially registered pair. This only transacts the two Adapter
    registries, not any extension-owned activation or model state.
    """
    _ensure_lifecycle_process()
    with _lifecycle_lock, _callbacks.lock, _mutation_guards.lock:
        if not _preflight_lifecycle_hooks(
            post_model_load_owner, post_model_load_callback,
            model_mutation_owner, model_mutation_guard,
        ):
            return
        before_load = dict(_callbacks.callbacks)
        before_mutation = dict(_mutation_guards.callbacks)
        try:
            register_post_model_load(post_model_load_owner, post_model_load_callback)
            register_model_mutation_guard(model_mutation_owner, model_mutation_guard)
        except BaseException:
            _callbacks.callbacks.clear()
            _callbacks.callbacks.update(before_load)
            _mutation_guards.callbacks.clear()
            _mutation_guards.callbacks.update(before_mutation)
            raise


def register_post_model_load(
    owner: str, callback: Callable[[PostModelLoadContext], None],
) -> None:
    """Register one callback per named owner; same callable is idempotent.

    Another callable cannot replace an owner. Register in ExtensionSpec.activate,
    not while dispatching. Fork preserves declarations, not dispatch state/locks;
    callbacks must keep their own mutable model state process-local.
    """
    if not isinstance(owner, str) or not owner.strip():
        raise ValueError("post-model-load owner must be a non-empty string")
    if not callable(callback):
        raise TypeError("post-model-load callback must be callable")
    _ensure_lifecycle_process()
    with _lifecycle_lock, _callbacks.lock:
        previous = _callbacks.callbacks.get(owner)
        if previous is callback:
            return
        if previous is not None:
            raise PostModelLoadError(f"post-model-load owner {owner!r} already registered")
        if _callbacks.dispatching:
            raise PostModelLoadError("cannot register during post-model-load dispatch")
        _callbacks.callbacks[owner] = callback


def _worker_unavailable_reason(vllm_config) -> str | None:
    """Fail-closed coverage query for a *resolved* VllmConfig.

    Only the Adapter's exact worker path and synchronous uni/mp startup are
    covered. Foreign workers/subclasses and unresolved configuration are not
    inferred to support this seam. This query does not register a callback.
    """
    parallel = vllm_config.parallel_config
    if parallel.worker_cls != POST_MODEL_LOAD_WORKER:
        return f"worker {parallel.worker_cls!r} does not provide the Adapter load seam"
    if parallel.distributed_executor_backend not in ("uni", "mp"):
        return "post-model-load requires the resolved uni or mp executor"
    if parallel.enable_elastic_ep:
        return "elastic model loading is outside the post-model-load contract"
    from vllm import envs

    if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
        return "elastic scale-up bypasses the synchronous worker load boundary"
    return None


def post_model_load_unavailable_reason(vllm_config) -> str | None:
    """Describe resolved worker coverage and the pre-env-cache load boundary."""
    reason = _worker_unavailable_reason(vllm_config)
    if reason is not None:
        return reason
    from vllm import envs

    if envs._is_envs_cache_enabled():
        return "post-model-load boundary is unavailable after env cache enabled"
    return None


def require_post_model_load(vllm_config) -> None:
    """Call before selecting any route that requires a post-load freeze."""
    reason = post_model_load_unavailable_reason(vllm_config)
    if reason is not None:
        raise PostModelLoadError(reason)


def dispatch_post_model_load(worker, *, load_dummy_weights: bool) -> None:
    """Worker-only dispatch after super.load_model; once per successful call.

    Owners run in name order. Exceptions abort loading with the owner's name;
    there is no implicit retry, swallowing, or global once-only load latch.
    """
    _ensure_lifecycle_process()
    with _lifecycle_lock, _callbacks.lock:
        if not _callbacks.callbacks:
            return
        require_post_model_load(worker.vllm_config)
        if _callbacks.dispatching:
            raise PostModelLoadError("recursive post-model-load dispatch")
        _callbacks.dispatching = True
        try:
            context = PostModelLoadContext(worker, load_dummy_weights)
            for owner, callback in sorted(_callbacks.callbacks.items()):
                try:
                    callback(context)
                except Exception as exc:
                    raise PostModelLoadError(
                        f"post-model-load owner {owner!r} failed: {exc}"
                    ) from exc
        finally:
            _callbacks.dispatching = False


def model_mutation_guard_unavailable_reason(vllm_config) -> str | None:
    """Query the resolved uni/mp worker seam; serving env cache is allowed."""
    return _worker_unavailable_reason(vllm_config)


def require_model_mutation_guard(vllm_config) -> None:
    """Call before selecting a route which requires weight-mutation protection."""
    reason = model_mutation_guard_unavailable_reason(vllm_config)
    if reason is not None:
        raise ModelMutationError(reason)


def register_model_mutation_guard(
    owner: str, callback: Callable[[ModelMutationContext], None],
) -> None:
    """Register a named pre-mutation guard; owners decide which models to protect.

    Same-callable registration is idempotent. Fork retains declarations and
    resets synchronization only; owners must preserve or reject inherited state.
    """
    if not isinstance(owner, str) or not owner.strip():
        raise ValueError("model-mutation owner must be a non-empty string")
    if not callable(callback):
        raise TypeError("model-mutation guard must be callable")
    _require_synchronous(callback, "model-mutation guard")
    _ensure_lifecycle_process()
    with _lifecycle_lock, _mutation_guards.lock:
        previous = _mutation_guards.callbacks.get(owner)
        if previous is callback:
            return
        if previous is not None:
            raise ModelMutationError(f"model-mutation owner {owner!r} already registered")
        if _mutation_guards.dispatching:
            raise ModelMutationError("cannot register during model-mutation dispatch")
        _mutation_guards.callbacks[owner] = callback


def dispatch_model_mutation(worker, *, operation: str, role: str) -> None:
    """Worker-only guard before super reload/start; an exception prevents mutation.

    No runtime config/cache availability gate is imposed here. Guards must use
    their loaded-model ownership, even after config overrides, and pass through
    workers they do not own. Registration/recursive dispatch during a guard is
    rejected. No guard can authorize or execute an update on its own.
    """
    if operation not in ("reload_weights", "start_weight_update") or role not in (
        "target", "draft",
    ) or (operation == "reload_weights" and role != "target"):
        raise ValueError("unsupported model-mutation operation/role")
    _ensure_lifecycle_process()
    with _lifecycle_lock, _mutation_guards.lock:
        if _mutation_guards.dispatching:
            raise ModelMutationError("recursive model-mutation dispatch")
        _mutation_guards.dispatching = True
        try:
            context = ModelMutationContext(worker, operation, role)
            for owner, callback in sorted(_mutation_guards.callbacks.items()):
                try:
                    result = callback(context)
                    if result is not None:
                        if inspect.iscoroutine(result):
                            result.close()
                        raise ModelMutationError("model-mutation guards must return None or raise")
                except Exception as exc:
                    raise ModelMutationError(
                        f"model-mutation owner {owner!r} refused {operation}/{role}: {exc}"
                    ) from exc
        finally:
            _mutation_guards.dispatching = False
