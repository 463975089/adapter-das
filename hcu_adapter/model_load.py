"""Framework-light, process-local callbacks for successful worker model loads.

Extensions register during activation (spawn) or inherit declarations (fork).
Importing this module imports no vLLM and selects no optimization. Availability
queries lazily inspect vLLM's cache/launch state. Callback failures fail the load.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock


POST_MODEL_LOAD_WORKER = "hcu_adapter.bridges.vllm_worker.Worker"
OFFICIAL_MODEL_LOAD_WORKER = "vllm.v1.worker.gpu_worker.Worker"


class PostModelLoadError(RuntimeError):
    """A callback conflicts, fails, or lacks a supported load boundary."""


@dataclass(frozen=True)
class PostModelLoadContext:
    worker: object
    load_dummy_weights: bool

    @property
    def vllm_config(self):
        return self.worker.vllm_config


class _Callbacks:
    def __init__(self):
        self.callbacks: dict[str, Callable[[PostModelLoadContext], None]] = {}
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


_callbacks = _Callbacks()


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
    _callbacks.ensure_process()
    with _callbacks.lock:
        previous = _callbacks.callbacks.get(owner)
        if previous is callback:
            return
        if previous is not None:
            raise PostModelLoadError(f"post-model-load owner {owner!r} already registered")
        if _callbacks.dispatching:
            raise PostModelLoadError("cannot register during post-model-load dispatch")
        _callbacks.callbacks[owner] = callback


def post_model_load_unavailable_reason(vllm_config) -> str | None:
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

    if envs._is_envs_cache_enabled():
        return "post-model-load boundary is unavailable after env cache enabled"
    if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
        return "elastic scale-up bypasses the synchronous worker load boundary"
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
    _callbacks.ensure_process()
    with _callbacks.lock:
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
