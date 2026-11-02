"""Public vLLM lifecycle facade: resolved Worker coverage and admission only.

The callback registry and synchronization live in framework-free model_load.
Importing this bridge does not import vLLM; availability queries do so lazily.
No loader, runner, quantization or model implementation lives here.
"""

from hcu_adapter import model_load as _shared
from hcu_adapter.model_load import (
    ModelMutationContext,
    ModelMutationError,
    PostModelLoadContext,
    PostModelLoadError,
    preflight_model_lifecycle_hooks,
    register_model_lifecycle_hooks,
    register_model_mutation_guard,
    register_post_model_load,
)

POST_MODEL_LOAD_WORKER = "hcu_adapter.bridges.vllm_worker.Worker"
OFFICIAL_MODEL_LOAD_WORKER = "vllm.v1.worker.gpu_worker.Worker"


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


def model_mutation_guard_unavailable_reason(vllm_config) -> str | None:
    """Query the resolved uni/mp worker seam; serving env cache is allowed."""
    return _worker_unavailable_reason(vllm_config)


def require_model_mutation_guard(vllm_config) -> None:
    """Call before selecting a route which requires weight-mutation protection."""
    reason = model_mutation_guard_unavailable_reason(vllm_config)
    if reason is not None:
        raise ModelMutationError(reason)


def dispatch_post_model_load(worker, *, load_dummy_weights: bool) -> None:
    """Apply vLLM admission inside the shared gate, only with registered owners."""
    _shared.dispatch_post_model_load(
        worker, load_dummy_weights=load_dummy_weights,
        before_dispatch=lambda: require_post_model_load(worker.vllm_config),
    )


def dispatch_model_mutation(worker, *, operation: str, role: str) -> None:
    """Validate vLLM mutation identifiers before dispatching shared guards."""
    if operation not in ("reload_weights", "start_weight_update") or role not in (
        "target", "draft",
    ) or (operation == "reload_weights" and role != "target"):
        raise ValueError("unsupported model-mutation operation/role")
    _shared.dispatch_model_mutation(worker, operation=operation, role=role)
