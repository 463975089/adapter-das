# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Install narrow HCU compatibility for existing official vLLM AR behavior.

No collective math, official class bodies, fusion patterns or platform identity
are copied. The existing official fusion-pass global admission stays unchanged.
"""
from functools import wraps
from importlib import import_module
import inspect
import os
from pathlib import Path
import sys
from threading import RLock

from hcu_adapter import allreduce as policy_api, allreduce_aiter, registry
from hcu_adapter.allreduce import AllReduceCompatibilityError, State, requested_policy
from hcu_adapter.extensions import ExtensionCommitError

FACTOR = "VLLM_HCU_ADAPTER_AITER_AR_POLICY"
WRAPPER = "vllm.distributed.device_communicators.aiter_custom_all_reduce"
PROVIDER = "aiter.dist.device_communicators.custom_all_reduce"
_lock = RLock()
_installed = None
_bindings = None
_attempted = False
_installing = False
_fork_error = False
_commit_failure = None


def _after_fork_child():
    global _lock, _fork_error
    _lock = RLock()
    _fork_error = _fork_error or _attempted or _installing


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def _callable(function, *args, **kwargs):
    methods = (function,) if inspect.isroutine(function) else (function, getattr(function, "__call__", None))
    if not callable(function) or any(test(value) for value in methods for test in (
        inspect.iscoroutinefunction, inspect.isasyncgenfunction, inspect.isgeneratorfunction
    )):
        raise AllReduceCompatibilityError("required synchronous public callable missing")
    inspect.signature(function).bind(*args, **kwargs)


def _require_host_healthy():
    from .vllm import require_runtime_healthy
    # Runtime takes host -> AR. Never call this while holding the AR lock.
    require_runtime_healthy()


def _check_owner_state():
    """Owner-local validation only; caller holds AR lock, never takes host lock."""
    if _commit_failure is not None:
        raise ExtensionCommitError(_commit_failure)
    if _fork_error:
        raise AllReduceCompatibilityError("AR construction/in-progress state inherited across fork; use spawn")
    if _installed is not None and requested_policy() != _installed[0]:
        raise AllReduceCompatibilityError("AR policy changed after installation; use a fresh process")
    if (_installed is not None and _installed[0].selected
            and import_module("vllm.envs").VLLM_BATCH_INVARIANT):
        raise AllReduceCompatibilityError("selected HCU AITER AR does not support VLLM_BATCH_INVARIANT")
    if _bindings is not None:
        cls, ops, constructor, gate = _bindings
        if cls.__init__ is not constructor or getattr(inspect.getattr_static(
            ops, "is_custom_all_reduce_enabled"
        ), "__func__", None) is not gate:
            raise AllReduceCompatibilityError("AR bindings changed after installation")


def state():
    """Read-only current-process state; never installs or activates a backend."""
    _require_host_healthy()
    with _lock:
        _check_owner_state()
        return State(False, None, False) if _installed is None else _installed[2]


def effective_compile_factor():
    host_error = None
    try:
        _require_host_healthy()
    except Exception as error:
        host_error = type(error).__name__ + ": " + str(error)
    with _lock:
        if _installed is None:
            raise AllReduceCompatibilityError("AR factor requested before installation")
        try:
            if host_error is not None:
                return policy_api.factor_value(_installed[0], {"invalid": host_error})
            _check_owner_state()
        except Exception as error:
            # Official compile_factors logs+SKIPS exceptions. Keep an invalid
            # state hashed in both ordinary/AOT layers; selectors/constructors
            # still raise and cannot compute under this sentinel.
            return policy_api.factor_value(
                _installed[0], {"invalid": type(error).__name__ + ": " + str(error)}
            )
        return _installed[1]


def _provider_identity():
    identity = allreduce_aiter.provider_identity(
        import_module("aiter"), import_module(PROVIDER), sys.modules
    )
    sources = (Path(__file__), Path(__file__).with_name("vllm.py"),
               Path(policy_api.__file__), Path(allreduce_aiter.__file__),
               Path(import_module("hcu_adapter.hcu_env").__file__),
               Path(import_module("hcu_adapter.kill_switch").__file__))
    identity["adapter"] = {str(path.relative_to(Path(__file__).parents[1])):
                           allreduce_aiter.digest(path) for path in sources}
    # These official files affect wrapper/dispatch, including eager Gemma.
    identity["vllm"] = {name: allreduce_aiter.digest(import_module(name).__file__)
                         for name in (WRAPPER, "vllm._aiter_ops")}
    return identity


def _install(*, require_available=False):
    global _installed, _bindings, _commit_failure
    from .vllm import detect_hcu, _ensure_vllm_compatible
    _check_owner_state()
    _ensure_vllm_compatible()
    if not detect_hcu().detected:
        raise AllReduceCompatibilityError("requires detected HCU; no architecture spoofing")
    policy = requested_policy()
    if _installed is not None:
        if require_available and _installed[2].available is not True:
            raise AllReduceCompatibilityError("availability must be required before first installation")
        return _installed[2]
    envs = import_module("vllm.envs")
    if policy.selected and envs.VLLM_BATCH_INVARIANT:
        raise AllReduceCompatibilityError("selected HCU AITER AR does not support VLLM_BATCH_INVARIANT")
    factors = getattr(envs, "environment_variables", None)
    _callable(getattr(envs, "compile_factors", None))
    if not isinstance(factors, dict) or envs._is_envs_cache_enabled():
        raise AllReduceCompatibilityError("install AR factor before vLLM env caching")
    if FACTOR in factors:
        raise AllReduceCompatibilityError("AR compile factor already owned")
    if import_module("torch.distributed").is_initialized():
        raise AllReduceCompatibilityError("prepare AR before distributed initialization")
    cls = import_module(WRAPPER).AiterCustomAllreduce
    module = import_module("vllm._aiter_ops")
    ops = module.rocm_aiter_ops
    original = cls.__init__
    _callable(original, object(), object(), "cuda:0", max_size=None, register_graph_buffers=False)
    _callable(ops.is_custom_all_reduce_enabled)
    gate = inspect.getattr_static(ops, "is_custom_all_reduce_enabled")
    if (not isinstance(gate, classmethod) or gate.__func__.__module__ != "vllm._aiter_ops"
            or original.__module__ != WRAPPER or inspect.unwrap(original) is not original
            or getattr(original, "_hcu_ar_owner", None) is not None):
        raise AllReduceCompatibilityError("official AR constructor/gate already replaced")
    for op in ("vllm_allreduce", "vllm_allreduce_admission"):
        if registry.get(op) is not None:
            raise AllReduceCompatibilityError("AR runtime binding already owned: " + op)
    identity = {}
    register_fused = registration_error = None
    available = None
    if policy.selected or require_available:
        register_fused = getattr(ops, "register_custom_allreduce_ops", None)
        registration_error = getattr(module, "CustomAllreduceRegistrationError", None)
        try:
            _callable(register_fused)
            if not inspect.isclass(registration_error) or not issubclass(registration_error, RuntimeError):
                raise TypeError("public registration error type missing")
            _callable(registration_error, "prepare", registration_started=False)
        except Exception as error:
            raise AllReduceCompatibilityError("requires public Custom AR registration ABI from target27c") from error
        provider = import_module(PROVIDER)
        if getattr(provider, "custom_ar", None) is not True:
            raise AllReduceCompatibilityError("public provider reports custom AR unavailable")
        _callable(provider.CustomAllreduce.__init__, object(), object(), "cuda:0",
                  max_size=policy.staging_bytes, enable_register_for_capturing=False)
        if policy.transport == "fabric":
            vendor = import_module("aiter")
            _callable(getattr(vendor, "fabric_ar_available", None))
            if not vendor.fabric_ar_available():
                raise AllReduceCompatibilityError("explicit fabric requires provider capability")
        identity = _provider_identity()  # Includes the actual loaded AR DSO.
        available = True
        if not policy.selected:
            identity = {}  # Required + OFF checks ABI but does not split OFF caches.
    factor = policy_api.factor_value(policy, identity)

    @wraps(original)
    def construct(self, group, device, max_size=None, *, register_graph_buffers=True):
        global _attempted, _commit_failure
        _require_host_healthy()
        with _lock:
            _check_owner_state()
            if not policy.selected:
                raise AllReduceCompatibilityError("AITER AR dispatch is OFF; use the official fallback chain")
            _attempted = True
            try:
                return original(self, group, device,
                                policy.staging_bytes if max_size is None else max_size,
                                register_graph_buffers=register_graph_buffers and policy.register_graph_buffers)
            except BaseException as error:
                _commit_failure = "HCU AR construction failed; native state may exist; restart process"
                raise ExtensionCommitError(_commit_failure) from error

    def selected(cls):
        _require_host_healthy()
        with _lock:
            _check_owner_state()
            return policy.selected

    construct._hcu_ar_owner = __name__
    clean_error = None
    try:
        if policy.selected:
            try:
                result = register_fused()
            except registration_error as error:
                if error.registration_started is False:
                    clean_error = error
                raise
            if type(result) is not bool:
                raise AllReduceCompatibilityError("public AR registration must return bool")
        status = State(True, available, policy.selected, policy.selected)
        _installed = (policy, factor, status)
        factors[FACTOR] = effective_compile_factor
        cls.__init__ = construct
        ops.is_custom_all_reduce_enabled = classmethod(selected)
        _bindings = (cls, ops, construct, selected)
        registry.register("vllm_allreduce", "official-aiter", WRAPPER + ".AiterCustomAllreduce.__init__",
                          registry.MECH_REBIND, source=__name__)
        registry.register("vllm_allreduce_admission", "official-aiter-gate",
                          "vllm._aiter_ops.rocm_aiter_ops.is_custom_all_reduce_enabled",
                          registry.MECH_REBIND, source=__name__)
    except BaseException as error:
        if error is clean_error:
            raise
        _commit_failure = "HCU AR compatibility commit failed; restart process"
        raise ExtensionCommitError(_commit_failure) from error
    return status


def install(*, require_available=False):
    """Prepare then install official AR compatibility; no Plugin is required.

    A required ABI may be available while kill/selection leaves dispatch OFF.
    A clean failure precedes all owned mutation; partial publication is fatal.
    """
    global _installing
    _require_host_healthy()
    with _lock:
        _check_owner_state()
        if _installing:
            raise AllReduceCompatibilityError("recursive AR installation")
        _installing = True
        try:
            return _install(require_available=require_available)
        finally:
            _installing = False


def prepare_runtime():
    """Default unselected startup is neutral; explicit config/kill is honored.

    Availability requirements are expressed by install(require_available=True),
    not by inventing a fourth Plugin capability owner.
    """
    _require_host_healthy()
    with _lock:
        _check_owner_state()
        if _installed is not None:
            return _installed[2]
    from hcu_adapter import hcu_env, kill_switch
    _, source = hcu_env.resolve("HCU_AITER_AR", host="vllm")
    if source in ("explicit", "config") or kill_switch.op_disabled("aiter_ar"):
        return install()
    return state()
