# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""V4.1 Channel-FP8 quantization admission owned by the adapter boundary.

The distinct channel-FP8 configuration is a native vLLM quantization route that
HCU only had to surface; it therefore belongs to the adapter. Discovery imports
no framework or vendor.

The DSpark draft registration is upstream admission too: signature check, single
owner check and a lazy string handoff. Two hooks in the old plugin implementation
were private mechanism and stay in the plugin: the Engram/EDP guard and the DSpark
Graph support gate. They are injected through seams, and a second, different
owner is refused rather than silently replaced.

Binding is deferred to activation on purpose. Discovery through the entry point
must import this module without torch or vLLM, so the plugin registers a binder
callable instead of the providers themselves; the binder runs inside the
framework, where the plugin mechanism becomes importable.
"""
from __future__ import annotations

from importlib import import_module as _import_module
import inspect
import os
from threading import RLock

from hcu_adapter import dsv41_dspark_registration, hcu_env, registry
from hcu_adapter.kill_switch import op_disabled
from hcu_adapter.extensions import ExtensionCommitError, ExtensionContext, ExtensionSpec
from packaging.version import InvalidVersion, Version

MODE_VARIABLE = "HCU_VLLM_DSV41_CHANNEL_FP8_IMPL"
QUANTIZATION_METHOD = "hcu_dsv41_channel_fp8"
CAPABILITY = "dsv41-channel-fp8-routing"
OP = "vllm_dsv41_channel_fp8"
TARGET = "vllm.model_executor.layers.quantization.register_quantization_config"

_lock = RLock()
_publication = None
_failure: str | None = None
_publishing = False
_engram_edp_hooks = None
_dspark_graph_hooks = None
_provider_binder = None
_draft_registration = dsv41_dspark_registration


class Dsv41ActivationError(RuntimeError):
    """The routing capability cannot be registered under the current contract."""


def set_provider_binder(binder) -> None:
    """Register the zero-argument binder that installs the plugin providers.

    Import stays cheap and framework-free; the binder is invoked once, at
    activation, right before the hooks are consumed. A second, different owner is
    refused like the hooks themselves.
    """
    global _provider_binder
    if not callable(binder):
        raise Dsv41ActivationError("DSV4.1 provider binder must be callable")
    if _provider_binder is not None and _provider_binder is not binder:
        raise Dsv41ActivationError("DSV4.1 provider binder already has another owner")
    _provider_binder = binder


def set_engram_edp_hooks(install, check) -> None:
    """Bind the plugin-owned Engram/EDP guard (install/check pair)."""
    global _engram_edp_hooks
    if not callable(install) or not callable(check):
        raise Dsv41ActivationError("Engram/EDP hooks must be callable")
    if _engram_edp_hooks is not None and _engram_edp_hooks != (install, check):
        raise Dsv41ActivationError("Engram/EDP guard already has another owner")
    _engram_edp_hooks = (install, check)


def set_dspark_graph_hooks(install, check) -> None:
    """Bind the plugin-owned DSpark Graph support gate (install/check pair)."""
    global _dspark_graph_hooks
    if not callable(install) or not callable(check):
        raise Dsv41ActivationError("DSpark Graph hooks must be callable")
    if _dspark_graph_hooks is not None and _dspark_graph_hooks != (install, check):
        raise Dsv41ActivationError("DSpark Graph support already has another owner")
    _dspark_graph_hooks = (install, check)


def selected_mode() -> str:
    """Effective mode: `native` (register the config) or `stock` (do nothing)."""
    if op_disabled(OP):
        return "stock"
    value, _ = hcu_env.resolve(MODE_VARIABLE, host="vllm")
    if value not in ("stock", "native"):
        raise Dsv41ActivationError(f"{MODE_VARIABLE} must be stock or native")
    return value


def _after_fork_child() -> None:
    global _lock, _failure
    _lock = RLock()
    if _publishing and _failure is None:
        _failure = "fork inherited incomplete DSV4.1 registration; use a fresh process"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def _fatal(reason: str) -> None:
    global _failure
    if _failure is None:
        _failure = reason
    raise ExtensionCommitError(_failure)


def _require_release(raw_version: str | None) -> None:
    try:
        version = Version(raw_version) if raw_version is not None else None
    except (InvalidVersion, TypeError):
        version = None
    if version is None or version.epoch != 0 or version.release[:2] != (0, 33):
        raise Dsv41ActivationError(
            f"DSV4.1 routing requires vLLM >=0.33.0,<0.34.0; got {raw_version!r}"
        )


def check_health() -> None:
    """Read the public selector and refuse a lost owner without repairing it."""
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _publication is None:
            raise Dsv41ActivationError("DSV4.1 routing registration is not active")
        quant, config_class, register, resolve, draft_state = _publication
        try:
            owned = (
                quant.register_quantization_config is register
                and quant.get_quantization_config is resolve
                and QUANTIZATION_METHOD in quant.QUANTIZATION_METHODS
                and resolve(QUANTIZATION_METHOD) is config_class
            )
            if draft_state is not None:
                if _draft_registration is None:
                    raise Dsv41ActivationError(
                        "DSpark draft registration owner is not bound"
                    )
                _draft_registration.check_registration(draft_state)
        except BaseException as error:
            try:
                _fatal("DSV4.1 public registration is unavailable")
            except ExtensionCommitError as refusal:
                raise refusal from error
        if not owned:
            _fatal("DSV4.1 public quantization registration changed owner")
        # The registry record is part of the published state, not bookkeeping:
        # activation makes `_publication` and the record one atomic step, so a
        # later change to either means the process is no longer in the state it
        # was admitted in. Without this, `registry.unregister()`-style external
        # edits (or a second owner overwriting the op) went unnoticed while the
        # extension still reported healthy.
        try:
            record = registry.get(OP)
        except BaseException as error:
            try:
                _fatal("DSV4.1 registry record is unreadable")
            except ExtensionCommitError as refusal:
                raise refusal from error
        if (record is None or record.get("implementation") != "native"
                or record.get("target") != TARGET):
            _fatal("DSV4.1 registry record changed owner or target")
        try:
            if _engram_edp_hooks is None or _dspark_graph_hooks is None:
                raise RuntimeError("DSV4.1 guard providers are not bound")
            _engram_edp_hooks[1]()
            _dspark_graph_hooks[1]()
        except RuntimeError as error:
            _fatal(str(error))


def activate_extension(context: ExtensionContext) -> None:
    """Publish one distinct config; never overwrite FP8, CT or oracle owners."""
    global _publication, _publishing
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _publishing:
            _fatal("recursive DSV4.1 quantization registration")
        _require_release(context.vllm_version)
        if selected_mode() == "stock":
            return
        if _provider_binder is not None:
            _provider_binder()
        if _publication is not None:
            _publishing = True
            try:
                check_health()
            except BaseException as error:
                try:
                    _fatal("DSV4.1 contract changed after public registration")
                except ExtensionCommitError as refusal:
                    raise refusal from error
            finally:
                _publishing = False
            return
        _publishing = True
        try:
            try:
                quant = _import_module("vllm.model_executor.layers.quantization")
                candidate = _import_module("hcu_adapter.dsv41_channel_fp8")
            except BaseException as error:
                # Native imports may register operators before failing. Do not
                # label such a partially prepared process as a clean fallback.
                try:
                    _fatal("DSV4.1 native quantization preparation failed")
                except ExtensionCommitError as refusal:
                    raise refusal from error
            config_class = candidate.HcuDSV41ChannelFP8Config
            for name in ("register_quantization_config", "get_quantization_config"):
                function = getattr(quant, name, None)
                if (
                    not callable(function)
                    or tuple(inspect.signature(function).parameters)
                    != ("quantization",)
                ):
                    raise Dsv41ActivationError(f"public quantization ABI changed: {name}")
            if not isinstance(quant.QUANTIZATION_METHODS, list):
                raise Dsv41ActivationError("public quantization registry contract changed")
            register = quant.register_quantization_config
            resolve = quant.get_quantization_config
            # Everything that can refuse has to refuse BEFORE the public
            # registry is touched. Core has no unregister, so a failure after
            # `register()` leaves this process with a config it cannot select
            # and cannot remove: the framework keeps starting, the native model
            # eventually fails at construction, and a same-process retry stacks
            # another draft publication on top. Both preconditions are checked
            # here, ahead of the registration below.
            if _engram_edp_hooks is None or _dspark_graph_hooks is None:
                raise Dsv41ActivationError("DSV4.1 guard providers are not bound")
            draft_plan = None
            if candidate._is_hcu():
                try:
                    draft_plan = _draft_registration.prepare_registration()
                except RuntimeError as error:
                    raise Dsv41ActivationError(str(error)) from error
            if QUANTIZATION_METHOD in quant.QUANTIZATION_METHODS:
                if resolve(QUANTIZATION_METHOD) is not config_class:
                    raise Dsv41ActivationError(
                        "DSV4.1 quantization name is already registered by another owner"
                    )
            else:
                try:
                    register(QUANTIZATION_METHOD)(config_class)
                    if resolve(QUANTIZATION_METHOD) is not config_class:
                        _fatal("DSV4.1 public registration did not retain its owner")
                except BaseException as error:
                    try:
                        _fatal("DSV4.1 public registration failed; use a fresh process")
                    except ExtensionCommitError as refusal:
                        raise refusal from error
            # From here on the name is registered and cannot be removed, so every
            # remaining failure has to become commit-fatal rather than a
            # recoverable error the framework would swallow.
            draft_state = None
            if draft_plan is not None:
                try:
                    draft_state = _draft_registration.publish_registration(draft_plan)
                except BaseException as error:
                    try:
                        _fatal("DSV4.1 draft registration failed; use a fresh process")
                    except ExtensionCommitError as refusal:
                        raise refusal from error
            try:
                _engram_edp_hooks[0]()
                _dspark_graph_hooks[0]()
            except BaseException as error:
                try:
                    _fatal(
                        "DSV4.1 guard install failed after public registration; "
                        "use a fresh process"
                    )
                except ExtensionCommitError as refusal:
                    raise refusal from error
            # `_publication` and the registry record are one step, not two.
            # `registry.register()` raises a plain ValueError for an unknown op,
            # an illegal mechanism or a duplicate target, and the extension
            # framework treats a non-fatal error as recoverable: it records the
            # failure and keeps booting. Publishing first and registering second
            # therefore left a window where `_publication` was set, `_failure`
            # was unset and `check_health()` -- which never inspects the registry
            # -- would call the result healthy.
            try:
                registry.register(
                    op=OP, impl="native", target=TARGET,
                    mechanism=registry.MECH_SEAM, source=__name__,
                    detail="V4.1 channel-FP8 config registered through the public "
                           "quantization API without replacing FP8/CT owners",
                )
                _publication = (quant, config_class, register, resolve, draft_state)
            except BaseException as error:
                try:
                    _fatal(
                        "DSV4.1 registry publication failed after public "
                        "registration; use a fresh process"
                    )
                except ExtensionCommitError as refusal:
                    raise refusal from error
        finally:
            _publishing = False


def extension_spec() -> ExtensionSpec:
    return ExtensionSpec(
        name="das-hcu-dsv41", api_version=1, priority=100,
        capabilities=frozenset({CAPABILITY}),
        activate=activate_extension, check_health=check_health,
    )


__all__ = [
    "QUANTIZATION_METHOD", "CAPABILITY", "Dsv41ActivationError",
    "activate_extension", "check_health", "extension_spec",
    "selected_mode", "set_engram_edp_hooks", "set_dspark_graph_hooks",
    "set_provider_binder",
]
