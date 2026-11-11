# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""V4.1 CT W4A8 quantization admission owned by the adapter boundary.

Registering a distinct compressed-tensors W4A8 configuration through the public
quantization API is native vLLM behaviour that HCU only surfaces, so it belongs
to the adapter. The config class itself stays plugin-private (it carries HCU MoE
routing) and is injected through a seam; a second, different owner is refused.
"""
from __future__ import annotations

from importlib import import_module
import inspect
import os
from threading import RLock

from hcu_adapter import hcu_env, registry
from hcu_adapter.kill_switch import op_disabled
from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec
from hcu_adapter.dsv41_activation import _require_release

MODE_VARIABLE = "HCU_VLLM_CT_W4A8_IMPL"
NAME = "hcu_dsv41_ct_w4a8"
OP = "vllm_ct_w4a8"
TARGET = "vllm.model_executor.layers.quantization.register_quantization_config:ct_w4a8"

_lock = RLock()
_publication = None
_failure = None
_publishing = False
_config_provider = None
_provider_binder = None


def set_provider_binder(binder) -> None:
    """Register the zero-argument binder that installs the plugin provider.

    Deferred to activation so discovery imports this module without vLLM; a
    second, different owner is refused.
    """
    global _provider_binder
    if not callable(binder):
        raise ValueError("CT W4A8 provider binder must be callable")
    if _provider_binder is not None and _provider_binder is not binder:
        raise RuntimeError("CT W4A8 provider binder already has another owner")
    _provider_binder = binder


def set_config_provider(provider) -> None:
    """Bind a zero-argument provider returning the plugin CT W4A8 config class.

    A provider keeps the plugin import lazy and the direction rule intact; a
    second, different owner is refused.
    """
    global _config_provider
    if not callable(provider):
        raise ValueError("CT W4A8 config provider must be callable")
    if _config_provider is not None and _config_provider is not provider:
        raise RuntimeError("CT W4A8 config already has another owner")
    _config_provider = provider


def _require_config_class():
    if _config_provider is None:
        raise RuntimeError("CT W4A8 config provider is not bound")
    return _config_provider()


def selected_mode() -> str:
    if op_disabled(OP):
        return "stock"
    value, _ = hcu_env.resolve(MODE_VARIABLE, host="vllm")
    if value not in ("stock", "native"):
        raise ValueError(f"{MODE_VARIABLE} must be stock or native")
    return value


def _fork():
    global _lock, _failure
    _lock = RLock()
    if _publishing and _failure is None:
        _failure = "fork inherited incomplete CT W4A8 registration"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_fork)


def _fatal(message):
    global _failure
    if _failure is None:
        _failure = message
    raise ExtensionCommitError(_failure)


def check_health():
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _publication is None:
            raise RuntimeError("CT W4A8 registration is not active")
        quant, cls, register, resolve = _publication
        try:
            owned = (
                quant.register_quantization_config is register
                and quant.get_quantization_config is resolve
                and NAME in quant.QUANTIZATION_METHODS
                and resolve(NAME) is cls
            )
        except BaseException as error:
            try:
                _fatal("CT W4A8 public selector is unavailable")
            except ExtensionCommitError as refusal:
                raise refusal from error
        if not owned:
            _fatal("CT W4A8 public registration changed owner")


def activate_extension(context):
    global _publication, _publishing
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _publishing:
            _fatal("recursive CT W4A8 activation")
        _require_release(context.vllm_version)
        if selected_mode() == "stock":
            return
        if _provider_binder is not None:
            _provider_binder()
        if _publication is not None:
            try:
                check_health()
            except BaseException as error:
                try:
                    _fatal("CT W4A8 contract changed after publication")
                except ExtensionCommitError as refusal:
                    raise refusal from error
            return
        _publishing = True
        try:
            try:
                quant = import_module("vllm.model_executor.layers.quantization")
                cls = _require_config_class()
            except BaseException as error:
                try:
                    _fatal("CT W4A8 native preparation failed")
                except ExtensionCommitError as refusal:
                    raise refusal from error
            register, resolve = quant.register_quantization_config, quant.get_quantization_config
            if any(
                tuple(inspect.signature(fn).parameters) != ("quantization",)
                for fn in (register, resolve)
            ):
                raise RuntimeError("CT W4A8 public quantization registration ABI changed")
            if NAME in quant.QUANTIZATION_METHODS:
                if resolve(NAME) is not cls:
                    raise RuntimeError("CT W4A8 name has a foreign owner")
            else:
                try:
                    register(NAME)(cls)
                except BaseException as error:
                    try:
                        _fatal("CT W4A8 publication failed; restart the process")
                    except ExtensionCommitError as refusal:
                        raise refusal from error
            _publication = (quant, cls, register, resolve)
            registry.register(
                op=OP, impl="native", target=TARGET,
                mechanism=registry.MECH_SEAM, source=__name__,
                detail="distinct CT W4A8 config registered through the public "
                       "quantization API; native CT owners are not replaced",
            )
            check_health()
        finally:
            _publishing = False


def extension_spec():
    return ExtensionSpec(
        name="das-hcu-ct-w4a8", api_version=1, priority=105,
        capabilities=frozenset({"dsv41-ct-w4a8"}),
        activate=activate_extension, check_health=check_health,
    )
