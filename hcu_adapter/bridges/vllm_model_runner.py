# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Explicit vLLM runner factories; the exact Adapter Worker retains load hooks.

Factories implement plugin capabilities through the upstream Worker._make_model_runner
subclass seam. The registry imports no framework and does not own runner algorithms.
"""
from dataclasses import dataclass
from threading import RLock
import os


@dataclass(frozen=True)
class ModelRunnerFactory:
    owner: str
    matches: object
    create: object


_lock = RLock()
_factories = {}
_frozen = False
_pid = os.getpid()


def _check_process():
    global _pid, _lock
    if os.getpid() != _pid:
        _lock = RLock()
        if _factories:
            raise RuntimeError("active runner registry inherited across fork; use spawn")
        _pid = os.getpid()


def register_model_runner_factory(owner, *, matches, create):
    global _pid
    if not isinstance(owner, str) or not owner or not callable(matches) or not callable(create):
        raise ValueError("runner factory requires owner, matches and create")
    candidate = ModelRunnerFactory(owner, matches, create)
    _check_process()
    with _lock:
        _check_process()
        current = _factories.get(owner)
        if current == candidate:
            return candidate
        if _frozen or current is not None:
            raise RuntimeError("runner factory registration is frozen or owner conflicts")
        _factories[owner] = candidate
        return candidate


def require_model_runner_factory(registration):
    _check_process()
    with _lock:
        _check_process()
        if _factories.get(registration.owner) is not registration:
            raise RuntimeError("runner factory ownership/process changed")


def create_model_runner(vllm_config, device, *, default):
    global _frozen
    _check_process()
    with _lock:
        _check_process()
        _frozen = True
        registrations = tuple(_factories.values())
    selected = []
    for entry in registrations:
        require_model_runner_factory(entry)
        matches = bool(entry.matches(vllm_config))
        require_model_runner_factory(entry)
        if matches:
            selected.append(entry)
    if len(selected) > 1:
        raise RuntimeError("multiple plugin runner factories matched the same configuration")
    if not selected:
        return default()
    registration = selected[0]
    require_model_runner_factory(registration)
    runner = registration.create(vllm_config, device)
    require_model_runner_factory(registration)
    if runner is None:
        raise RuntimeError("selected plugin runner factory returned no runner")
    return runner
