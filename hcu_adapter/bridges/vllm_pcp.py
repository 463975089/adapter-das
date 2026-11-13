# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""V4.1 PCP attention platform switch and catalog record.

The owned PCP classes live in `hcu_adapter/kernels/pcp_fork.py`. This
bridge only selects the mode, keeps the once-per-process lifecycle and records
which implementation serves `vllm_pcp`; it never imports vLLM at module load and
never builds the classes eagerly, so a host that does not request PCP (or does
not run V4.1) pays nothing and cannot fail startup here.

Publication direction: the plugin owns `ModelRegistry` publication and injects
the FP8 reference scorer through `pcp_fork.set_indexer_reference`. The
adapter cannot import the plugin, so it exposes the seam instead.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os

from hcu_adapter import hcu_env, registry
from hcu_adapter.kill_switch import op_disabled
from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle

MODE_VARIABLE = "HCU_VLLM_PCP_IMPL"
OP_NAME = "vllm_pcp"
COMPILE_FACTOR = "VLLM_HCU_ADAPTER_PCP_IMPLEMENTATION"
# The seam the plugin actually uses: it publishes the owned model class here.
TARGET = "vllm.model_executor.models.registry.ModelRegistry.register_model"

_LOG = logging.getLogger("hcu.adapter")


class PcpCompatibilityError(RuntimeError):
    """Requested PCP compatibility is unavailable or its contracts drifted."""


@dataclass(frozen=True)
class _Publication:
    pid: int
    mode: str


_lifecycle = ProcessLocalLifecycle(
    runtime_action="vllm-pcp", error_type=PcpCompatibilityError
)
_publication = None
_failure = None


def selected_mode() -> str:
    """Effective PCP implementation mode: `native` or `stock`."""
    if op_disabled(OP_NAME):
        return "stock"
    value, _ = hcu_env.resolve(MODE_VARIABLE, host="vllm")
    if value not in ("stock", "native"):
        raise ValueError(f"{MODE_VARIABLE} must be stock or native")
    return value


def _refuse(reason, error=None):
    global _failure
    # Publish the refusal before any user-defined exception formatting or reentry.
    if _failure is None:
        _failure = reason
        if error is not None:
            try:
                _failure = f"{reason}: {type(error).__name__}: {error}"
            except BaseException:
                pass
    raise PcpCompatibilityError(_failure) from error


def require_healthy() -> None:
    """Read only: never activate, restore a registry or clear failed state."""
    _lifecycle.require_runtime_healthy()
    if _failure is not None:
        raise PcpCompatibilityError(_failure)
    p = _publication
    if p is None:
        if selected_mode() == "native":
            raise PcpCompatibilityError("PCP requested before runtime activation")
        return
    if p.pid != os.getpid():
        raise PcpCompatibilityError(
            "PCP publication inherited from another PID; activate in this child first"
        )
    try:
        mode = selected_mode()
    except BaseException as error:
        _refuse("PCP policy invalid after registration; start a fresh process", error)
    if mode != p.mode:
        _refuse("PCP policy changed after registration; start a fresh process")


def install() -> bool:
    """Record the seam once per process. No vLLM import, no class building."""
    global _publication
    if _failure is not None:
        raise PcpCompatibilityError(_failure)
    try:
        mode = selected_mode()
    except BaseException as error:
        if _publication is not None:
            _refuse("PCP policy invalid after registration; start a fresh process", error)
        raise
    if mode == "stock" and _publication is None:
        return False

    def initialize():
        global _publication
        if _publication is not None:
            if selected_mode() != _publication.mode:
                _refuse("PCP policy changed after registration; start a fresh process")
            _publication = _Publication(pid=os.getpid(), mode=_publication.mode)
            return
        if mode == "stock":
            _publication = _Publication(pid=os.getpid(), mode="stock")
            return
        try:
            registry.register(
                op=OP_NAME, impl="native", target=TARGET,
                mechanism=registry.MECH_SEAM, source=__name__,
                detail="V4.1 PCP attention classes served by the adapter kernel module",
            )
        except BaseException as error:
            _refuse("PCP seam registration failed; start a fresh process", error)
        _publication = _Publication(pid=os.getpid(), mode="native")
        _LOG.info("HCU PCP attention: adapter-owned classes active (mode=native)")

    _lifecycle.run_once("vllm-pcp", initialize)
    return True
