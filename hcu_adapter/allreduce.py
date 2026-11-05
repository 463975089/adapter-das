# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Framework-free all-reduce policy. No framework or provider imports.

These are existing registered deployment/debug inputs, not a second set of
enable switches. OFF describes the official fallback chain, not necessarily RCCL.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from . import hcu_env, kill_switch

MIB = 1024 * 1024


class AllReduceCompatibilityError(RuntimeError):
    """Preparation or reuse of an official AR backend is unsafe."""


@dataclass(frozen=True)
class Policy:
    selected: bool
    admission_bytes: int = 64 * MIB
    staging_bytes: int = 128 * MIB
    transport: str = "ipc"
    register_graph_buffers: bool = False
    small_message: str = "off"


@dataclass(frozen=True)
class State:
    installed: bool
    available: bool | None
    selected: bool
    fused_ops_registered: bool = False
    fusion_pass: str = "official-global-gate-unchanged"


def _value(name):
    return hcu_env.resolve(name, host="vllm")[0]


def _provider_value(name):
    value, source = hcu_env.resolve(name, host="vllm")
    direct = hcu_env.environment_value(name)
    # The admitted provider reads these inputs itself and exposes no matching
    # public constructor arguments. Never advertise config which it won't see.
    if value.strip().lower() != direct.strip().lower():
        raise AllReduceCompatibilityError(
            f"{name} ({source}) differs from provider environment; use its explicit "
            "environment input or the default; config is not exported to AITER"
        )
    return value


def _flag(name):
    value = _value(name).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise AllReduceCompatibilityError(f"{name} must be a boolean")


def requested_policy():
    # A kill veto is not an availability failure, even when required. Do not
    # parse irrelevant vendor options (or probe any architecture) while killed.
    if kill_switch.op_disabled("aiter_ar"):
        return Policy(False)
    value = _value("HCU_AITER_AR").strip().lower()
    if value.startswith("refused") or value in ("0", "false", "off", "no", ""):
        return Policy(False)
    if value not in ("1", "true", "on", "yes") and not value.startswith("allowed"):
        raise AllReduceCompatibilityError("invalid effective HCU_AITER_AR setting")
    if not _flag("VLLM_ROCM_USE_AITER_CUSTOM_AR"):
        return Policy(False)
    try:
        size = int(_provider_value("AITER_AR_MAX_SIZE_MB"))
        if size <= 0:
            raise ValueError
    except ValueError:
        raise AllReduceCompatibilityError("AITER_AR_MAX_SIZE_MB must be a positive integer") from None
    transport = _provider_value("AITER_AR_TRANSPORT").strip().lower()
    if transport not in ("ipc", "fabric", "auto"):
        raise AllReduceCompatibilityError("AITER_AR_TRANSPORT must be ipc, fabric or auto")
    small = _provider_value("AITER_AR_SMALL_MESSAGE").strip().lower()
    if small not in ("off", "ll", "ll128", "auto"):
        raise AllReduceCompatibilityError("invalid AITER_AR_SMALL_MESSAGE")
    return Policy(True, size * MIB, max(128, 2 * size) * MIB, transport,
                  _flag("AITER_AR_ENABLE_REG_CAPTURE"), small)


def effective_sources():
    """Read current resolution provenance; this does not activate anything."""
    names = ("HCU_DISABLE_OPS", "HCU_AITER_AR", "VLLM_ROCM_USE_AITER_CUSTOM_AR",
             "AITER_AR_MAX_SIZE_MB", "AITER_AR_TRANSPORT",
             "AITER_AR_ENABLE_REG_CAPTURE", "AITER_AR_SMALL_MESSAGE")
    return {name: hcu_env.resolve(name, host="vllm") for name in names}


def factor_value(policy, identity):
    return json.dumps({"abi": 2, "policy": vars(policy), "implementation": identity},
                      sort_keys=True, separators=(",", ":"))
