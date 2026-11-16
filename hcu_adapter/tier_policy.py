# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""One tier ladder for sglang-host operators.

Policy (2026-11): prefer vendor kernels over JIT/triton for the operators the
sglang bridge owns::

    lightop > aiter > jit > triton

``silu_and_mul`` is the one deliberate exception: its sglang JIT body costs a
33-49 s ninja build per cold cache key (PR #96), so its ladder is
``lightop > aiter > triton`` (``SILU_PRIORITY``).

Explicit env always wins (the hcu_env contract), but an explicit tier that is
missing/broken is not trusted silently: the probe result wins over the name and
the caller falls through the ladder with one warning. Auto only walks tiers the
caller marks auto-eligible; tiers listed in ``candidates`` but omitted from
``auto_tiers`` remain reachable by explicit request (used for catalog
``planned`` tiers such as lightop quant / lightop MQA).

Probes are zero-arg callables returning the implementation object, or ``None``
when unavailable. A raising probe is treated as unavailable (logged once).
This module only depends on logging, so ``sgl_kernel`` can import it lazily
without an adapter import cycle.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Sequence

LOG = logging.getLogger("hcu.adapter")

PRIORITY: tuple[str, ...] = ("lightop", "aiter", "jit", "triton")
SILU_PRIORITY: tuple[str, ...] = ("lightop", "aiter", "triton")

Probe = Callable[[], Any]

_LOGGED: set[tuple[str, str]] = set()
_FAILED: set[tuple[str, str]] = set()


def _probe(op: str, tier: str, probe: Probe) -> Any:
    try:
        impl = probe()
    except Exception as exc:  # noqa: BLE001 - availability probe
        key = (op, tier)
        if key not in _FAILED:
            _FAILED.add(key)
            LOG.info("[HCU tier] %s: %s probe failed (%s)", op, tier, exc)
        return None
    return impl


def _log(op: str, tier: str, source: str) -> None:
    key = (op, tier)
    if key not in _LOGGED:
        _LOGGED.add(key)
        LOG.info("[HCU tier] %s -> %s (%s)", op, tier, source)


def pick(
    op: str,
    candidates: Mapping[str, Probe],
    *,
    priority: Sequence[str] = PRIORITY,
    auto_tiers: Sequence[str] | None = None,
    requested: str | None = None,
) -> tuple[str, Any]:
    """Return ``(tier, implementation)`` for ``op``.

    ``candidates`` maps tier name -> zero-arg probe. ``auto_tiers`` is the
    ordered auto ladder (defaults to ``priority``); tiers absent from it are
    explicit-only. ``requested`` is the raw value from ``hcu_env.resolve``;
    ``None``/``auto``/``""`` mean no explicit preference.
    """
    order = tuple(auto_tiers) if auto_tiers is not None else tuple(priority)
    if requested not in (None, "", "auto"):
        if requested in candidates:
            impl = _probe(op, requested, candidates[requested])
            if impl is not None:
                if requested not in order:
                    LOG.warning(
                        "[HCU tier] %s: explicit %s is outside the auto "
                        "ladder %s (catalog gate); using it as requested",
                        op,
                        requested,
                        " > ".join(order),
                    )
                _log(op, requested, "explicit")
                return requested, impl
            LOG.warning(
                "[HCU tier] %s: explicit tier %r unavailable; falling "
                "through to %s",
                op,
                requested,
                " > ".join(order),
            )
        else:
            LOG.warning(
                "[HCU tier] %s: unknown tier %r; falling through to %s",
                op,
                requested,
                " > ".join(order),
            )

    for tier in order:
        probe = candidates.get(tier)
        if probe is None:
            continue
        impl = _probe(op, tier, probe)
        if impl is not None:
            _log(op, tier, "auto")
            return tier, impl

    raise LookupError(
        f"no usable tier for {op!r} (auto ladder: {' > '.join(order)})"
    )


def reset_for_tests() -> None:
    """Clear once-logged sets (tests only)."""
    _LOGGED.clear()
    _FAILED.clear()
