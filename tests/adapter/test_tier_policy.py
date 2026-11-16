# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unit guards for the sglang-host tier ladder.

The ladder is lightop > aiter > jit > triton; silu_and_mul deliberately skips
jit (33-49 s cold ninja build, PR #96). No GPU, no lightop/aiter wheels needed:
the probes are plain callables.
"""

import pytest

from hcu_adapter import tier_policy


@pytest.fixture(autouse=True)
def _clean_log_state():
    tier_policy.reset_for_tests()
    yield
    tier_policy.reset_for_tests()


def _cand(**values):
    """Build a candidates mapping; None means an unavailable probe."""
    return {
        name: (lambda value=value: value) if value is not None else (lambda: None)
        for name, value in values.items()
    }


def test_priority_is_lightop_aiter_jit_triton():
    assert tier_policy.PRIORITY == ("lightop", "aiter", "jit", "triton")
    assert tier_policy.SILU_PRIORITY == ("lightop", "aiter", "triton")


def test_lightop_wins_over_everything():
    tier, impl = tier_policy.pick(
        "op",
        _cand(lightop="L", aiter="A", jit="J", triton="T"),
    )
    assert (tier, impl) == ("lightop", "L")


def test_aiter_wins_when_lightop_is_absent():
    tier, impl = tier_policy.pick(
        "op",
        _cand(lightop=None, aiter="A", jit="J", triton="T"),
    )
    assert (tier, impl) == ("aiter", "A")


def test_jit_outranks_triton_by_default():
    tier, impl = tier_policy.pick("op", _cand(jit="J", triton="T"))
    assert (tier, impl) == ("jit", "J")


def test_silu_ladder_skips_jit():
    tier, impl = tier_policy.pick(
        "silu_and_mul",
        _cand(lightop=None, aiter=None, jit="J", triton="T"),
        priority=tier_policy.SILU_PRIORITY,
    )
    assert (tier, impl) == ("triton", "T")


def test_explicit_tier_wins_over_ladder():
    tier, impl = tier_policy.pick(
        "op",
        _cand(lightop="L", aiter="A", jit="J", triton="T"),
        requested="triton",
    )
    assert (tier, impl) == ("triton", "T")


def test_explicit_planned_tier_is_reachable_but_not_auto():
    candidates = _cand(lightop="L", aiter=None, jit="J", triton="T")
    # Auto ladder excludes lightop (catalog planned).
    tier, impl = tier_policy.pick(
        "op", candidates, auto_tiers=("aiter", "jit", "triton")
    )
    assert (tier, impl) == ("jit", "J")
    # Explicit request still reaches it.
    tier, impl = tier_policy.pick(
        "op", candidates, auto_tiers=("aiter", "jit", "triton"),
        requested="lightop",
    )
    assert (tier, impl) == ("lightop", "L")


def test_unavailable_explicit_tier_falls_through():
    tier, impl = tier_policy.pick(
        "op",
        _cand(lightop=None, aiter="A", triton="T"),
        requested="lightop",
    )
    assert (tier, impl) == ("aiter", "A")


def test_unknown_explicit_tier_falls_through():
    tier, impl = tier_policy.pick(
        "op",
        _cand(aiter="A", triton="T"),
        requested="wat",
    )
    assert (tier, impl) == ("aiter", "A")


def test_probe_exception_counts_as_unavailable():
    def boom():
        raise ImportError("no wheel")

    tier, impl = tier_policy.pick(
        "op", {"lightop": boom, "triton": lambda: "T"}
    )
    assert (tier, impl) == ("triton", "T")


def test_no_usable_tier_raises():
    with pytest.raises(LookupError):
        tier_policy.pick("op", _cand(lightop=None, aiter=None, triton=None))
