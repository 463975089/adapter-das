# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unified micro-bench harness for --autotune.

One timing idiom for every op-tier pair (the repo's established
cuda.Event pattern: warmup N, sync, time M iters, sync), one JSON output
shame. Shapes are registered per-op in SHAPES as callables that return
(real inputs, fn) so tier-specific layouts (mqa pool interleaving etc.)
stay next to the judge map. Numbers are THIS machine only -- the repo's
standing rule is perf numbers do not transfer across boxes.

Structure only in v1: harness + shape registry + mock op (pure-torch)
so the runner and gates run on GPU-less machines and CI. Real op
factories are added alongside choices.py entries as they get verified on
a card; a missing factory = runner reports the op bench skipped (never
fabricates a number).
"""

import json
import time

import torch


class BenchResult:
    def __init__(self, ok, ms_per_iter=None, iters=0, error=None, note=None):
        self.ok = ok
        self.ms_per_iter = ms_per_iter
        self.iters = iters
        self.error = error
        self.note = note

    def as_dict(self):
        d = {"ok": self.ok, "iters": self.iters, "note": self.note}
        if self.ms_per_iter is not None:
            d["ms_per_iter"] = round(self.ms_per_iter, 4)
        if self.error:
            d["error"] = str(self.error)[:300]
        return d


def time_callable(fn, warmup=5, iters=20):
    """The repo's standard timing idiom (cuda Events on GPU); CPU-only
    machines fall back to wall clock so gates/tests still exercise the
    harness. Returns ms/iter."""
    if not torch.cuda.is_available():
        t0 = time.perf_counter()
        for _ in range(warmup):
            fn()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1e3
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# shape factories: kind -> callable() -> (fn, note). `fn` binds its own
# inputs; the harness only times it. Register realfactory variants here as
# they get card-verified; nothing in the runner special-cases op names.
# ---------------------------------------------------------------------------

def _mock_matmul_factory(m=512, k=512, n=512, dtype=torch.float32):
    """Pure-torch stand-in used by gates/tests on GPU-less machines.
    Real per-op factories (activations, norms, mqa, quant) land here as
    `real:` kinds once verified on a card."""
    a = torch.randn(m, k)
    b = torch.randn(k, n)

    def fn():
        return a @ b

    return fn, f"mock matmul {m}x{k}x{n}"


SHAPES = {}


def register(kind):
    def deco(fn):
        SHAPES[kind] = fn
        return fn
    return deco


def bench_kind(kind, warmup=5, iters=20):
    """Run one registered kind. GPU absence or missing factory -> not-ok
    result with reason (caller ladders down / falls back; never guesses)."""
    factory = SHAPES.get(kind)
    if factory is None:
        return BenchResult(ok=False, error=f"no shape factory for kind={kind}",
                           note="unbenched")
    try:
        fn, note = factory()
        t0 = time.monotonic()
        ms = time_callable(fn, warmup=warmup, iters=iters)
        return BenchResult(ok=True, ms_per_iter=ms, iters=iters,
                           note=f"{note} (wall {time.monotonic()-t0:.1f}s)")
    except Exception as e:  # noqa: BLE001 -- any bench failure must degrade
        return BenchResult(ok=False, error=repr(e), note="bench-exception")


# ---------------------------------------------------------------------------
# v1 registered kinds (mock class only; real ones added per-op on-card)
# ---------------------------------------------------------------------------

register("mock_matmul")(_mock_matmul_factory)


def results_to_json(results):
    return {k: v.as_dict() for k, v in results.items()}
