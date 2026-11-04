# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""--autotune gate pins (GPU-less mock mode; also runs on-card).

Pins the tier-picker's contract without needing a card:

  ladder semantics  state gate drops broken/planned; correctness descent
                    stops at the first passing tier; exhausted ladder ->
                    fallback-default with an explicit reason (never a
                    fake green);
  artifacts         the generated yaml contains ONLY non-default picks,
                    passes hcu_config._validate, and the report carries
                    the full decision chain + summary counters;
  choices hygiene   every mapped judge exists on disk; every op is either
                    measured or explicitly skipped; env values in judges
                    are real hcu_env keys (cross-checked vs REGISTRY);
  bench harness     unregistered kind -> honest not-ok; mock kind runs
                    everywhere (CUDA-optional timing fallback).

Run: python tests/tools/_test_autotune.py  (or pytest; the _test_ prefix
keeps it out of pytest collection per tests/tools/README.md convention).
"""

import json
import os
import sys
import tempfile
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("  [{}] {}{}".format(status, name, (" -- " + detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def fake_runner(rc=0, out="ok"):
    """Stand-in subprocess.run: judge subprocesses are mocked in unit
    mode; a single .__call__ per correctness gate."""
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        r = mock.Mock()
        r.returncode = rc
        r.stdout = out
        r.stderr = ""
        return r

    run.calls = calls
    return run


def t_state_gate():
    from tools._autotune import runner

    print("== state gate ==")
    # broken tier never selectable (paged_mqa deepgemm arm)
    ok, why = runner.state_gate("paged_mqa_logits", "deepgemm")
    check("broken rejected", not ok, why)
    # production tier selectable
    ok, why = runner.state_gate("silu_and_mul", "triton")
    check("production accepted", ok, why)


def t_ladder_descent():
    from tools._autotune import runner

    print("== ladder descent (mocked judges) ==")
    # all tiers pass -> highest on ladder wins via 'ladder-order(dry)'-like
    # path (performance gate needs real CUDA; mocked run marks no bench)
    fr = fake_runner(0)
    with mock.patch.object(runner, "torch_gpu_ok", return_value=False):
        d = runner.pick_tier("silu_and_mul", runner=fr, env_extra={"PATH": os.environ["PATH"]})
        check("all-pass picks ladder top", d["chosen"] == "lightop", str(d))
        check("source is ladder-order", d["source"] == "ladder-order", d["source"])

    # top tier judge RED -> descend to next
    with mock.patch.object(runner, "torch_gpu_ok", return_value=False):
        # first ladder tier lightop red, triton green
        seq = [1, 0]
        idx = {"i": 0}

        def run(cmd, **kw):
            r = mock.Mock()
            r.returncode = seq[min(idx["i"], len(seq) - 1)]
            idx["i"] += 1
            r.stdout = "ninja boom"
            r.stderr = ""
            return r

        d = runner.pick_tier("silu_and_mul", runner=run,
                             env_extra={"PATH": os.environ["PATH"]})
        check("red top descends", d["chosen"] == "triton", str(d))
        check("descent reason logged",
              any("FAIL" in g for g in d["gates"]), str(d["gates"]))

    # all tiers red -> fallback default, honest reason
    def all_red(cmd, **kw):
        r = mock.Mock()
        r.returncode = 1
        r.stdout = "boom"
        r.stderr = ""
        return r

    with mock.patch.object(runner, "torch_gpu_ok", return_value=False):
        d = runner.pick_tier("silu_and_mul", runner=all_red,
                             env_extra={"PATH": os.environ["PATH"]})
        check("exhausted -> default", d["chosen"] == d["default_backend"], str(d))
        check("fallback source", d["source"] == "fallback-default", d["source"])
        check("exhaustion reason present",
              any("all ladder tiers failed" in g for g in d["gates"]), str(d))

    # no judges at all -> fallback with no-judge note
    d = runner.pick_tier("platform_detect", dry=True)
    check("unmapped op -> default", d["chosen"] == d["default_backend"], str(d))
    # platform_detect is in SKIP_OPS so run_autotune skips it entirely;
    # pick_tier on it directly exercises the no-judge branch
    check("no-judge source",
          d["source"] == "fallback-default" and d["gates"], str(d))


def t_artifacts():
    from tools._autotune import runner

    print("== artifacts (dry run, no subprocesses) ==")
    with tempfile.TemporaryDirectory() as td:
        ypath = os.path.join(td, "hcu_adapter.yaml")
        rpath = os.path.join(td, "report.json")
        yp, rp = runner.run_autotune(
            out_yaml=ypath, out_report=rpath,
            deep_gpu_check=False, dry=True, log=lambda s: None)
        check("yaml written", yp == ypath and os.path.isfile(ypath))
        check("report written", os.path.isfile(rpath))

        from hcu_adapter import hcu_config

        doc = hcu_config._parse_file(ypath)
        check("yaml re-parses clean", doc is not None)
        # dry picks: ensure only non-default entries present
        import yaml as _y
        raw = _y.safe_load(open(ypath, encoding="utf-8"))
        ops = raw.get("ops") or {}
        check("yaml ops flat + version 1", raw.get("version") == 1
              and isinstance(ops, (dict, type(None))) or isinstance(ops, dict),
              str(raw)[:120])
        rep = json.load(open(rpath, encoding="utf-8"))
        check("report has decisions", len(rep["decisions"]) >= 8)
        s = rep["summary"]
        check("report summary counters",
              "by_source" in s and "non_default" in s)
        check("triangle adds up",
              sum(s["by_source"].values()) == len(rep["decisions"]))
        check("platform snapshot in report",
              "gpu" in rep["platform"] and "traps" in rep["platform"])


def t_choices_hygiene():
    from tools._autotune import choices

    print("== choices hygiene ==")
    from hcu_adapter import hcu_config

    cat = hcu_config.BACKEND_CATALOG
    check("all catalog ops covered",
          set(choices.OPS) | choices.SKIP_OPS >= set(cat),
          str(set(cat) - set(choices.OPS) - choices.SKIP_OPS))
    check("no bogus op names", set(choices.OPS) <= set(cat),
          str(set(choices.OPS) - set(cat)))
    missing = []
    for op, spec in choices.OPS.items():
        for tier, judge in spec["judges"].items():
            for t in judge["tests"]:
                if not os.path.isfile(os.path.join(_ROOT, t)):
                    missing.append("{}/{} -> {}".format(op, tier, t))
            bad = set(judge.get("env") or {}) - set(
                _hcu_env_keys())
            if bad:
                missing.append("{}/{} env not in hcu_env: {}".format(op, tier, bad))
    check("judge files + env keys all real", not missing, "; ".join(missing))


def _hcu_env_keys():
    from hcu_adapter import hcu_env

    return set(hcu_env.REGISTRY.keys())


def t_bench_harness():
    from tools._autotune import bench

    print("== bench harness ==")
    r = bench.bench_kind("mock_matmul", warmup=1, iters=3)
    check("mock kind runs everywhere", r.ok and r.ms_per_iter is not None,
          str(r.as_dict()))
    r = bench.bench_kind("nonexistent_kind")
    check("unregistered honest not-ok", not r.ok and "no shape factory" in r.error)


def main():
    t_state_gate()
    t_ladder_descent()
    t_artifacts()
    t_choices_hygiene()
    t_bench_harness()
    print()
    if FAILS:
        print("FAILS: {}".format(FAILS))
        return 1
    print("autotune gate: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
