"""--autotune runner: three-gate tier picking + the two artifacts.

Per op, backends walk the op's ladder (choices.py) through three gates:

    state gate     catalog status must be selectable (broken/planned out)
   正确性 gate     the registered judge test passes under a child process
                   with the tier's env override (explicit-env wins the
                   adapter's own裁决链, so routing is proven, not assumed)
    性能 gate      among tiers that passed correctness: same-window A/B
                   microbench, fastest wins

Fail a gate -> descend the ladder; exhaust it -> keep the catalog default
and say so ("no-judge fallback" when no judge existed at all). Nothing
here ever guesses a verdict it did not measure.

Artifacts: autotune_report.json (auditable decision chain) and a
minimal hcu_adapter.yaml containing only non-default picks, validated in
the same process via hcu_config._validate before handoff.
"""

import json
import os
import pathlib
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools._autotune import bench as bench_mod  # noqa: E402
from tools._autotune import choices  # noqa: E402
from tools._autotune import probe as probe_mod  # noqa: E402

_SELECTABLE = ("production", "validated", "candidate")


# ----------------------------------------------------------------- gates

def state_gate(op, backend):
    from hcu_adapter import hcu_config

    meta = hcu_config.BACKEND_CATALOG[op]["backends"].get(backend)
    if meta is None:
        return False, "backend not in catalog"
    if meta["status"] not in _SELECTABLE:
        return False, "status={} ({}-authored gate)".format(
            meta["status"], "broken/planned never auto-picked")
    return True, "status={}".format(meta["status"])


def correctness_gate(op, backend, judge, timeout_s=900, env_extra=None,
                     runner=None):
    """Run the judge in a child python. PYTEST_BIN lets gates inject a
    runner; default resolves pytest or falls back to direct execution."""
    env = {**os.environ, **(judge.get("env") or {}), **(env_extra or {})}
    stamp = ";".join("{}={}".format(k, v) for k, v in sorted((judge.get("env") or {}).items()))
    cmds = []
    if judge["mode"] == "pytest":
        pytest_bin = _which_pytest()
        if pytest_bin:
            cmds.append([pytest_bin, "-q", "--no-header", "-x", *judge["tests"]])
        else:
            cmds.append([sys.executable, "-m", "pytest", "-q", "--no-header",
                         "-x", *judge["tests"]])
    else:  # std standalone: each test executed directly
        cmds = [[sys.executable, t] for t in judge["tests"]]

    for cmd in cmds:
        try:
            r = (runner or subprocess.run)(
                cmd, env=env, capture_output=True, text=True,
                timeout=timeout_s, cwd=_ROOT,
            )
        except subprocess.TimeoutExpired:
            return False, "judge timeout ({}s): {}".format(timeout_s, cmd[-1])
        tail = (r.stdout or "")[-400:]
        if r.returncode == 0:
            return True, "PASS [{} {}]".format(cmd[0], " ".join(cmd[1:]))
        why = "rc={}".format(r.returncode)
        if judge.get("expect_fail"):
            # a loud-fail sentinel (e.g. quant aiter stock arm) — red here
            # is the CORRECT answer for this tier
            return True, "expected-fail confirmed [{}]".format(tail[-120:])
        return False, "FAIL {} :: {} | env=[{}]".format(why, tail[-200:], stamp)
    return False, "no judge command constructed"


def _which_pytest():
    from shutil import which
    return which("pytest")


def performance_gate(op, passing, bench_runs, kind=None, warmup=5, iters=20):
    """Same-window A/B among correctness-passing tiers. bench_runs: dict
    filled with per-tier results for the report."""
    kind = kind or choices.bench_kind(op)
    if not kind:
        return None, "no microbench registered for op (choice by ladder order)"
    if not torch_gpu_ok():
        return None, "no CUDA in this process (bench skipped)"
    for backend in passing:
        factory = bench_mod.SHAPES.get("{}:{}".format(kind, backend))
        if factory is None:
            bench_runs[backend] = {"ok": False, "note": "no {} factory".format(kind)}
            continue
        res = bench_mod.bench_kind("{}:{}".format(kind, backend),
                                   warmup=warmup, iters=iters)
        bench_runs[backend] = res.as_dict()
    measured = {b: r for b, r in bench_runs.items() if r.get("ok")}
    if len(measured) < 2:
        return (list(measured)[0] if measured else None), (
            "insufficient measured tiers (need >=2 for A/B)")
    return min(measured, key=lambda b: measured[b]["ms_per_iter"]), (
        "measured A/B among {}".format(sorted(measured)))


def torch_gpu_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ------------------------------------------------------------- per-op flow

def pick_tier(op, dry=False, runner=None, env_extra=None):
    """Returns the decision dict for one op. `dry=True` consults the
    ladder/catalog only (no subprocesses) -- gates/mock mode."""
    from hcu_adapter import hcu_config

    catspec = hcu_config.BACKEND_CATALOG[op]
    decision = {
        "op": op,
        "default_backend": _default_backend(op),
        "ladder": choices.tier_order(op),
        "gates": [],
        "chosen": None,
        "source": None,   # measured | ladder-order | fallback-default
        "bench": {},
    }

    judge_tiers = [t for t in decision["ladder"]
                   if choices.judge_for(op, t)]
    if not judge_tiers:
        decision["chosen"] = decision["default_backend"]
        decision["source"] = "fallback-default"
        decision["gates"].append("no judge registered for any ladder tier")
        return decision

    passing = []
    for tier in judge_tiers:
        if not dry:
            ok, why = state_gate(op, tier)
            if not ok:
                decision["gates"].append("state[{}] {}".format(tier, why))
                continue
            judge = choices.judge_for(op, tier)
            ok, why = correctness_gate(op, tier, judge, runner=runner,
                                       env_extra=env_extra)
            decision["gates"].append("correctness[{}] {}".format(tier, why))
            if not ok:
                continue
        passing.append(tier)

    if not passing:
        decision["chosen"] = decision["default_backend"]
        decision["source"] = "fallback-default"
        decision["gates"].append("all ladder tiers failed; keep default")
        return decision

    if dry:
        decision["chosen"] = passing[0]
        decision["source"] = "ladder-order(dry)"
        return decision

    winner, why = performance_gate(op, passing, decision["bench"])
    decision["gates"].append("performance: " + why)
    if winner is None:
        # nothing measured (or single survivor): fall through by ladder
        decision["chosen"] = passing[0]
        decision["source"] = "ladder-order"
    else:
        decision["chosen"] = winner
        decision["source"] = "measured"
    return decision


def _default_backend(op):
    from hcu_adapter import hcu_config

    spec = hcu_config.BACKEND_CATALOG[op]
    for b, meta in spec["backends"].items():
        if meta["status"] == "production":
            return b
    return None


# --------------------------------------------------------------- artifacts

def write_yaml(picks, path):
    """Minimal config: only non-default picks. Raises if the produced doc
    fails hcu_config._validate (closed loop before handoff)."""
    from hcu_adapter import hcu_config

    doc = {"version": 1}
    ops = {}
    for op, d in sorted(picks.items()):
        if d["chosen"] and d["chosen"] != d["default_backend"]:
            ops[op] = d["chosen"]
    doc["ops"] = ops
    text = ["# generated by hcu_config_wizard --autotune on {}".format(
        time.strftime("%Y-%m-%d %H:%M:%S")),
        "# decision chain: see the sibling autotune_report.json", ""]
    if ops:
        text.append("ops:")
        for op, backend in ops.items():
            text.append("  {}: {}".format(op, backend))
    else:
        text.append("ops: {}  # all measured picks equal defaults")
    pathlib.Path(path).write_text("\n".join(text) + "\n", encoding="utf-8")

    # closed loop: the file must pass the SAME validator the adapter
    # uses at boot. _parse_file+load path is stateful; validate raw doc.
    hcu_config._validate(doc, str(path))
    return path


def _summarize(picks):
    by = {}
    for d in picks.values():
        by[d["source"]] = by.get(d["source"], 0) + 1
    s = {
        "by_source": by,
        "non_default": sorted(op for op, d in picks.items()
                              if d["chosen"] != d["default_backend"]),
    }
    return s


def write_report(pic, picks, path):
    doc = {
        "schema": "hcu-autotune-report/1",
        "platform": pic,
        "decisions": [picks[op] for op in sorted(picks)],
        "summary": _summarize(picks),
    }
    pathlib.Path(path).write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


# ------------------------------------------------------------------- main

def run_autotune(out_yaml="hcu_adapter.yaml", out_report="autotune_report.json",
                 deep_gpu_check=True, dry=False, ops_filter=None, runner=None,
                 env_extra=None, log=print):
    log("== autotune: platform probe ==")
    pic = probe_mod.probe(deep_gpu_check=deep_gpu_check)
    log(probe_mod.format_pic(pic))
    if pic["aborted"]:
        log("autotune: ABORT (fix the FAIL rows first)")
        return None, None

    op_list = [op for op in choices.measured_ops()
               if not ops_filter or op in ops_filter]
    log("== autotune: {} measurable ops ==".format(len(op_list)))
    picks = {}
    for op in op_list:
        d = pick_tier(op, dry=dry, runner=runner, env_extra=env_extra)
        picks[op] = d
        bench_note = ""
        if d["bench"]:
            best = min((r.get("ms_per_iter", 9e9), b)
                       for b, r in d["bench"].items() if r.get("ok"))
            bench_note = " {:.3f}ms".format(best[0])
        log("  {:<18} -> {:<10} [{}]{}  {}".format(
            op, d["chosen"], d["source"], bench_note,
            "; ".join(d["gates"][-1:])))

    ypath = write_yaml(picks, out_yaml)
    rpath = write_report(pic, picks, out_report)
    log("autotune: yaml -> {}  report -> {}".format(ypath, rpath))
    s = json.load(open(rpath, encoding="utf-8"))["summary"]
    log("autotune: by_source={} non-default={}".format(
        s["by_source"], s["non_default"]))
    return ypath, rpath
