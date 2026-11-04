"""Four-stage precision triage for the HCU adapter (fast op localization).

When a run shows precision symptoms (garbled decode, divergence vs eager,
wrong tokens), run this INSIDE the compute container from the adapter root:

    python tests/tools/run_triage.py                 # env -> ops -> layers -> e2e
    python tests/tools/run_triage.py --stage ops     # one stage only
    python tests/tools/run_triage.py --full          # incl. 2-GPU AR stress
    python tests/tools/run_triage.py --url http://127.0.0.1:30000   # e2e server

Stages narrow the blast radius; every FAIL line names the suspected op,
the kill switch to flip (HCU_DISABLE_OPS, see kill_switch.py) and the
stress test that reproduces it:

  env    arch / wheels / kill-switch states — is the run even configured
         the way you think (gate active, emu on/off)?
  ops    operator-level: rope / rmsnorm / gemm sanity / graph-safety of the
         bridges / sgl_kernel triton suite / FA scenario matrix / AR gate.
  layers per-layer hidden-state alignment server-vs-pure-torch reference
         (needs dump artifacts; prints the exact recipe when missing).
  e2e    golden prompts (short + >index_topk long) against a live server;
         semantic anchor or last-known-good text comparison.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # tests/tools/ -> adapter root
sys.path.insert(0, ROOT)

RESULTS = []  # (stage, name, status, detail)


def record(stage, name, status, detail=""):
    RESULTS.append((stage, name, status, detail))
    mark = {"PASS": "ok", "FAIL": "FAIL", "SKIP": "skip", "INFO": "--"}[status]
    print(f"[{stage}] {mark:4} {name}" + (f" :: {detail}" if detail else ""),
          flush=True)


def run_probe(stage, name, script, predicate, timeout=600, env=None):
    e = dict(os.environ)
    e.setdefault("HCU_ADAPTER_ROOT", ROOT)
    if env:
        e.update(env)
    try:
        p = subprocess.run(
            [sys.executable, os.path.join(HERE, script)],
            capture_output=True, text=True, env=e, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        record(stage, name, "FAIL", f"timeout after {timeout}s")
        return None
    tail = (p.stdout + p.stderr).strip().splitlines()[-6:]
    if p.returncode != 0:
        record(stage, name, "FAIL", "exit=%d | %s"
               % (p.returncode, " / ".join(tail[-3:])))
        return p
    verdict, detail = predicate(p.stdout)
    record(stage, name, verdict, detail if verdict != "PASS"
           else detail or " | ".join(tail[-2:]))
    return p


# ---------------------------------------------------------------- env ----

def stage_env():
    import torch

    record("env", "torch", "INFO", torch.__version__)
    record("env", "hip", "INFO", str(torch.version.hip))
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
        record("env", "arch", "INFO", arch)
    except Exception as e:
        record("env", "arch", "FAIL", repr(e))
        return
    for lib in ("aiter", "lightop"):
        try:
            m = __import__(lib)
            record("env", lib, "INFO",
                   getattr(m, "__version__", getattr(m, "__file__", "?")))
        except Exception as e:
            record("env", lib, "INFO", f"not importable ({e!r:.60})")
    from hcu_adapter import kill_switch

    off = sorted(kill_switch.disabled_ops())
    record("env", "kill-switches", "INFO",
           "ACTIVE: " + ",".join(off) if off else "none (HCU_DISABLE_OPS)")
    hu = {k: v for k, v in os.environ.items() if k.startswith("HCU_")}
    if hu:
        record("env", "HCU-env", "INFO", json.dumps(hu))


# ---------------------------------------------------------------- ops ----

def _min_cos_pred(threshold=0.9999):
    def pred(out):
        # only the parity metrics (`cos=`); probes also print negative
        # controls (`cos_neox=...`, deliberately ~0.9) that must NOT count
        vals = [float(m) for m in re.findall(r"\bcos=([0-9.]+)", out)]
        if not vals:
            return "FAIL", "no cos metrics found in probe output"
        worst = min(vals)
        return ("PASS" if worst >= threshold else "FAIL",
                f"min cos={worst:.6f}")

    return pred


def _gemm_pred(out):
    m = re.search(r"excess vs floor: ([0-9.]+)x", out)
    if not m:
        return "FAIL", "no excess metric found"
    x = float(m.group(1))
    return ("PASS" if x <= 1.05 else "FAIL", f"excess={x}x (bf16 floor ~1x)")


def _contains_pred(token):
    return lambda out: ("PASS", "") if token in out else \
        ("FAIL", f"expected {token!r} in output")


def stage_ops(full):
    run_probe("ops", "rope-parity", "_rope_probe.py", _min_cos_pred())
    run_probe("ops", "rmsnorm-bridge", "_rms_probe.py", _min_cos_pred())
    run_probe("ops", "gemm-bf16-sanity", "_gemm_sanity.py", _gemm_pred)
    run_probe("ops", "bridge-graph-safety", "_graph_safety_probe.py",
              _contains_pred("none — bridges are graph-safe"))
    _pytest("ops", "sgl_kernel-triton-suite",
            ["tests/kernels", "-q"], timeout=1200)
    _pytest("ops", "fa-scenario-matrix",
            ["tests/vendor/test_flash_attn.py", "-q", "--no-header"],
            timeout=1800)
    _pytest("ops", "ar-gate",
            ["tests/vendor/test_allreduce.py", "-q", "--no-header",
             "-k", "gate"], timeout=900)
    if full:
        _pytest("ops", "ar-stress-2gpu",
                ["tests/vendor/test_allreduce.py", "-q", "--no-header",
                 "-k", "stress"], timeout=1200)


def _pytest(stage, name, args, timeout=900):
    cmd = [sys.executable, "-m", "pytest"] + args
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=ROOT, timeout=timeout)
    except subprocess.TimeoutExpired:
        record(stage, name, "FAIL", f"timeout after {timeout}s")
        return
    output = p.stdout + p.stderr
    # Preserve the whole pytest summary: a substring beginning at "passed"
    # drops preceding failures, making a real FAIL look like a green suite.
    summaries = re.findall(
        r"^\s*(?:=+\s*)?\d+ (?:failed|passed|skipped|error|deselected|xfailed|xpassed)"
        r"\b[^\n]*", output, flags=re.MULTILINE,
    )
    summ = summaries[-1].strip(" =\r") if summaries else " | ".join(
        output.strip().splitlines()[-2:])
    record(stage, name, "PASS" if p.returncode == 0 else "FAIL", summ)
    if p.returncode:
        # Keep the traceback and exact failing node IDs in the triage log.
        print(output, flush=True)


# ------------------------------------------------------------- layers ----

def stage_layers():
    dump = os.environ.get("HCU_DUMP_LAYERS_PATH", "/tmp/hcu_layers.pt")
    ref = os.path.join(HERE, "_ref_hiddens.pt")
    if not (os.path.exists(dump) and os.path.exists(ref)):
        record("layers", "artifacts", "SKIP",
               "need both files; recipe below")
        print(
            "\nrecipe (12-layer truncated model, container):\n"
            "  1) reference: python tests/tools/_ref_forward_h.py   "
            "(NL_OVERWRITE=12; writes tests/tools/_ref_hiddens.pt)\n"
            "  2) server:    bash tests/tools/launch_w4a8_12l.sh    "
            "(HCU_DUMP_LAYERS=1 -> /tmp/hcu_layers.pt)\n"
            "  3) one short /generate request (T<=64, fresh prompt),\n"
            "  4) rerun:     python tests/tools/run_triage.py --stage layers\n",
            flush=True,
        )
        return
    p = run_probe("layers", "layer-align", "_bisect_cmp.py",
                  lambda out: ("INFO", "see DIVERGE lines below"),
                  timeout=300)
    if p:
        for ln in p.stdout.splitlines():
            if "DIVERGE" in ln or "first" in ln.lower():
                print("   " + ln, flush=True)


# ---------------------------------------------------------------- e2e ----

GOLDEN = os.path.join(HERE, "golden", "e2e_greedy.json")
SHORT_PROMPT = "The distance from London to Paris is"
LONG_PROMPT = (
    "Paris is the capital of France. The Seine river divides the city "
    "into the Left Bank and the Right Bank. " * 260
)  # ~2700 tokens > index_topk 2048 -> sparse decode variant


def _generate(url, prompt, max_tokens=16):
    req = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps({
            "text": prompt, "sampling_params": {
                "max_new_tokens": max_tokens, "temperature": 0.0,
            },
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["text"]


def stage_e2e(update_golden):
    url = os.environ.get("HCU_TRIAGE_URL", "http://127.0.0.1:30000")
    golden = {}
    if os.path.exists(GOLDEN) and not update_golden:
        golden = json.load(open(GOLDEN, encoding="utf-8"))
    try:
        outs = {
            "short": _generate(url, SHORT_PROMPT),
            "long>topk": _generate(url, LONG_PROMPT),
        }
    except Exception as e:
        record("e2e", "server", "SKIP", f"{url} unreachable: {e!r:.120}")
        return
    for k, text in outs.items():
        text = text[len(SHORT_PROMPT):] if k == "short" else text
        anchor_ok = "Paris" in text if k == "short" else True
        if update_golden or k not in golden:
            golden[k] = text
            record("e2e", k, "INFO", f"recorded: {text[:60]!r}")
            continue
        if golden[k] == text:
            record("e2e", k, "PASS", f"{text[:60]!r}")
        elif anchor_ok:
            record("e2e", k, "FAIL",
                   f"text drifted vs golden: {text[:60]!r} != "
                   f"{golden[k][:60]!r} (bf16 reordering? check layers)")
        else:
            record("e2e", k, "FAIL", f"semantic anchor lost: {text[:80]!r}")
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    with open(GOLDEN, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=1)


# ------------------------------------------------------------- verdict ----

HINTS = {
    "rope-parity": "bug class fbc540c: same-name != same-semantics; check"
                   " tests/tools/_rope_probe.py vs sglang cache layout",
    "bridge-graph-safety": "capture-baked host state / H2D; see"
                           " tests/tools/_graph_safety_probe.py",
    "fa-scenario-matrix": "vendor FA: try HCU_DISABLE_OPS=aiter_fa,"
                          " rerun; stress: tests/vendor/test_flash_attn.py",
    "ar-gate": "gate must refuse on gfx93*; HCU_AITER_AR=1 opts back in",
    "ar-stress-2gpu": "vendor AR: HCU_DISABLE_OPS=aiter_ar (NCCL fallback);"
                      " stress: tests/vendor/test_allreduce.py",
    "layer-align": "first DIVERGE layer +1 locates the op; cross-check"
                   " tests/tools/_inner_cmp.py for xa/xb",
}


def verdict():
    fails = [r for r in RESULTS if r[2] == "FAIL"]
    print("\n" + "=" * 72)
    for stage, name, status, detail in RESULTS:
        if status == "FAIL":
            hint = HINTS.get(name, "")
            print(f"FAIL {stage}/{name}: {detail}")
            if hint:
                print(f"     -> {hint}")
    if not fails:
        npass = sum(1 for r in RESULTS if r[2] == "PASS")
        print(f"ALL GREEN ({npass} passed, "
              f"{sum(1 for r in RESULTS if r[2] == 'SKIP')} skipped)")
    print("=" * 72)
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    help="comma-separated subset of env,ops,layers,e2e or all")
    ap.add_argument("--full", action="store_true",
                    help="include 2-GPU AR stress")
    ap.add_argument("--url", help="server base URL for e2e")
    ap.add_argument("--update-golden", action="store_true")
    a = ap.parse_args()
    stages = [s.strip() for s in a.stage.split(",")]
    if "all" in stages:
        stages = ["env", "ops", "layers", "e2e"]
    bad = [s for s in stages if s not in ("env", "ops", "layers", "e2e")]
    if bad:
        ap.error(f"invalid stage(s) {bad}; choose from env,ops,layers,e2e,all")
    if a.url:
        os.environ["HCU_TRIAGE_URL"] = a.url

    t0 = time.time()
    if "env" in stages:
        stage_env()
    if "ops" in stages:
        stage_ops(a.full)
    if "layers" in stages:
        stage_layers()
    if "e2e" in stages:
        stage_e2e(a.update_golden)
    print(f"\n({time.time() - t0:.0f}s)")
    raise SystemExit(verdict())


if __name__ == "__main__":
    main()
