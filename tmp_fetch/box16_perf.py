#!/usr/bin/env python3
"""box16 perf accounting: step time / GPU busy / tick share from saved
traces + decode throughput lines from arm logs. Zero side effects."""
import gzip, json, re, sys, glob, os


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def analyze(trace_path):
    ev = load(trace_path).get("traceEvents", [])
    launches = [
        e for e in ev
        if e.get("cat") == "cuda_runtime" and "GraphLaunch" in e.get("name", "")
    ]
    kernels = [e for e in ev if e.get("cat") == "kernel"]
    if not launches:
        print(f"PERF {os.path.basename(trace_path)[:40]}: no graph launches")
        return
    t0 = min(e["ts"] for e in launches)
    t1 = max(e["ts"] + e.get("dur", 0) for e in launches)
    span = (t1 - t0) / 1e6
    n = len(launches)
    lspan = sum(e.get("dur", 0) for e in launches) / 1e3
    # GPU busy = merged kernel intervals
    iv = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in kernels)
    busy = 0.0
    cs, ce = None, None
    for s, e2 in iv:
        if cs is None:
            cs, ce = s, e2
        elif s <= ce:
            ce = max(ce, e2)
        else:
            busy += ce - cs
            cs, ce = s, e2
    if cs is not None:
        busy += ce - cs
    busy_s = busy / 1e6
    print(f"PERF {os.path.basename(trace_path)[:46]}")
    print(f"  window={span:.2f}s launches={n} step_ms~{span/max(n,1)*1000:.1f}"
          f" launchAPI_ms_sum={lspan:.0f} ({lspan/1000/max(span,1e-9)*100:.0f}% of window)")
    print(f"  GPU busy={busy_s:.2f}s util={busy_s/max(span,1e-9)*100:.0f}%"
          f" kernels={len(kernels)}")


def log_thru(log_path):
    thr, steps = [], 0
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except OSError:
        return
    for m in re.finditer(
        r"Decode batch.*?#running-req:\s*(\d+).*?gen throughput \(token/s\):\s*([\d.]+)", txt
    ):
        thr.append((int(m.group(1)), float(m.group(2))))
    if thr:
        tail = thr[-8:]
        print(f"LOG {os.path.basename(log_path)}: decode lines={len(thr)} "
              f"tail(bs,tok/s)={tail}")


dirs = sorted(glob.glob("/root/prof*"))
for d in dirs:
    for tp0 in sorted(glob.glob(d + "/*TP-0*.trace.json.gz"))[:1]:
        analyze(tp0)
print("===LOGS===")
for lg in sorted(glob.glob("/root/*.log") + glob.glob("/root/nohup*")):
    log_thru(lg)
