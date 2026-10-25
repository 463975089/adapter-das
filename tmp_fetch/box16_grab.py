"""box16 stall census: warmup -> start_profile(40 steps) -> load -> stop -> census.
Mirrors the nmz P0' protocol: 16x32 warmup, profile 40 steps, 16x600 decode.
Outputs gap histogram + hipGraphLaunch tick histogram on TP-0 trace."""
import json, time, urllib.request, concurrent.futures, os, glob, gzip, collections, sys

HOST = "http://127.0.0.1:30900"
N_WARM, WARM_LEN = 16, 32
N_RUN, RUN_LEN = 16, 600

def post(path, payload=None, timeout=600):
    req = urllib.request.Request(HOST + path,
                                 data=json.dumps(payload or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()[:200]

def one(n, _=None):
    body = {
        "text": "Once upon a time there was a very long story about" + " word" * 8,
        "sampling_params": {"temperature": 0.7, "max_new_tokens": n,
                            "ignore_eos": True},
    }
    req = urllib.request.Request(HOST + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return len(r.read())

print("warmup", N_WARM, "x", WARM_LEN)
with concurrent.futures.ThreadPoolExecutor(N_WARM) as ex:
    list(ex.map(one, [WARM_LEN] * N_WARM))
print("warm done", time.strftime("%H:%M:%S"))

post("/start_profile", {"output_dir": "/root/prof_arm1", "num_steps": 40,
                        "activities": ["CPU", "GPU"]})
print("profile started", time.strftime("%H:%M:%S"))
with concurrent.futures.ThreadPoolExecutor(N_RUN) as ex:
    list(ex.map(one, [RUN_LEN] * N_RUN))
try:
    post("/stop_profile", timeout=120)
except Exception as e:
    print("stop_profile:", e)
print("profile stopped", time.strftime("%H:%M:%S"), "waiting for trace flush")
time.sleep(25)

# census on TP-0
cands = sorted(glob.glob("/root/prof_arm1/TP-0*.gz") + glob.glob("/root/prof_arm1/*TP-0*.gz"))
if not cands:
    cands = sorted(glob.glob("/root/prof_arm1/*.gz"))
print("traces:", cands)
if not cands:
    sys.exit("no trace files")
p = cands[0]
print("census on", p)
with gzip.open(p, "rt") as f:
    ev = json.load(f)["traceEvents"]
gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
gpu.sort(key=lambda x: x["ts"])
busy = []
for e in gpu:
    s, t = e["ts"], e["ts"] + e["dur"]
    if busy and s <= busy[-1][1]:
        busy[-1][1] = max(busy[-1][1], t)
    else:
        busy.append([s, t])
gaps = [(busy[i][0] - busy[i - 1][1]) / 1000 for i in range(1, len(busy))
        if busy[i][0] - busy[i - 1][1] > 15000]
h = collections.Counter()
for m in gaps:
    if m < 25: h["15-25ms"] += 1
    elif m < 40: h["25-40ms"] += 1
    elif m < 70: h["40-70ms(A-family)"] += 1
    elif m < 130: h["70-130ms"] += 1
    else: h[">130ms"] += 1
tot = sum(gaps)
print(f"GAPS>15ms: {len(gaps)} total={tot:.0f}ms  {dict(h)}")
gl = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"
      and e.get("name") in ("hipGraphLaunch", "cudaGraphLaunch")]
tick = [e for e in gl if e.get("dur", 0) > 15000]
fast = [e.get("dur", 0) / 1000 for e in gl if e.get("dur", 0) <= 15000]
print(f"graphLaunches n={len(gl)} tick(>15ms)={len(tick)} fast_p50={sorted(fast)[len(fast)//2] if fast else -1:.2f}ms")
if tick:
    print("  tick durs(ms):", sorted(f"{e.get('dur',0)/1000:.1f}" for e in tick)[:20])
print()
print("VERDICT: gaps~0 & tick~0  -> no stall on healthy box3 => nmz18 KFD degradation confirmed")
print("VERDICT: gaps or tick present -> driver-family-wide: bisect with GPU_MAX_HW_QUEUES/GPU_USE_SYNC_OBJECTS")
