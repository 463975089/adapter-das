#!/bin/bash
# box16 arm runner: $1 = env assignments (quoted), $2 = tag
set -u
ENVEXTRA="${1:-}"
TAG="${2:-arm}"
export TAGX="${TAG}"
rm -rf /root/prof_${TAG}

# kill previous server (our own container's)
pkill -f "sglang.launch_server" 2>/dev/null
sleep 5

cd /root
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TOPO_FILE=
if [ -n "$ENVEXTRA" ]; then
  echo "ARM env: $ENVEXTRA"
  export $ENVEXTRA
fi
nohup python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-V4.1-Flash-Channel-FP8 \
  --served-model-name dsv41f \
  --trust-remote-code \
  --host 0.0.0.0 --port 30900 \
  --tp-size 8 --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
  --ep-size 8 --moe-a2a-backend none --moe-runner-backend triton \
  --page-size 64 --mem-fraction-static 0.80 \
  --chunked-prefill-size 8192 --max-running-requests 16 \
  --cuda-graph-max-bs-decode 16 \
  --disable-radix-cache \
  --nccl-port 30901 --dist-timeout 10000 --watchdog-timeout 1500 \
  > /root/logs_srv_${TAG}.log 2>&1 &
echo "ARM ${TAG} FIRED env=[${ENVEXTRA}] $(date +%H:%M:%S)"

for i in $(seq 1 80); do
  sleep 10
  if grep -q "The server is fired up" /root/logs_srv_${TAG}.log 2>/dev/null; then
    echo "UP after $((i*10))s"; break
  fi
  if ! pgrep -f "sglang.launch_server" >/dev/null; then
    echo "DIED"; tail -5 /root/logs_srv_${TAG}.log; exit 1; fi
  if [ $i -eq 80 ]; then echo "TIMEOUT"; exit 1; fi
done

python3 - <<'PYEOF'
import json, time, urllib.request, concurrent.futures, os
TAG = os.environ["TAGX"]
HOST = "http://127.0.0.1:30900"
def one(n):
    body = {"text": "Once upon a time" + " word" * 8,
            "sampling_params": {"temperature": 0.7, "max_new_tokens": n,
                                "ignore_eos": True}}
    req = urllib.request.Request(HOST + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return len(r.read())
print("single warm...", flush=True)
one(8)
print("single done", flush=True)
with concurrent.futures.ThreadPoolExecutor(8) as ex:
    list(ex.map(one, [32] * 8))
print("warm8 done", flush=True)
req = urllib.request.Request(HOST + "/start_profile",
                             data=json.dumps({"output_dir": "/root/prof_" + TAG,
                                              "num_steps": 20,
                                              "activities": ["CPU", "GPU"]}).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    r.read()
print("profile on", flush=True)
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(8) as ex:
    list(ex.map(one, [200] * 8))
print(f"run done in {time.time()-t0:.1f}s", flush=True)
PYEOF
echo "waiting for trace flush..."
sleep 30
python3 - <<'PYEOF'
import gzip, json, glob, collections, os
TAG = os.environ["TAGX"]
cands = sorted(glob.glob(f"/root/prof_{TAG}/*TP-0*"))
if not cands:
    print("NO TRACE"); raise SystemExit
p = cands[0]
print("census on", p, flush=True)
ev = json.load(gzip.open(p, "rt"))["traceEvents"]
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
    elif m < 70: h["40-70ms(A)"] += 1
    elif m < 130: h["70-130ms"] += 1
    else: h[">130ms"] += 1
print(f"GAPS>15ms: {len(gaps)} {dict(h)}", flush=True)
gl = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"
      and "GraphLaunch" in e.get("name", "")]
durs = sorted(e.get("dur", 0) / 1000 for e in gl)
tick = [d for d in durs if d > 15]
print(f"graphLaunches n={len(gl)} tick(>15ms)={len(tick)} "
      f"p50={durs[len(durs)//2] if durs else -1:.2f}ms max={durs[-1] if durs else -1:.2f}ms", flush=True)
print(f"VERDICT[{TAG}]: tick_frac={len(tick)}/{len(gl)}", flush=True)
PYEOF
echo "ARM ${TAG} COMPLETE $(date +%H:%M:%S)"
