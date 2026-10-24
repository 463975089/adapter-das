#!/bin/bash
# W13: one-shot post-reboot verification for nmz18.
# Usage: bash w13_postreboot.sh [tag]
# Chain: card sanity -> launch rt07 recipe -> warmup -> 40-step profile ->
#        census (gap histogram + graph-launch histogram) -> verdict.
set -u
TAG="${1:-w13}"
LOG=/root/logs
mkdir -p $LOG

echo "[$(date +%H:%M:%S)] == 1) card sanity =="
python3 /root/probe_cards.py 2>/dev/null || echo "probe_cards.py missing (push it)"
echo "expect: all 8 cards ~140G free. If still ~29G -> KFD still sick, STOP."

read -p "cards clean? press Enter to launch, Ctrl-C to abort" _

echo "[$(date +%H:%M:%S)] == 2) launch rt07 =="
cd /root
nohup bash /root/launch_rt07.sh > $LOG/launch_${TAG}.log 2>&1 &
echo "fired; poll boot:"

for i in $(seq 1 60); do
  sleep 10
  if grep -q "The server is fired up" $LOG/srv_rt07.log 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] server UP"; break
  fi
  if ! pgrep -f "sglang.launch_server" >/dev/null; then
    echo "LAUNCH DIED — tail:"; tail -20 $LOG/srv_rt07.log; exit 1
  fi
done

echo "[$(date +%H:%M:%S)] == 3) warmup + 40-step profile =="
bash /root/grab_q4.sh ${TAG} || { echo "grab failed"; exit 1; }

echo "[$(date +%H:%M:%S)] == 4) census =="
python3 /root/trace_census.py 2>/dev/null | tail -40

echo "[$(date +%H:%M:%S)] == 5) stall verdict =="
python3 - <<'PYEOF'
import gzip, json, glob, collections
paths = sorted(glob.glob("/root/prof_out/${TAG}/TP-*.trace.json.gz")) or \
        sorted(glob.glob("/root/${TAG}/TP-*.trace.json.gz"))
if not paths:
    print("no traces found — check grab_q4.sh output dir"); raise SystemExit
p = paths[0]
print("census on", p)
with gzip.open(p, "rt") as f:
    ev = json.load(f)["traceEvents"]
gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")]
gpu.sort(key=lambda x: x["ts"])
busy = []
for e in gpu:
    s, t = e["ts"], e["ts"] + e["dur"]
    if busy and s <= busy[-1][1]:
        busy[-1][1] = max(busy[-1][1], t)
    else:
        busy.append([s, t])
gaps = [(busy[i][0] - busy[i-1][1]) / 1000 for i in range(1, len(busy))
        if busy[i][0] - busy[i-1][1] > 15000]
h = collections.Counter()
for m in gaps:
    if m < 25: h["15-25ms"] += 1
    elif m < 40: h["25-40ms"] += 1
    elif m < 70: h["40-70ms(A-family)"] += 1
    elif m < 130: h["70-130ms"] += 1
    else: h[">130ms(B-family)"] += 1
print(f"GAPS>15ms: {len(gaps)}  {dict(h)}")
gl = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"
      and e.get("name") == "hipGraphLaunch"]
tick = sum(1 for e in gl if e.get("dur", 0) > 15000)
print(f"hipGraphLaunch n={len(gl)} tick(>15ms)={tick}")
print()
print("VERDICT: gaps ~0 and tick ~0  -> KFD degradation was the root. CLOSE.")
print("VERDICT: gaps/tick persist    -> driver-version bug; next arms:")
print("   GPU_MAX_HW_QUEUES=16, GPU_MAX_HW_QUEUES=32, GPU_USE_SYNC_OBJECTS=1")
PYEOF
echo "[$(date +%H:%M:%S)] done."
