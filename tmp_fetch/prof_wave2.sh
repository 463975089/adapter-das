#!/bin/bash
# P0' 终审 + R-20 serving 验证，一键链（窗口期省分钟）：
#   0) 独占复核（wave8 教训：server 在役时 min_free 应 ~30-40G；<20G=邻居还在，拒抓）
#   1) 16 路预热 -> start_profile 40 步 -> 16x600 temp0.7 负载
#   2) 等 trace 落盘 -> trace_census.py 出判决块
#   3) pkill 自家 sglang + 重起 launch_rt07（避开已知的抓后死锁）
REP=/root/window_report.txt
SRVLOG=/root/logs/srv_rt_1025.log
echo "=== prof_wave2 start $(date +%F' '%H:%M:%S)" >> $REP

FREE=$(python3 - <<'PY'
import torch
m = 999
for i in range(8):
    try:
        f, _ = torch.cuda.mem_get_info(i)
        m = min(m, f / 2**30)
    except Exception:
        pass
print(int(m))
PY
)
echo "exclusivity: min_free=${FREE}G (server up; >=20G == clean)" >> $REP
if [ "$FREE" -lt 20 ]; then
  echo "ABORT: neighbors present (${FREE}G) — refusing to prof a contaminated window" >> $REP
  exit 1
fi

rm -rf /root/prof_new && mkdir -p /root/prof_new
for i in $(seq 1 16); do
  curl -s -m 180 http://127.0.0.1:30296/generate -H 'Content-Type: application/json' \
    -d '{"text":"Warmup story number","sampling_params":{"temperature":0.7,"max_new_tokens":32,"ignore_eos":true}}' > /dev/null &
done
wait
echo WARM_DONE $(date +%H:%M:%S) >> $REP
T0=$(date +%s)
curl -s -m 10 http://127.0.0.1:30296/start_profile -H 'Content-Type: application/json' \
  -d '{"output_dir":"/root/prof_new","num_steps":40,"activities":["CPU","GPU"]}'
echo PROF_ARMED $(date +%H:%M:%S) >> $REP
for i in $(seq 1 16); do
  curl -s -m 900 http://127.0.0.1:30296/generate -H 'Content-Type: application/json' \
    -d '{"text":"Once upon a time in a distant land there lived","sampling_params":{"temperature":0.7,"max_new_tokens":600,"ignore_eos":true}}' > /dev/null &
done
wait
T1=$(date +%s)
echo LOAD_DONE $(date +%H:%M:%S) "wall=$((T1-T0))s tokens=16x600" >> $REP

for t in $(seq 1 30); do
  n=$(find /root/prof_new -name '*.trace.json.gz' 2>/dev/null | wc -l)
  d=$(grep -c 'Profiling done' $SRVLOG)
  if [ "$n" -ge 8 ] || [ "$d" -ge 1 ]; then break; fi
  sleep 20
done
find /root/prof_new -type f | head -8 >> $REP
grep -m2 'Profiling done' $SRVLOG >> $REP 2>/dev/null

python3 /root/trace_census.py >> $REP 2>&1
echo "=== prof_wave2 done $(date +%H:%M:%S) — restarting server (post-prof deadlock workaround)" >> $REP

pkill -f 'sglang.launch_server' ; sleep 12
nohup bash /root/launch_rt07.sh > /dev/null 2>&1 &
echo "server relaunched; WINDOW_CHAIN_COMPLETE" >> $REP
