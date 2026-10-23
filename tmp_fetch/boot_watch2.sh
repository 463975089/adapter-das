#!/bin/bash
# boot_watch v2：同 v1（8 卡 >=120G 连续 2 次 -> launch_rt07），加防重触发：
# 服务已在役（进程/健康口）则不点火。v1 到期后由本脚本接力。
LOG=/root/boot_watch.log
echo "watch2 start $(date +%H:%M:%S)" >> $LOG
ok=0
for i in $(seq 1 320); do
  if pgrep -f 'sglang.launch_server' > /dev/null 2>&1; then
    echo "$(date +%H:%M:%S) server already up (watch2 idle)" >> $LOG
    exit 0
  fi
  min_free=$(python3 - <<'PY'
import torch
m = 99
for i in range(8):
    try:
        free, _ = torch.cuda.mem_get_info(i)
        m = min(m, free / 2**30)
    except Exception:
        pass
print(int(m))
PY
)
  if [ "$min_free" -ge 120 ]; then
    ok=$((ok+1))
  else
    ok=0
  fi
  echo "$(date +%H:%M:%S) min_free=${min_free}G streak=$ok (w2)" >> $LOG
  if [ "$ok" -ge 2 ]; then
    echo "WINDOW OPEN (w2) -> firing launch_rt07" >> $LOG
    bash /root/launch_rt07.sh &
    exit 0
  fi
  sleep 30
done
echo "watch2 gave up $(date +%H:%M:%S)" >> $LOG
