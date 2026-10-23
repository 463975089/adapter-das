#!/bin/bash
# Fire launch_rt07.sh when all 8 GPUs have >=120G free twice in a row.
LOG=/root/boot_watch.log
echo "watch start $(date +%H:%M:%S)" >> $LOG
ok=0
for i in $(seq 1 240); do
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
  echo "$(date +%H:%M:%S) min_free=${min_free}G streak=$ok" >> $LOG
  if [ "$ok" -ge 2 ]; then
    echo "WINDOW OPEN -> firing launch_rt07" >> $LOG
    bash /root/launch_rt07.sh &
    exit 0
  fi
  sleep 30
done
echo "watch gave up after 2h" >> $LOG
