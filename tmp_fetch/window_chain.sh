#!/bin/bash
# 窗口链轮询器：boot_watch 已点火 launch_rt07 后，本脚本盯 'fired up'，
# 图捕获再等 150s，然后自动跑 prof_wave2（终审+R-20 验证+自动重启）。
# 只动自己容器里自己起的服务（pkill 范围=sglang.launch_server）。
# 注：launch_rt07 每次启动都截断 srv_rt_1025.log，故全量 grep 安全。
SRVLOG=/root/logs/srv_rt_1025.log
MARK=/root/window_report.txt
echo "window_chain poller armed $(date +%F' '%H:%M:%S)" >> $MARK
for i in $(seq 1 480); do
  if grep -q 'The server is fired up' $SRVLOG 2>/dev/null; then
    echo "fired up detected $(date +%H:%M:%S); settling 150s" >> $MARK
    sleep 150
    bash /root/prof_wave2.sh
    exit $?
  fi
  sleep 30
done
echo "window_chain: 4h no boot — exiting" >> $MARK
