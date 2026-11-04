#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# launch_g53f_tp4_pinned.sh — NUMA 绑核变体（2026-10-16）
# 映射经 torch PCI 总线 + 宿主 sysfs 核实（drm 卡号≠GPU 号！）：
#   TP0 GPU4(bus e8) -> node7 112-127 | TP1 GPU5(bus c1) -> node5 80-87
#   TP2 GPU6(bus ca) -> node5 88-95   | TP3 GPU7(bus b1) -> node4 64-79
# 辅助进程（主进程/detokenizer/tokenizer）-> node2 32-47（socket0，无本服务 GPU）
# 目标：宿主包提交/doorbell 写保持 socket 内局部性；TP0 免遭跨节点迁移。
/root/launch_g53f_tp4.sh &
MAIN=$!
declare -A CORES=( [TP0]=112-127 [TP1]=80-87 [TP2]=88-95 [TP3]=64-79 )
pinned=""
for i in $(seq 1 600); do
  for tp in TP0 TP1 TP2 TP3; do
    case $pinned in *"$tp"*) continue ;; esac
    pid=$(pgrep -f "scheduler_$tp" | head -1)
    if [ -n "$pid" ]; then
      taskset -pc ${CORES[$tp]} "$pid" >/dev/null 2>&1 && pinned="$pinned $tp=$pid"
      echo "[pin] $tp ($pid) -> ${CORES[$tp]}"
    fi
  done
  case $pinned in *TP0*|*:*) ;; esac
  [[ $pinned == *TP0* && $pinned == *TP1* && $pinned == *TP2* && $pinned == *TP3* ]] && break
  sleep 1
done
echo "[pin] schedulers done:$pinned"
sleep 20
# 辅助 python 进程（主进程/detokenizer 等，非 scheduler）钉到 socket0
taskset -pc 32-47 "$MAIN" >/dev/null 2>&1 && echo "[pin] main ($MAIN) -> 32-47"
for p in $(pgrep -f "sglang::" | grep -vf <(pgrep -f "scheduler_TP")); do
  taskset -pc 32-47 "$p" >/dev/null 2>&1 && echo "[pin] aux $p -> 32-47"
done
wait $MAIN
