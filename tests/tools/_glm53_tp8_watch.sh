#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# GLM-5.3-Flash bf16 TP8: wait for all-8-card window, launch + four-gate.
# Priority insert per user: runs as soon as 8 cards are free. Single-card
# validations on cards 0-3 (par suite) are expected to finish well before.
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:$LD_LIBRARY_PATH
PORT=30053
LOGD=/root/logs
export PYTHONPATH=/root/hcu_glm:/sglang-v0521/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_RUST_BUILD_MODE=never
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
MODEL=/models_home/GLM-5.3-Flash

free_card() {
  local out t u
  out=$(/opt/dtk/bin/rocm-smi --showmeminfo vram 2>/dev/null)
  t=$(echo "$out" | grep -A1 "HCU\[$1\]" | grep "Total Memory" | head -1 | tr -d ' ' | grep -o '[0-9]*$')
  u=$(echo "$out" | grep -A2 "HCU\[$1\]" | grep "Used Memory" | head -1 | tr -d ' ' | grep -o '[0-9]*$')
  echo $(( (t - u) / 1024 ))
}

echo "[glm53] waiting for all-8-card window (>=42GiB each)..."
while true; do
  ok=1
  for c in 0 1 2 3 4 5 6 7; do
    f=$(free_card $c)
    if [ "${f:-0}" -lt 42000 ]; then ok=0; break; fi
  done
  [ "$ok" = "1" ] && break
  sleep 60
done
echo "[glm53] WINDOW OPEN, launching TP8 page256"
pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 8
nohup python -m sglang.launch_server \
  --model-path $MODEL --host 0.0.0.0 --port $PORT \
  --tp-size 8 --page-size 256 --mem-fraction-static 0.85 \
  --trust-remote-code --log-level info \
  > $LOGD/launch_glm53.log 2>&1 &

t=0
while [ $t -lt 2700 ]; do
  curl -s --max-time 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "[glm53] G1-launch PASS"; break; }
  pgrep -f sglang.launch_server >/dev/null || { echo "[glm53] G1-launch DIED"; tail -30 $LOGD/launch_glm53.log; exit 2; }
  sleep 15; t=$((t + 15))
done
[ $t -ge 2700 ] && { echo "[glm53] G1-launch TIMEOUT"; tail -30 $LOGD/launch_glm53.log; exit 1; }

chat=$(curl -s --max-time 180 http://127.0.0.1:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"What is the capital of France? One word."}],"max_tokens":16,"temperature":0}')
echo "$chat" > $LOGD/glm53_chat.json
echo "$chat" | grep -qi paris && echo "[glm53] G3-precision PASS" || echo "[glm53] G3-precision FAIL"

long=$(curl -s --max-time 300 http://127.0.0.1:$PORT/generate \
  -H 'Content-Type: application/json' \
  -d "{\"text\": \"$(printf 'Paris is the capital of France. The Seine river divides the city into the Left Bank and the Right Bank. %.0s' $(seq 1 260))\", \"sampling_params\": {\"max_new_tokens\": 24, \"temperature\": 0}}")
echo "$long" | head -c 300 > $LOGD/glm53_long.json
echo "[glm53] G4-long out-tail: $(echo "$long" | tail -c 160)"
echo "[glm53] DONE (server left running for interactive checks)"
