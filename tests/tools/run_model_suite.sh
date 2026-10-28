#!/bin/bash
# Model-suite batch runner: wait for a free HCU, then validate each model in
# docs/models.md, one at a time, four-gate each (launch/generation/precision/
# long-context). State + logs under /root/logs/model_suite.
# Usage (inside pr418_test):  bash /root/run_model_suite.sh [model-id ...]
set -u
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:$LD_LIBRARY_PATH
PORT=30036
LOGD=/root/logs/model_suite
mkdir -p $LOGD
SGLANG_V0521=$HOME/sglang-v0521/python
ADAPTER=/root/hcu_t2
export PYTHONPATH=$ADAPTER:$SGLANG_V0521
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_RUST_BUILD_MODE=never
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1

M=/public/opendas/DL_DATA/llm-models

declare -A MODELS=(
  # id | model_path | extra args
  [q25-05b]="$M/qwen2.5/Qwen2.5-0.5B-Instruct::"
  [q3-32b]="$M/qwen3/Qwen3-32B::--mem-fraction-static 0.85"
  [q3-30b-a3b]="$M/qwen3/Qwen3-30B-A3B::--mem-fraction-static 0.85"
  [q38-27b-fp8]="$M/qwen3.8/Qwen3.8-27B-FP8::--mem-fraction-static 0.85 --trust-remote-code"
  [q35-122b-awq]="$M/qwen3.5/Qwen3.5-122B-A10B-AWQ::--tp-size 4 --mem-fraction-static 0.85 --trust-remote-code"
)
ORDER=(q25-05b q3-32b q3-30b-a3b q38-27b-fp8 q35-122b-awq)

need_mib() {  # rough VRAM need per model id (MiB), includes graph/act margin
  case $1 in
    q25-05b) echo 4200 ;;
    q3-32b) echo 78000 ;;
    q3-30b-a3b) echo 74000 ;;
    q38-27b-fp8) echo 40000 ;;
    q35-122b-awq) echo 50000 ;;   # per-GPU with TP4
    *) echo 999999 ;;
  esac
}

free_mib() {  # free VRAM (MiB) of HCU[$1]
  local out t u
  out=$(/opt/dtk/bin/rocm-smi --showmeminfo vram 2>/dev/null)
  t=$(echo "$out" | grep -A1 "HCU\[$1\]" | grep "Total Memory" | head -1 \
      | tr -d ' ' | grep -o '[0-9]*$')
  u=$(echo "$out" | grep -A2 "HCU\[$1\]" | grep "Used Memory" | head -1 \
      | tr -d ' ' | grep -o '[0-9]*$')
  echo $(( (t - u) / 1024 ))
}

stop_server() {
  pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null
  pkill -9 -f "sglang::" 2>/dev/null
  sleep 6
}

start_server() {  # $1 model-path, $2 extra args
  stop_server
  echo "[suite] launching: $1 $2"
  nohup python -m sglang.launch_server \
    --model-path "$1" --host 0.0.0.0 --port $PORT \
    --page-size 64 --log-level info $2 \
    > $LOGD/launch_${1##*/}.log 2>&1 &
}

wait_ready() {  # $1 timeout seconds
  local t=0
  while [ $t -lt $1 ]; do
    if curl -s --max-time 3 http://127.0.0.1:$PORT/health | grep -q '^$'; then
      sleep 2
      curl -s --max-time 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0
    fi
    # dead early?
    if ! pgrep -f "sglang.launch_server.*--port $PORT" >/dev/null; then
      return 2
    fi
    sleep 10; t=$((t + 10))
  done
  return 1
}

gen_gate() {  # $2 prompt -> prints generated text tail
  curl -s --max-time 90 http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d "{\"text\": \"$2\", \"sampling_params\": {\"max_new_tokens\": 32, \"temperature\": 0}}" \
  | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("text", "")[-160:])
except Exception as e:
    print("parse-err:", e)'
}

judge_model() {  # $1 id, $2 model-path
  local id=$1 name=${2##*/}
  local res=$LOGD/result_$id.txt
  : > $res

  # G1 launch handled by caller; G2 generation
  local short short_out
  short_out=$(gen_gate short "The distance from London to Paris is 344 km. Distance from Paris to Madrid is")
  echo "GEN: $short_out" >> $res
  if [ "$(echo "$short_out" | tr -d '[:space:]' | wc -c)" -gt 3 ]; then
    echo "G2-generation PASS" >> $res
  else
    echo "G2-generation FAIL" >> $res
  fi

  # G3 precision: semantic anchor via /v1/chat/completions
  local chat chat_tail
  chat=$(curl -s --max-time 120 http://127.0.0.1:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "x", "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}], "max_tokens": 16, "temperature": 0}')
  chat_tail=$(echo "$chat" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin)["choices"][0]["message"]["content"][-80:])
except Exception as e:
    print("parse-err:", e)' 2>/dev/null)
  echo "CHAT: $chat_tail" >> $res
  if echo "$chat" | grep -qi paris; then
    echo "G3-precision PASS" >> $res
  else
    echo "G3-precision FAIL" >> $res
  fi

  # G4 long context ~2700 tokens (sparse variants only; dense fine too)
  local long_out
  long_out=$(curl -s --max-time 240 http://127.0.0.1:$PORT/generate \
    -H 'Content-Type: application/json' \
    -d @- <<REQEOF
{"text": "$(printf 'Paris is the capital of France. The Seine river divides the city into the Left Bank and the Right Bank. %.0s' $(seq 1 260))", "sampling_params": {"max_new_tokens": 24, "temperature": 0}}
REQEOF
)
  echo "LONG: $(echo "$long_out" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("text","")[-100:].replace(chr(10)," "))
except Exception as e:
    print("parse-err", e)' 2>/dev/null)" >> $res
  if echo "$long_out" | python3 -c 'import json,sys
try:
    t = json.load(sys.stdin).get("text","")
    sys.exit(0 if len(t.strip()) > 8 else 1)
except Exception:
    sys.exit(1)' 2>/dev/null; then
    echo "G4-long PASS" >> $res
  else
    echo "G4-long FAIL" >> $res
  fi
}

ITS="${*:-${ORDER[@]}}"
echo "[suite] models: $ITS"

for id in $ITS; do
  spec=${MODELS[$id]}
  path=${spec%%::*}
  extra=${spec##*::}
  need=$(need_mib $id)

  echo "[suite] [$id] waiting for >= ${need}MiB free on any HCU..."
  while true; do
    avail=0
    for card in 0 1 2 3 4 5 6 7; do
      a=$(free_mib $card)
      if [ "${a:-0}" -ge "$need" ]; then avail=$a
        export HIP_VISIBLE_DEVICES=$card CUDA_VISIBLE_DEVICES=$card
        break
      fi
    done
    if [ "$avail" -ge "$need" ]; then break; fi
    sleep 120
  done

  echo "[suite] [$id] starting (avail=${avail}MiB)"
  start_server "$path" "$extra"
  wait_ready 1500
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$id G1-launch FAIL (wait rc=$rc)" >> $LOGD/result_$id.txt
    echo "[suite] [$id] LAUNCH FAILED, next"
    continue
  fi
  echo "$id G1-launch PASS" >> $LOGD/result_$id.txt
  judge_model "$id" "$path"
  echo "[suite] [$id] verified: $(grep -c PASS $LOGD/result_$id.txt)/4 gates"
  tail -5 $LOGD/result_$id.txt
done
stop_server
echo "[suite] DONE"
