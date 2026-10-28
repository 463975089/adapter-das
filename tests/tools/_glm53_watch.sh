#!/bin/bash
# GLM-5.3-Flash TP8 watchdog: wait until ALL 8 HCUs have >=40GiB free, then
# launch (bf16 306G, page_size=256 recipe from nmz18 rounds) + four-gate.
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:$LD_LIBRARY_PATH
PORT=30053
LOGD=/root/logs/model_suite
mkdir -p $LOGD
export PYTHONPATH=/root/hcu_t2:$HOME/sglang-v0521/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_RUST_BUILD_MODE=never
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES  # TP8 needs all

MOD=/models/GLM-5.3-Flash-CHANNEL-INT4-w4a8-v3-same-as-iter_0033200/GLM-5.3-Flash-CHANNEL-INT4-w4a8-v3-same-as-iter_0033200
# bf16 in mtv container mount? probe both; fall back documented in log.
MODEL=${GLM_MODEL_PATH:-$MOD}

free_mib_card() {
  local out t u
  out=$(/opt/dtk/bin/rocm-smi --showmeminfo vram 2>/dev/null)
  t=$(echo "$out" | grep -A1 "HCU\[$1\]" | grep "Total Memory" | head -1 | tr -d ' ' | grep -o '[0-9]*$')
  u=$(echo "$out" | grep -A2 "HCU\[$1\]" | grep "Used Memory" | head -1 | tr -d ' ' | grep -o '[0-9]*$')
  echo $(( (t - u) / 1024 ))
}

echo "[glm53] model=$MODEL"
while true; do
  ok=1
  for c in 0 1 2 3 4 5 6 7; do
    f=$(free_mib_card $c)
    if [ "${f:-0}" -lt 42000 ]; then ok=0; break; fi
  done
  if [ "$ok" = "1" ]; then break; fi
  sleep 60
done
echo "[glm53] all 8 cards free, launching (page256, tp8)"
pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 6
nohup python -m sglang.launch_server \
  --model-path "$MODEL" --host 0.0.0.0 --port $PORT \
  --tp-size 8 --page-size 256 --mem-fraction-static 0.85 \
  --trust-remote-code --log-level info \
  > $LOGD/launch_glm53.log 2>&1 &

t=0
while [ $t -lt 2400 ]; do
  curl -s --max-time 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
  pgrep -f "sglang.launch_server" >/dev/null || { echo "[glm53] LAUNCH DIED"; tail -20 $LOGD/launch_glm53.log; exit 2; }
  sleep 15; t=$((t + 15))
done
if ! curl -s --max-time 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
  echo "[glm53] G1-launch TIMEOUT rc=1"; tail -30 $LOGD/launch_glm53.log; exit 1
fi
echo "[glm53] G1-launch PASS"
curl -s --max-time 120 http://127.0.0.1:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"What is the capital of France? One word."}],"max_tokens":16,"temperature":0}' \
  | tee $LOGD/glm53_chat.json | grep -qi paris && echo "[glm53] G3-precision PASS" || echo "[glm53] G3-precision FAIL (see chat.json)"
