#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Four-gate all live servers: short-gen / semantic anchor / long-context.
LOGD=/root/logs/model_suite
probe() {  # $1 name, $2 port
  local name=$1 port=$2 chat long short
  short=$(curl -s --max-time 120 http://127.0.0.1:$port/generate \
    -H 'Content-Type: application/json' \
    -d '{"text": "The distance from London to Paris is 344 km. Distance from Paris to Madrid is", "sampling_params": {"max_new_tokens": 48, "temperature": 0}}' \
    | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("text","")[-160:])
except Exception as e: print("parse-err:", e)')
  chat=$(curl -s --max-time 180 http://127.0.0.1:$port/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"What is the capital of France? One word."}],"max_tokens":16,"temperature":0}' \
    | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["choices"][0]["message"]["content"][-80:])
except Exception as e: print("parse-err:", e)')
  long=$(curl -s --max-time 300 http://127.0.0.1:$port/generate \
    -H 'Content-Type: application/json' \
    -d "{\"text\": \"$(printf 'Paris is the capital of France. The Seine river divides the city into the Left Bank and the Right Bank. %.0s' $(seq 1 260))\", \"sampling_params\": {\"max_new_tokens\": 24, \"temperature\": 0}}" \
    | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("text","")[-120:])
except Exception as e: print("parse-err:", e)')
  {
    echo "== $name (port $port)"
    echo "SHORT: $short"
    echo "CHAT: $chat"
    echo "LONG: $long"
    echo "$chat" | grep -qi paris && echo "ANCHOR PASS" || echo "ANCHOR FAIL"
  } >> $LOGD/fourgate_$(date +%H%M).txt
}
: > $LOGD/fourgate_$(date +%H%M).txt
probe q25-05b 30036
probe q38-fp8 30046
probe q3-30b  30056
probe q36-int8 30066
cat $LOGD/fourgate_$(date +%H%M).txt
