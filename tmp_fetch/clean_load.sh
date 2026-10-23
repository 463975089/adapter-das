#!/bin/bash
for i in $(seq 1 16); do
  curl -s -m 120 http://127.0.0.1:30296/generate -H 'Content-Type: application/json' \
    -d '{"text":"Warmup story","sampling_params":{"temperature":0.7,"max_new_tokens":32,"ignore_eos":true}}' > /dev/null &
done
wait
echo WARM_DONE $(date +%H:%M:%S)
for i in $(seq 1 16); do
  curl -s -m 1200 http://127.0.0.1:30296/generate -H 'Content-Type: application/json' \
    -d '{"text":"Once upon a time in a distant land there lived","sampling_params":{"temperature":0.7,"max_new_tokens":600,"ignore_eos":true}}' > /dev/null &
done
wait
echo LOAD_DONE $(date +%H:%M:%S)
