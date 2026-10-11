#!/bin/bash
# Build the custom-AR device extension (HIP C++). Run on the HCU node with
# DTK hipcc available. The output libhcu_ar.so must sit next to this script
# (triton_ar.py loads it from its own directory).
set -e
cd "$(dirname "$0")"
hipcc -O2 -fPIC -shared ar_ext.cu -o libhcu_ar.so
echo "built: $(pwd)/libhcu_ar.so"
