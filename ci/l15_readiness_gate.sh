#!/bin/bash
# Strict L1.5 HCU gate. Run inside the designated container on one visible GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
UP_TREE="${1:?usage: l15_readiness_gate.sh /path/to/sglang [output.json]}"
OUTPUT="${2:-$ROOT/results/l15-readiness/$(date +%Y%m%dT%H%M%S).json}"
# The orchestrator has no host; test children explicitly select SGLang.
HCU_ADAPTER_HOST=none python3 tools/l15_family_matrix.py run \
    --adapter-root "$ROOT" --up-tree "$UP_TREE" --out "$OUTPUT"
