#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# L1.6 model launch readiness gate. Run inside the designated HCU container.
# It is intentionally separate from L1.5 component tests and L2 model suites.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

UP_TREE="${1:?usage: l16_model_readiness_gate.sh /path/to/sglang [checkpoint] [out.json]}"
CHECKPOINT="${2:-}"
OUTPUT="${3:-$ROOT/results/l16-readiness/model-readiness.json}"

args=(
  run
  --up-tree "$UP_TREE"
  --out "$OUTPUT"
)
if [ -n "$CHECKPOINT" ]; then
  args+=(--checkpoint "$CHECKPOINT")
fi

python3 -m tools.l16_model_readiness "${args[@]}"
