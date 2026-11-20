#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Nightly / manual lane for full-scale emu x triton numeric parity
# (governance 3b-9 + P8, 2026-11-19): CI gates deselect the emu_parity
# mark, this lane runs exactly that mark on an HCU container. Manual
# single-op debugging:
#   pytest "tests/kernels/test_kda_chunk_gfx938.py::test_chunk_kda_zero_gate_parity" -m emu_parity
# Shape override backdoors (smaller smoke): GATE_T1 / GATE_T2 env vars.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${SGLANG_UP_TREE:-/home/up/python}:${ROOT}:${PYTHONPATH:-}"
exec python3 -m pytest tests/kernels -m emu_parity -vv --no-header "$@"
