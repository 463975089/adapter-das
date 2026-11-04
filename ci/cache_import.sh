#!/bin/sh
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Import a cache payload produced by ci/cache_export.sh into this (cold)
# container: instant JIT/triton hits instead of a full first-compile storm
# (R-17 CI init-time control; bootstrap.md 4d). Keying rules live with the
# exporter: ISA (gfx93x) + upstream tree revision (build-<hash>) must
# match; mocker-built .so travel inside and break cleanly if the mock
# toolchain differs (loader falls back to rebuild).
#
# Usage:  bash ci/cache_import.sh [SEED.tgz]     (default /tmp/hcu_cache_seed.tgz)
set -e
SEED="${1:-/tmp/hcu_cache_seed.tgz}"
[ -f "$SEED" ] || { echo "cache_import: $SEED not found"; exit 2; }
mkdir -p "${HOME}/.cache"
tar xzf "$SEED" -C "${HOME}/.cache"
echo "cache_import: seeded from $SEED"
echo "  jit: $(find ~/.cache/sglang/jit -maxdepth 1 -type d 2>/dev/null | wc -l) modules, gfx dirs: $(ls ~/.cache/sglang/jit 2>/dev/null | tr '\n' ' ')"
echo "  tvm-ffi addon: $(ls ~/.cache/tvm-ffi/*.so 2>/dev/null | wc -l) .so (cold-build deadlock guard; none -> set TVM_FFI_DISABLE_TORCH_C_DLPACK=1)"
