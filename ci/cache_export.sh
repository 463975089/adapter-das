#!/bin/sh
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Export warm sglang-JIT + triton caches from this container for cold-
# container seeding (R-17 CI init-time control; bootstrap.md 4d). Caches
# are ISA-keyed (gfx93x dir names) and tree-hash-keyed (build-<hash>), so
# one export serves any container on the same gfx family + same upstream
# tree revision. Adapter-side artifacts (cuda-mocker-built .so) travel in
# the same payload and are name-keyed to that hash -- keep the exporter
# tree at the revision you will test on the importer.
#
# THIRD KEY (2026-11-19 forensics): every sglang-JIT build key also folds in
# the materialized adapter compiler bundle path ~/.cache/hcu_adapter/jit-inputs/<key>,
# and that key hashes hcu_adapter/features/jit/* + jit_sources.py + jit_compat.py
# bytes. Any edit there (even a comment-level change) changes the -I path and
# therefore EVERY build key -- the seed goes 100% stale. Evidence: jit_seed_v3
# (exported 08:04) was exported before dcc4d33 (12:26, jit_sources.py); the
# next acceptance round rebuilt 17 modules, ~16 min inside l15_exact alone,
# and the 512 variant appeared under a THIRD build-<hash> whose build.ninja
# differed only in the two -I bundle paths. Rule: export the seed from the
# FINAL tree after the last adapter commit; re-export whenever the jit
# transform layer changes.
#
# Also ships tvm-ffi (+ torch_extensions): a cold ~/.cache/tvm-ffi makes
# tvm_ffi spawn its optional torch-dlpack builder at import, and that
# builder self-deadlocks on HCU (forked child re-imports torch/HSA, both
# sides poll pipes forever; validation-host-18 2026-10-20 forensics). Seeding the ~200K
# addon .so skips the build entirely; without a seed, a cold container
# must set TVM_FFI_DISABLE_TORCH_C_DLPACK=1 instead.
#
# Usage:  bash ci/cache_export.sh [OUT.tgz]      (default /tmp/hcu_cache_seed.tgz)
set -e
OUT="${1:-/tmp/hcu_cache_seed.tgz}"
C="${HOME}/.cache"
[ -d "$C/sglang/jit" ] || { echo "cache_export: no $C/sglang/jit -- run the suites once first"; exit 2; }
PAYLOAD="sglang/jit"
# The DTK sglang stack parks its triton cache at sglang/triton, not the
# upstream ~/.cache/triton (2026-11-17: seeds without it made every cold
# container recompile the whole triton corpus -- kernels ran 2x its warm
# pace on the seeded 0.5.22 image).
for d in sglang/triton; do
    [ -d "$C/$d" ] && PAYLOAD="$PAYLOAD $d"
done
for d in triton tvm-ffi torch_extensions; do
    [ -d "$C/$d" ] && PAYLOAD="$PAYLOAD $d"
done
case " $PAYLOAD " in
    *" tvm-ffi "*) ;;
    *) echo "cache_export: WARNING -- no $C/tvm-ffi in payload; cold importer hits the optional-build deadlock unless TVM_FFI_DISABLE_TORCH_C_DLPACK=1" ;;
esac
# shellcheck disable=SC2086
tar czf "$OUT" -C "$C" $PAYLOAD
echo "cache_export: $OUT ($(du -sh "$OUT" | cut -f1))"
echo "  payload: $PAYLOAD"
echo "  gfx dirs: $(ls "$C/sglang/jit" 2>/dev/null | tr '\n' ' ')"
echo "  import on a cold container: bash ci/cache_import.sh $OUT"
