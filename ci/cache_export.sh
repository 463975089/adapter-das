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
# Also ships tvm-ffi (+ torch_extensions): a cold ~/.cache/tvm-ffi makes
# tvm_ffi spawn its optional torch-dlpack builder at import, and that
# builder self-deadlocks on HCU (forked child re-imports torch/HSA, both
# sides poll pipes forever; nmz18 2026-10-20 forensics). Seeding the ~200K
# addon .so skips the build entirely; without a seed, a cold container
# must set TVM_FFI_DISABLE_TORCH_C_DLPACK=1 instead.
#
# Usage:  bash ci/cache_export.sh [OUT.tgz]      (default /tmp/hcu_cache_seed.tgz)
set -e
OUT="${1:-/tmp/hcu_cache_seed.tgz}"
C="${HOME}/.cache"
[ -d "$C/sglang/jit" ] || { echo "cache_export: no $C/sglang/jit -- run the suites once first"; exit 2; }
PAYLOAD="sglang/jit"
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
