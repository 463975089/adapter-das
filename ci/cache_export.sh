#!/bin/sh
# Export warm sglang-JIT + triton caches from this container for cold-
# container seeding (R-17 CI init-time control; bootstrap.md 4d). Caches
# are ISA-keyed (gfx93x dir names) and tree-hash-keyed (build-<hash>), so
# one export serves any container on the same gfx family + same upstream
# tree revision. Adapter-side artifacts (cuda-mocker-built .so) travel in
# the same payload and are name-keyed to that hash -- keep the exporter
# tree at the revision you will test on the importer.
#
# Usage:  bash ci/cache_export.sh [OUT.tgz]      (default /tmp/hcu_cache_seed.tgz)
set -e
OUT="${1:-/tmp/hcu_cache_seed.tgz}"
C="${HOME}/.cache"
[ -d "$C/sglang/jit" ] || { echo "cache_export: no $C/sglang/jit -- run the suites once first"; exit 2; }
tar czf "$OUT" -C "$C" sglang/jit triton
echo "cache_export: $OUT ($(du -sh "$OUT" | cut -f1))"
echo "  gfx dirs: $(ls "$C/sglang/jit" 2>/dev/null | tr '\n' ' ')"
echo "  import on a cold container: bash ci/cache_import.sh $OUT"
