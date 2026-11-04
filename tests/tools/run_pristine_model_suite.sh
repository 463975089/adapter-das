#!/usr/bin/env bash
# Pristine-source wrapper for the real-model suite.
#
# Required:
#   SGLANG_TREE   community checkout containing python/sglang
#   ADAPTER_TREE  adapter-das checkout
set -euo pipefail

: "${SGLANG_TREE:?set SGLANG_TREE to the pristine community SGLang checkout}"
: "${ADAPTER_TREE:?set ADAPTER_TREE to the adapter-das checkout}"

if [[ ! -d "$SGLANG_TREE/python/sglang" ]]; then
  echo "missing community SGLang package: $SGLANG_TREE/python/sglang" >&2
  exit 2
fi
if [[ ! -d "$ADAPTER_TREE/hcu_adapter" ]]; then
  echo "missing adapter package: $ADAPTER_TREE/hcu_adapter" >&2
  exit 2
fi

cd "$ADAPTER_TREE"
export ADAPTER="$ADAPTER_TREE"
export SGLANG_V0521="$SGLANG_TREE/python"
export HCU_ADAPTER_HOST="${HCU_ADAPTER_HOST:-sglang}"
export PYTHONPATH="$ADAPTER_TREE:$SGLANG_TREE/python${PYTHONPATH:+:$PYTHONPATH}"

python -m tools.upstream_source --up-tree "$SGLANG_TREE"
if [[ "${1:-}" == "--preflight-only" ]]; then
  python -c 'import sglang; print("sglang:", sglang.__file__)'
  exit 0
fi
exec bash "$ADAPTER_TREE/tests/tools/run_model_suite.sh" "$@"
