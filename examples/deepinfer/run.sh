#!/usr/bin/env bash
# Launch an upstream DeepInfer command with the shared HCU adapter.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: DEEPINFER_SOURCE=/path/to/DeepInfer $0 CONFIG COMMAND [ARGS...]" >&2
  exit 2
fi
: "${DEEPINFER_SOURCE:?Set DEEPINFER_SOURCE to the upstream DeepInfer checkout}"
if [[ ! -d "$DEEPINFER_SOURCE/deepinfer" || ! -f "$1" ]]; then
  echo "DeepInfer checkout or config file does not exist" >&2
  exit 2
fi
adapter_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
config_dir=$(cd -- "$(dirname -- "$1")" && pwd)
export HCU_CONFIG="$config_dir/$(basename -- "$1")"
export HCU_ADAPTER_HOST=deepinfer
export PYTHONPATH="$adapter_root:$DEEPINFER_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
shift
exec "$@"
