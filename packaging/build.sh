#!/usr/bin/env bash
# Build BOTH adapter-das wheels from the single version source
# (sgl_kernel/version.py -- governance §7.4). One command, two artifacts:
#   dist/sgl_kernel-<version>-*.whl   (pure shadow kernel)
#   dist/hcu_adapter-<version>-*.whl  (adapter: bootstrap + bridges + hooks)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$ROOT/sgl_kernel/version.py")"
if [ -z "$VERSION" ]; then
    echo "build.sh: cannot read __version__ from sgl_kernel/version.py" >&2
    exit 1
fi

# interlock pin must equal the single source (l0_gate re-checks; fail here
# too so a release never builds with a drifted pin)
if ! grep -q "sgl-kernel==${VERSION}" "$ROOT/packaging/hcu-adapter/pyproject.toml"; then
    echo "build.sh: packaging/hcu-adapter/pyproject.toml interlock pin" \
         "does not match version ${VERSION} (governance §7.4)" >&2
    exit 1
fi

# the adapter dist carries its own independent version line (hand-edited
# single source: packaging/hcu-adapter/VERSION)
ADAPTER_VERSION="$(tr -d ' \t\r\n' < "$ROOT/packaging/hcu-adapter/VERSION")"
if [ -z "$ADAPTER_VERSION" ]; then
    echo "build.sh: packaging/hcu-adapter/VERSION is empty" >&2
    exit 1
fi

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    # probe: some Windows boxes ship a Microsoft-Store python3 stub that
    # exists on PATH but cannot run
    for cand in python3 python; do
        if "$cand" -c "import sys" >/dev/null 2>&1; then PY="$cand"; break; fi
    done
fi
[ -n "$PY" ] || { echo "build.sh: no working python3/python on PATH" >&2; exit 1; }
mkdir -p "$ROOT/dist"

echo "== building sgl-kernel ${VERSION} (repo root) =="
"$PY" -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir "$ROOT/dist" "$ROOT"

echo "== building hcu-adapter ${VERSION} (packaging/hcu-adapter) =="
"$PY" -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir "$ROOT/dist" "$ROOT/packaging/hcu-adapter"

ls -l "$ROOT/dist"/sgl_kernel-"${VERSION}"-*.whl "$ROOT/dist"/hcu_adapter-"${ADAPTER_VERSION}"-*.whl
