#!/usr/bin/env bash
# Build BOTH adapter-das wheels from the single version source
# (sgl_kernel/version.py -- governance §7.4). One command, two artifacts:
#   dist/sgl_kernel-<ver>+<platform>[.<stamp>]-<py>-<plat>.whl  (native kernel)
#   dist/hcu_adapter-<ver>+<platform>[.<stamp>]-py3-none-any.whl (adapter)
# The kernel is compiled from csrc/ via setup_hip.py inside a DTK container;
# both wheels carry the same dtk/torch ABI label and hcu-adapter pins the
# exact kernel build (build-time interlock). --no-stamp drops the
# time/commit suffix but keeps the platform label.
# Compile targets = every arch listed in AMDGPU_TARGETS ("gfx936;gfx938").
set -euo pipefail

USE_STAMP=1
for arg in "$@"; do
    case "$arg" in
        --no-stamp) USE_STAMP=0 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "build.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$ROOT/sgl_kernel/version.py")"
if [ -z "$VERSION" ]; then
    echo "build.sh: cannot read __version__ from sgl_kernel/version.py" >&2
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

# native build prerequisites (DTK container): HIP torch + hipcc + build tools
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
    echo "build.sh: torch is not importable; install the DTK HIP torch first" >&2
    exit 1
fi
if ! "$PY" -c "import torch; assert torch.version.hip is not None" >/dev/null 2>&1; then
    echo "build.sh: installed torch is not a HIP/DTK build (torch.version.hip)" >&2
    exit 1
fi
if ! command -v hipcc >/dev/null 2>&1; then
    echo "build.sh: hipcc not on PATH (source /opt/dtk/env.sh?)" >&2
    exit 1
fi
for mod in setuptools.build_meta wheel ninja; do
    if ! "$PY" -c "import $mod" >/dev/null 2>&1; then
        echo "== installing build deps (setuptools/wheel/ninja) =="
        "$PY" -m pip install "setuptools>=64" wheel ninja
        break
    fi
done
if [ -z "${AMDGPU_TARGETS:-}" ] && \
   ! "$PY" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    echo "build.sh: AMDGPU_TARGETS is empty and no HCU device is visible" >&2
    exit 1
fi

# DTK/torch ABI label: the compiled extension is bound to both, so the label
# is part of the version even without a build stamp.
DTK_TAG="${DAS_DTK_TAG:-${DTK_VERSION:-}}"
if [ -z "$DTK_TAG" ] && [ -n "${DTK_VER:-}" ]; then
    DTK_TAG="dtk${DTK_VER}"
fi
DTK_TAG="$(printf '%s' "$DTK_TAG" | tr 'A-Z' 'a-z')"
DTK_TAG="${DTK_TAG#dtk}"
DTK_TAG="${DTK_TAG//[^a-z0-9]/}"
if [ -z "$DTK_TAG" ]; then
    echo "build.sh: cannot determine DTK version" \
         "(set DAS_DTK_TAG/DTK_VERSION/DTK_VER)" >&2
    exit 1
fi
TORCH_TAG="$("$PY" -c 'import re, torch; print("torch" + "".join(re.findall(r"\d+", torch.__version__.split("+", 1)[0])[:3]))')"
DAS_PLATFORM_TAG="dtk${DTK_TAG}.${TORCH_TAG}"
export DAS_PLATFORM_TAG

# one stamp for both wheels: <YYMMDDHHMM>.g<commit6>; --no-stamp keeps the
# ABI platform label but drops the per-build suffix
if [ "$USE_STAMP" = 1 ]; then
    DAS_BUILD_STAMP="${DAS_BUILD_STAMP:-$(date +%y%m%d%H%M).g$(git -C "$ROOT" rev-parse --short=6 HEAD)}"
    export DAS_BUILD_STAMP
else
    unset DAS_BUILD_STAMP
fi

LOCAL_SUFFIX="${DAS_PLATFORM_TAG}${DAS_BUILD_STAMP:+.$DAS_BUILD_STAMP}"
KERNEL_BUILD_VERSION="${VERSION%%+*}+${LOCAL_SUFFIX}"
ADAPTER_BUILD_VERSION="${ADAPTER_VERSION%%+*}+${LOCAL_SUFFIX}"

# pin check: _pin.py is the build-time interlock source, ci/l0_gate re-checks
EXPECTED_PIN="sgl-kernel==${VERSION%%+*}"
ACTUAL_PIN="$(SGL_ADAPTER_ROOT="$ROOT" "$PY" -c '
import os, sys
sys.path.insert(0, os.path.join(os.environ["SGL_ADAPTER_ROOT"], "packaging", "hcu-adapter"))
import _pin
print(_pin.kernel_pin(stamp="", platform=""))
')"
if [ "$ACTUAL_PIN" != "$EXPECTED_PIN" ]; then
    echo "build.sh: hcu-adapter interlock pin ${ACTUAL_PIN}" \
         "does not match public version ${EXPECTED_PIN#sgl-kernel==} (governance §7.4)" >&2
    exit 1
fi

mkdir -p "$ROOT/dist"

echo "== building native sgl-kernel ${KERNEL_BUILD_VERSION} (setup_hip.py) =="
( cd "$ROOT" && "$PY" setup_hip.py bdist_wheel --dist-dir "$ROOT/dist" )

echo "== building hcu-adapter ${ADAPTER_BUILD_VERSION} (packaging/hcu-adapter) =="
"$PY" -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir "$ROOT/dist" "$ROOT/packaging/hcu-adapter"

ls -l "$ROOT/dist"/sgl_kernel-"${VERSION%%+*}"+*.whl "$ROOT/dist"/hcu_adapter-"${ADAPTER_VERSION%%+*}"+*.whl
