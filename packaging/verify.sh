#!/usr/bin/env bash
# Release-flow install-combo verification over the two wheels built by
# packaging/build.sh. Three combos (dual-dist semantics, governance §7.4):
#   A sglang combo   -- both wheels: shadow import, entry-points, sitecustomize
#   B vLLM-only      -- hcu-adapter alone: entry-points + sitecustomize live
#                        with it; sgl_kernel is not installed in the venv
#   C half-install   -- sgl-kernel alone: import sgl_kernel must raise the
#                        sentinel (unless an ambient sgl_kernel shadows it)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
# build.sh stamps each pair with hcu.<time>.g<commit>; pick the newest pair
KERNEL_WHL="$(ls -t "$DIST"/sgl_kernel-*.whl | head -1)"
ADAPTER_WHL="$(ls -t "$DIST"/hcu_adapter-*.whl | head -1)"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for cand in python3 python; do
        if "$cand" -c "import sys" >/dev/null 2>&1; then PY="$cand"; break; fi
    done
fi
[ -n "$PY" ] || { echo "verify.sh: no working python3/python on PATH" >&2; exit 1; }

mkvenv() {  # torch/triton come from the system site; wheels install --no-deps
    "$PY" -m venv --system-site-packages "$1"
}

venv_py() {  # linux venvs use bin/, windows venvs use Scripts/
    if [ -x "$1/bin/python" ]; then echo "$1/bin/python"
    else echo "$1/Scripts/python.exe"; fi
}

run_py() {  # run_py <venv> <code>
    "$(venv_py "$1")" -c "$2"
}

venv_pip() {  # venv_pip <venv> <args...>
    "$(venv_py "$1")" -m pip "${@:2}"
}

echo "== combo A: sglang (both wheels) =="
VA="$(mktemp -d)/va"; mkvenv "$VA"
venv_pip "$VA" install --no-deps --quiet "$KERNEL_WHL" "$ADAPTER_WHL"
run_py "$VA" '
import sys, importlib.metadata as im
import sgl_kernel
assert "sitecustomize" in sys.modules, "sitecustomize not effective"
eps = {e.name: e for e in im.entry_points(group="vllm.platform_plugins")
       if (e.dist.name if e.dist else "") == "hcu-adapter"}
assert "hcu_adapter" in eps, "vllm platform plugin entry-point missing"
pin = [r.replace(" ", "") for r in (im.metadata("hcu-adapter").get_all("Requires-Dist") or [])
       if r.startswith("sgl-kernel")]
expected = "sgl-kernel==" + im.version("sgl-kernel")
assert pin == [expected], (pin, expected)
print("A ok:", sgl_kernel.__version__)
'

echo "== combo B: vLLM-only (hcu-adapter alone) =="
VB="$(mktemp -d)/vb"; mkvenv "$VB"
venv_pip "$VB" install --no-deps --quiet "$ADAPTER_WHL"
if venv_pip "$VB" show sgl-kernel >/dev/null 2>&1; then
    echo "B FAIL: sgl-kernel must not be installed in this venv" >&2; exit 1
fi
run_py "$VB" '
import sys, importlib.metadata as im
assert "sitecustomize" in sys.modules
eps = [e for e in im.entry_points(group="vllm.general_plugins")
       if (e.dist.name if e.dist else "") == "hcu-adapter"]
assert eps, "vllm general plugin entry-point missing"
try:
    import sgl_kernel
except RuntimeError:
    pass
else:
    print("B note: ambient sgl_kernel shadowed the sentinel path; "
          "unit test test_sgl_kernel_*_sentinel covers the clean case")
print("B ok")
'

echo "== combo C: half-install (sgl-kernel alone) =="
VC="$(mktemp -d)/vc"; mkvenv "$VC"
venv_pip "$VC" install --no-deps --quiet "$KERNEL_WHL"
run_py "$VC" '
import importlib.util as ilu
import sys
try:
    import sgl_kernel
except RuntimeError as e:
    assert "hcu-adapter" in str(e), e
    print("C ok: sentinel message:", str(e)[:70])
else:
    # an ambient hcu_adapter (e.g. an editable install leaking through
    # --system-site-packages) shadowed the half-install; re-probe with
    # hcu_adapter blocked so the sentinel branch is deterministic
    real = ilu.find_spec
    def blocked(name, *a, **k):
        if name == "hcu_adapter" or name.startswith("hcu_adapter."):
            return None
        return real(name, *a, **k)
    ilu.find_spec = blocked
    for m in [m for m in sys.modules if m == "sgl_kernel" or m.startswith("sgl_kernel.")]:
        del sys.modules[m]
    try:
        import sgl_kernel
    except RuntimeError as e:
        assert "hcu-adapter" in str(e), e
        print("C ok (ambient shadowed; blocked re-probe):", str(e)[:70])
    else:
        raise SystemExit("C FAIL: sgl_kernel imported without hcu-adapter")
'

echo "== verify.sh: all combos passed =="
