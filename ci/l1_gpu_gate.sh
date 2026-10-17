#!/bin/bash
# L1 GPU gate -- numeric/kernel tier, runs INSIDE the HCU container
# (torch + DTK stack + aiter/lightop wheels present).
#
# Usage:   bash ci/l1_gpu_gate.sh [GPU_ID]      # omitted -> auto-pick free GPU
# Budget:  ~5-15 min on one gfx938 card.
# Exit:    0 green / 1 red. Summary lines:  "L1: <segment> PASS|FAIL".
#
# Segments (each independently timed out; all must pass):
#   l0        structural gate (no GPU) -- fast fail before touching the card
#   cpu       tests/test_ops_cpu.py            pure-torch contract paths
#   kernels   tests/kernels + tests/adapter    triton numerics vs torch refs
#   l15_up    upstream registered/attention/unittests (dsa+kda+dense+mla)
#             via pinned PYTHONPATH; tilelang variants excluded (DTK bug)
#   gates     curated tools gates: config catalog, fa3-emu CP shapes, rope
#             paths, aiter-MoE bridge dispatch, deepgemm contiguous triton
#   vendor    flash_attn wheel sanity probe    (WARN tier: bad wheel -> warn)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"

FAILS=()

# ---------------------------------------------------------------- GPU pick
pick_gpu() {
    local want="${1:-}"
    if [ -n "$want" ]; then echo "$want"; return 0; fi
    # free = VRAM% under 5 (rocm-smi may print mkfd noise on some hosts)
    local line pct id
    rocm-smi --showmemuse vram 2>/dev/null | grep -E '^[0-9]+[[:space:]]' >/dev/null || true
    # columns: id temp pwr perf cap VRAM% ...
    while read -r id _ _ _ _ pct _; do
        [ -z "$id" ] && continue
        pct="${pct%\%}"
        case "$pct" in ''|*[!0-9]*) continue;; esac
        if [ "$pct" -lt 5 ]; then echo "$id"; return 0; fi
    done < <(rocm-smi 2>/dev/null | grep -E '^[0-9]+[[:space:]]' )
    return 1
}

GPU="$(pick_gpu "${1:-}")" || { echo "L1: no free GPU (all >=5% VRAM) -- SKIP/QUEUE"; exit 2; }
export HIP_VISIBLE_DEVICES="$GPU" CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# some gates exercise upstream classes (rotary etc.) -- mount the sglang
# source the container actually serves from, when present. Both known
# locations: nmz26 serves from /sglang-v0521; box16 clones under
# /root/sglang-v0521-src. Without this, tests/kernels imports (e.g.
# kpool_fp8_index) resolve sglang from pip dist-packages, which lacks the
# dsa submodule -> collection error (2026-10-18 rerun2 forensics).
for d in /sglang-v0521/python /root/sglang-v0521-src/python; do
    [ -d "$d" ] && export PYTHONPATH="$d:$PYTHONPATH"
done
echo "L1: using GPU $GPU"

run_seg() {  # name timeout_s cmd...
    local name="$1"; local tmo="$2"; shift 2
    echo "L1: === $name (timeout ${tmo}s) ==="
    if timeout "$tmo" "$@" > "/tmp/l1_${name}.log" 2>&1; then
        echo "L1: $name PASS"
    else
        local rc=$?
        echo "L1: $name FAIL (rc=$rc) -- tail:"
        tail -n 15 "/tmp/l1_${name}.log" | sed 's/^/    /'
        FAILS+=("$name")
    fi
}

# ---------------------------------------------------------------- segments
run_seg l0      60   python3 ci/l0_gate.py
run_seg cpu     300  python3 -m pytest tests/test_ops_cpu.py -q --no-header
run_seg kernels 900  python3 -m pytest tests/kernels tests/adapter -q --no-header

# L1.5: upstream attention-unittest selection (dsa + kda + dense + mla
# dirs; mla extend subtests over flashinfer stubs -> legal skips, see
# _hcu_stub_raise in sitecustomize.py).
# Env iron rule (2026-10-17): the container hosts TWO sglang trees (pip
# dist-packages vs the source tree serving actually uses); the tests must
# pin PYTHONPATH to the upstream source tree FIRST and the adapter tree
# under test ($ROOT, providing sitecustomize.py) second, or they silently
# test the wrong tree with no adapters loaded. -k "not tilelang": DTK
# tilelang 0.1.12+dtk2604 warp-partition bug (threads=384 -> 6 warps,
# 1*4*1 != 6) fails at compile -- platform issue, see docs/ci.md L1.5.
# fa4/flashinfer/trtllm skip via attn_backend_gate / kit support checks.
# Cold JIT cache costs ~5 min/kernel (~30 min first run); warm = ~35 s.
UP_TREE=""
for d in /sglang-v0521 /root/sglang-v0521-src; do
    [ -d "$d/test/registered/attention/unittests" ] && UP_TREE="$d" && break
done
if [ -n "$UP_TREE" ]; then
    UP="$UP_TREE/test/registered/attention/unittests"
    run_seg l15_upstream 1800 env HCU_DEEPGEMM_BRIDGE=1 \
        PYTHONPATH="$UP_TREE/python:$ROOT" \
        python3 -m pytest "$UP/dsa" "$UP/kda" "$UP/dense" "$UP/mla" \
            -q --no-header \
        -k "not tilelang" -o faulthandler_timeout=580 --tb=line
else
    echo "L1: l15_upstream SKIP (no upstream unittest tree found)"
fi

GATES_LOG=/tmp/l1_gates.log; : > "$GATES_LOG"
gates_fail=0
for g in _test_hcu_config _test_fa3_emu_cp _test_rope_fallback_vs_fused \
         _test_aiter_moe_gate _test_contig_triton; do
    f="tests/tools/$g.py"
    [ -f "$f" ] || { echo "    SKIP $g (missing)" >> "$GATES_LOG"; continue; }
    echo "=== $g ===" >> "$GATES_LOG"
    if timeout 300 python3 "$f" >> "$GATES_LOG" 2>&1; then
        echo "L1: gate $g PASS"
    else
        echo "L1: gate $g FAIL -- tail:"; tail -n 8 "$GATES_LOG" | sed 's/^/    /'
        gates_fail=1; FAILS+=("gate:$g")
    fi
done
[ "$gates_fail" = 0 ] && echo "L1: gates PASS"

# vendor wheel sanity: WARN tier (bad wheel is an environment fact, not an
# MR regression) -- printed, never fails the gate
python3 -m pytest tests/vendor/test_flash_attn.py -q --no-header \
    > /tmp/l1_vendor.log 2>&1 \
    && echo "L1: vendor PASS" \
    || echo "L1: vendor WARN (wheel sanity; see /tmp/l1_vendor.log)"

# ---------------------------------------------------------------- verdict
if [ "${#FAILS[@]}" -gt 0 ]; then
    echo "L1: RED  (${FAILS[*]})"
    exit 1
fi
echo "L1: GREEN (gpu=$GPU)"
exit 0
