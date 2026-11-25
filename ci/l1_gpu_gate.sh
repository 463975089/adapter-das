#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

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
#   kernels   tests/kernels (not hisparse) + tests/adapter   triton numerics vs torch refs
#   l15_up    upstream registered/attention/unittests (dsa+kda+dense+mla)
#             via pinned PYTHONPATH; tilelang variants excluded (DTK bug)
#   l15_manifest  L1.5 v2 manifest collect/verify shadow (non-blocking by
#                 default; set L15_MANIFEST_MODE=blocking to gate)
#   l15_family_matrix  Dense/MLA/MoE/Quant/Mamba no-weight representative
#                 matrix (off by default; observe/blocking explicitly enabled)
#   gates     curated tools gates: config catalog, fa3-emu CP shapes, rope
#             paths, aiter-MoE bridge dispatch, deepgemm contiguous triton
#   vendor    flash_attn wheel sanity probe    (WARN tier: bad wheel -> warn)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"

FAILS=()

# L1.5 v2 manifest migration is observe-only by default.  The output root is
# caller-controlled so durable CI artifacts do not depend on /tmp.  Existing
# segment logs remain unchanged for backward compatibility until MR-06.
L15_MANIFEST="$ROOT/tests/manifests/l15_v0522.json"
L15_MANIFEST_MODE="${L15_MANIFEST_MODE:-observe}"
# Include every candidate batch by default.  The selectors are selected by
# "current"; omitting batch-2..4 silently reduced the shadow inventory to the
# first candidate batch and made the default gate look more complete than it
# was.  Callers may still narrow the set explicitly for a staged migration.
L15_MANIFEST_BATCHES="${L15_MANIFEST_BATCHES:-current,batch-1,batch-2,batch-3,batch-4}"
L15_RESULT_ROOT="${L15_RESULT_ROOT:-$ROOT/results/l15-manifest}"
L15_FAMILY_MATRIX_MODE="${L15_FAMILY_MATRIX_MODE:-off}"
L15_FAMILY_MATRIX_FAMILIES="${L15_FAMILY_MATRIX_FAMILIES:-dense,mla,moe,quantization,mamba}"
# L1.5 upstream registered suites are upstream-owned: every kernel they touch
# on HCU is an upstream JIT module (sgl_kernel_jit_kvcache_*, dpsk_v4_*,
# concat_mla_absorb_q, fused_metadata_copy, causal_conv1d, ...), and upstream's
# AMD CI already gates them on AMD hardware. 2026-11-19 decision (user): the CI
# gate tests the adapter, not upstream's kernels -- a cold container paid 17
# modules x ~58s (~16 min) inside l15_exact alone, which is pure iteration lag.
# Default OFF. Manual/nightly re-run: L15_UPSTREAM_MODE=manifest (gate manifest
# + five-family matrix) or dirs (legacy l15_upstream/l15_mech directory runs).
L15_UPSTREAM_MODE="${L15_UPSTREAM_MODE:-off}"
case "$L15_MANIFEST_MODE" in observe|blocking|off) ;; *)
    echo "L1: invalid L15_MANIFEST_MODE='$L15_MANIFEST_MODE' (use off|observe|blocking)"
    exit 2;;
esac
case "$L15_FAMILY_MATRIX_MODE" in observe|blocking|off) ;; *)
    echo "L1: invalid L15_FAMILY_MATRIX_MODE='$L15_FAMILY_MATRIX_MODE' (use off|observe|blocking)"
    exit 2;;
esac
case "$L15_UPSTREAM_MODE" in manifest|dirs|off) ;; *)
    echo "L1: invalid L15_UPSTREAM_MODE='$L15_UPSTREAM_MODE' (use off|manifest|dirs)"
    exit 2;;
esac
L1_KERNELS_HISPARSE="${L1_KERNELS_HISPARSE:-0}"
case "$L1_KERNELS_HISPARSE" in 0|1) ;; *)
    echo "L1: invalid L1_KERNELS_HISPARSE='$L1_KERNELS_HISPARSE' (use 0|1)"
    exit 2;;
esac

# R-17 three-way binding: branch <-> upstream tree <-> container. Explicit
# tree (arg --up-tree <path> / env HCU_UP_TREE) must EXIST -- a typo'd path
# loud-fails instead of silently falling back to auto-probe (wrong-tree
# fake-green is the #1 killer of the dual-line model). Legacy usage
# `l1_gpu_gate.sh [GPU_ID]` unchanged.
UP_TREE=""
GPU_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --up-tree) [ -n "${2:-}" ] || { echo "L1: --up-tree needs a path"; exit 2; }
                   UP_TREE="$2"; shift 2;;
        ''|*[!0-9]*) echo "L1: unknown arg '$1' (supported: [GPU_ID] --up-tree <path>)"; exit 2;;
        *) GPU_ARG="$1"; shift;;
    esac
done

# R-17 three-way binding: branch <-> upstream tree <-> container. Explicit
# tree (arg --up-tree <path> / env HCU_UP_TREE) must EXIST -- a typo'd path
# loud-fails here, BEFORE GPU pick and before any test, instead of silently
# falling back to auto-probe (wrong-tree fake-green is the #1 killer of the
# dual-line model). No override -> auto-probe the legacy layout. The chosen
# tree also feeds the global PYTHONPATH below, so cpu/kernels segments stay
# on the same tree as the l15 segments.
UP_TREE="${UP_TREE:-${HCU_UP_TREE:-}}"
UP_TREE="${UP_TREE%/}"
if [ -n "$UP_TREE" ] && [ ! -d "$UP_TREE/test/registered/attention/unittests" ]; then
    echo "L1: FAIL -- --up-tree/HCU_UP_TREE='$UP_TREE' has no test/registered/attention/unittests (typo? wrong container? refusing auto-probe fallback)"
    exit 2
fi
if [ -z "$UP_TREE" ]; then
  for d in /sglang-v0522 /root/sglang-v0522-src /sglang-v0521 /root/sglang-v0521-src; do
    [ -d "$d/test/registered/attention/unittests" ] && UP_TREE="$d" && break
  done
fi

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

GPU="$(pick_gpu "${GPU_ARG:-}")" || { echo "L1: no free GPU (all >=5% VRAM) -- SKIP/QUEUE"; exit 2; }
export HIP_VISIBLE_DEVICES="$GPU" CUDA_VISIBLE_DEVICES="$GPU"
# Live observability for long segments (2026-10-20): (a) unbuffered pytest
# output so `tail -f /tmp/l1_<seg>.log` streams per-test dots instead of a
# block-buffered void; (b) a per-run JIT progress journal written by the
# adapter's build reporter (hcu_adapter/jit_build_log.py, one line
# per real ninja compile) that run_seg's heartbeat tails. Unique per gate
# run ($$) so a manual gate and a CI gate on the same container
# never cross-read each other's compiles.
export PYTHONUNBUFFERED=1
JPROG="/tmp/hcu_jit_progress.$$.log"; export HCU_JIT_PROGRESS_LOG="$JPROG"
: > "$JPROG"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# some gates exercise upstream classes (rotary etc.) -- mount the sglang
# source the container actually serves from, when present. Both known
# locations: validation-host-26 serves from /sglang-v0521; box16 clones under
# /root/sglang-v0521-src. Without this, tests/kernels imports (e.g.
# kpool_fp8_index) resolve sglang from pip dist-packages, which lacks the
# dsa submodule -> collection error (2026-10-18 rerun2 forensics).
for d in "$UP_TREE/python" /sglang-v0521/python /root/sglang-v0521-src/python; do
    [ -d "$d" ] && export PYTHONPATH="$d:$PYTHONPATH"
done
echo "L1: using GPU $GPU"

# Cold-container relief (bw47 forensics 2026-10-20): import surface varies
# 5-10x across boxes (34s/proc adapter boot, 241s adapter collection on
# bw47) while every window below is nmz-calibrated. Scale all windows via
# HCU_L1_TIMEOUT_SCALE (integer, default 1 = nmz stock) instead of hand-
# editing the script per box. Cache seeding cuts the other axis entirely
# (ci/cache_export.sh / ci/cache_import.sh, bootstrap.md 4d).
TSCALE="${HCU_L1_TIMEOUT_SCALE:-1}"
case "$TSCALE" in ''|*[!0-9]*) TSCALE=1;; esac
[ "$TSCALE" -gt 1 ] && echo "L1: window scale x$TSCALE (cold container)"

run_seg() {  # name timeout_base cmd...
    local name="$1"; local tmo=$(( $2 * TSCALE )); shift 2
    local log="/tmp/l1_${name}.log"
    : > "$log"
    echo "L1: === $name (timeout ${tmo}s) ==="
    # heartbeat: one line/60s = elapsed + last JIT journal entry, so a long
    # compile storm (journal keeps advancing) is tellable from a wedge
    # (frozen last line) without waiting for the timeout. Cancelled after
    # the segment, so short segments just never print one.
    ( t0=$(date +%s); shown=0
      while sleep 60; do
          last="$(tail -n 1 "$JPROG" 2>/dev/null)"
          echo "L1 heartbeat [$name]: elapsed=$(( $(date +%s) - t0 ))s last=${last:-none}"
          # Surface completed log lines while the segment is still running.
          # A timeout must retain the active nodeid and faulthandler stack,
          # rather than only a row of dots after the entire job finishes.
          lines=$(wc -l < "$log")
          if [ "$lines" -gt "$shown" ]; then
              sed -n "$((shown + 1)),${lines}p" "$log" | sed 's/^/    /'
              shown=$lines
          fi
      done ) & hbpid=$!
    if timeout "$tmo" "$@" > "$log" 2>&1; then
        echo "L1: $name PASS"
    else
        local rc=$?
        echo "L1: $name FAIL (rc=$rc) -- tail:"
        tail -n 15 "/tmp/l1_${name}.log" | sed 's/^/    /'
        [ -s "$JPROG" ] && { echo "    jit journal tail:"; tail -n 3 "$JPROG" | sed 's/^/      /'; }
        FAILS+=("$name")
    fi
    kill "$hbpid" 2>/dev/null; wait "$hbpid" 2>/dev/null
}

# ---------------------------------------------------------------- segments
# 300s: l0 is 6-7s warm, but a COLD container's first NFS metadata walk
# of the tree exceeded the old 120s twice (bw47 2026-11-19) -- budget for
# the cold walk, not the scan itself.
# DTK platform runtime wants /var/log/hylog/ for its banner probe;
# containers lack it -- create to silence the "Could not open" line
# (harmless either way; image-side fix is the durable one).
mkdir -p /var/log/hylog 2>/dev/null || true

run_seg l0      300  python3 ci/l0_gate.py
# CPU oracles import sgl_kernel only -- no sglang host surface, so the
# interpreter skips the bridge install (~26 s of sglang import per cold
# start, and with no host loaded the _cpu_process suites also skip their
# clean-subprocess rerun: they already ARE the clean process).
run_seg cpu     300  env HCU_ADAPTER_HOST=none python3 -m pytest tests/test_ops_cpu.py -q --no-header
# Family-representative cases are owned by the family lane, never by the fast
# kernels lane. Measured on the 0.5.22 tree (2026-11-19, seed-warm): they carry
# 8 pre-existing failures (6x quantized_linear_chain fp8/ct-fp8, 2x dsv41
# compression -- byte-identical failures on the pre-zero-fallback base tree
# 759ac7d, so not this branch's regressions) plus a long/likely-hung case; they
# also blow the 900s seed budget. The family lane is manual now
# (L15_UPSTREAM_MODE=manifest); CI keeps the broad suite lean. The transfer
# needs no upstream tree -- it reads the in-repo matrix.
FAMILY_DESELECT=()
if [ -f "$ROOT/tests/manifests/l15_family_matrix_v1.json" ]; then
    family_nodes="$(python3 - "$ROOT/tests/manifests/l15_family_matrix_v1.json" "$L15_FAMILY_MATRIX_FAMILIES" <<'PY'
import json
import sys

from tools.l15_family_matrix import validate

document = json.loads(open(sys.argv[1], encoding="utf-8").read())
errors = validate(document)
if errors:
    raise ValueError("; ".join(errors))
selected = set(sys.argv[2].split(","))
known = {family["id"] for family in document["families"]}
if not selected or selected - known:
    raise ValueError("invalid family selection")
for family in document["families"]:
    if family["id"] in selected:
        for case in family["cases"]:
            if case.get("kind") == "adapter-pytest":
                print(case["nodeid"])
PY
    )" || { echo "L1: invalid family manifest/selection"; exit 2; }
    while IFS= read -r nodeid; do
        [ -n "$nodeid" ] && FAMILY_DESELECT+=(--deselect "$nodeid")
    done <<< "$family_nodes"
    echo "L1: kernels transfers $((${#FAMILY_DESELECT[@]} / 2)) family-representative cases to the manual family lane"
fi
# 900s = the seed-contract budget, now with W4 evidence: xdist-4 measured
# 391-396s (acceptance rounds 3/4, 2026-11-19) WITH 7-18 cold JIT builds
# inside; tree-frozen + seeded containers run colder still. The 1500s
# interim (1122s serial, pre-W4) is retired -- cold container blowing this
# exports a new seed; the budget does not move.
# W4-3: xdist parallelizes the kernels tail (fork workers on Linux: torch
# imports once per worker, ~1870 small tests spread over 4 lanes). Opt out
# with L1_KERNELS_XDIST=0; absent pytest-xdist degrades to serial. Worker
# count 4: the concurrent gates phase already shares one HCU safely, so 4
# pytest workers on small tensors is the proven concurrency envelope.
XDIST_FLAGS=()
if [ "${L1_KERNELS_XDIST:-1}" != "0" ] && python3 -c "import xdist" 2>/dev/null; then
    XDIST_FLAGS=(-n 4 --dist loadfile)
else
    # zero-fallback policy applies to our own gate: a missing speed
    # dependency must be VISIBLE, not a silent serial degrade (2026-11-19:
    # the acceptance round ran serial for 15 min before anyone noticed
    # pytest-xdist was absent from the image).
    echo "L1: WARN pytest-xdist absent -- kernels tail runs SERIAL (~250s slower); pip install pytest-xdist in the image"
fi
# tests/kernels/hisparse needs the HiSparse native library built and installed
# (setup_hip.py --hisparse-only, or conftest's --hisparse-library). Neither this
# gate nor CI builds it, so on a stock image its three files error in setup:
# 104 errors on main since #159 (2026-11-24). The suite's own header reserves it
# for an explicitly authorized lane -- docs/frameworks/vllm033-hisparse.md,
# "Reproduce the isolated native lane" -- so the default kernels segment leaves
# that directory to it. L1_KERNELS_HISPARSE=1 collects it here once the library
# is installed.
# ${VAR:-0}, not $VAR: tests/adapter/test_l1_gate_ownership.py runs this block
# on its own under set -u, without the defaults assigned at the top of the file.
KERNELS_IGNORE=()
if [ "${L1_KERNELS_HISPARSE:-0}" != "1" ] && [ -d "$ROOT/tests/kernels/hisparse" ]; then
    KERNELS_IGNORE=(--ignore=tests/kernels/hisparse)
    echo "L1: WARN kernels leaves tests/kernels/hisparse to its own lane (needs the HiSparse native library; L1_KERNELS_HISPARSE=1 collects it here)"
fi
run_seg kernels 900  python3 -m pytest tests/kernels tests/adapter -vv --no-header -m "not emu_parity and not fresh_process" "${XDIST_FLAGS[@]}" \
    --tb=short -o faulthandler_timeout=120 --durations=60 \
    "${FAMILY_DESELECT[@]}" "${KERNELS_IGNORE[@]}"

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
if [ -n "$UP_TREE" ] && [ "$L15_UPSTREAM_MODE" != "off" ]; then
    echo "L1: upstream tree = $UP_TREE (L15_UPSTREAM_MODE=$L15_UPSTREAM_MODE)"
    UP="$UP_TREE/test/registered/attention/unittests"
    if [ "$L15_UPSTREAM_MODE" = "manifest" ] || [ "$L15_FAMILY_MATRIX_MODE" = "blocking" ]; then
        ownership_out="$L15_RESULT_ROOT/ownership-$(date +%Y%m%dT%H%M%S)-$$"
        python3 -m tools.l15_ci_plan --families "$L15_FAMILY_MATRIX_FAMILIES" \
            --out-dir "$ownership_out" || exit 2
        # Green nodes have one owner; known diagnostics/platform baselines are
        # listed in ownership.json as NOT EXECUTED, never as a successful test.
        run_seg l15_exact 1800 env HCU_ADAPTER_HOST=none HCU_DEEPGEMM_BRIDGE=1 \
            PYTHONPATH="$ROOT:$UP_TREE/python" python3 -m tools.l15_manifest run \
            --up-tree "$UP_TREE" --manifest "$ownership_out/manifest.json" \
            --classifications core-green,contract-green --out-dir "$ownership_out/execution"
        if [ -s "$ownership_out/observe-nodeids.txt" ]; then
            env HCU_ADAPTER_HOST=none HCU_DEEPGEMM_BRIDGE=1 \
                PYTHONPATH="$ROOT:$UP_TREE/python" python3 -m tools.l15_manifest run \
                --up-tree "$UP_TREE" --manifest "$ownership_out/observation-manifest.json" \
                --classifications candidate-observe --out-dir "$ownership_out/observation" \
                || echo "L1: mixed subtest observation WARN; not core coverage"
        fi
    else
    run_seg l15_upstream 1800 env HCU_DEEPGEMM_BRIDGE=1 \
        PYTHONPATH="$UP_TREE/python:$ROOT" \
        python3 -m pytest "$UP/dsa" "$UP/kda" "$UP/dense" "$UP/mla" \
            -q --no-header \
        -k "not tilelang" -o faulthandler_timeout=580 --tb=line
    # L1.5b (R-14 batch A): upstream MECHANISM dirs -- gdn/hybrid_linear/
    # mamba/swa/lightning (Qwen3-Next family attention faces) + dsv4. Same
    # env contract as l15_upstream; most GPU-less cases self-skip via the
    # kit support checks. Missing dirs (older trees) are simply omitted.
    MECH=""
    for d in gdn hybrid_linear mamba swa lightning dsv4; do
        [ -d "$UP/$d" ] && MECH="$MECH $UP/$d"
    done
    # v0.5.23 face drift: cutedsl_gdn_mtp_ring imports cuda.bindings at
    # module level, so collecting gdn/test_gdn_cutedsl_ring_verify.py dies
    # with ModuleNotFoundError on HCU containers (no cuda python pkg) and
    # takes the whole l15_mech segment down at collection. Probe the import
    # under the segment's own path contract; on failure ignore that one
    # file loudly. Platform-bug ledger treatment, same spirit as
    # `-k "not tilelang"` (docs/ci.md L1.5b).
    MECH_IGNORE=""
    GDN_VERIFY="$UP/gdn/test_gdn_cutedsl_ring_verify.py"
    if [ -f "$GDN_VERIFY" ] && ! env HCU_DEEPGEMM_BRIDGE=1 \
            PYTHONPATH="$UP_TREE/python:$ROOT" \
            python3 -c "import sglang.kernels.ops.attention.cutedsl_gdn_mtp_ring" \
            >/dev/null 2>&1; then
        MECH_IGNORE="--ignore=$GDN_VERIFY"
        echo "L1: l15_mech ignores $GDN_VERIFY (cutedsl_gdn_mtp_ring not importable on this stack; ledger docs/ci.md L1.5b)"
    fi
    if [ -n "$MECH" ]; then
        # Batch A first-run triage (2026-10-19): baseline 38 failed / 99
        # passed -- every one ledgered in docs/ci.md L1.5b (gdn cuteDSL
        # toolchain x2, dsv4 JIT/l1c/page-table/numeric faces, mamba2
        # triton page-boundary x3). Trip-wire on the COUNTS, not on
        # pytest rc: ledgered failures are tolerated, a regression
        # (passed below floor / failed above ceiling) is not.
        echo "L1: === l15_mech (timeout $(( 1200 * TSCALE ))s, known-fail budget) ==="
        ( t0=$(date +%s)
          while sleep 60; do
              last="$(tail -n 1 "$JPROG" 2>/dev/null)"
              echo "L1 heartbeat [l15_mech]: elapsed=$(( $(date +%s) - t0 ))s last=${last:-none}"
          done ) & hbpid=$!
        rc=0
        timeout $(( 1200 * TSCALE )) env HCU_DEEPGEMM_BRIDGE=1 \
            PYTHONPATH="$UP_TREE/python:$ROOT" \
            python3 -m pytest $MECH $MECH_IGNORE -q --no-header \
            -k "not tilelang" -rs -o faulthandler_timeout=580 --tb=line \
            > /tmp/l1_l15_mech.log 2>&1 || rc=$?
        kill "$hbpid" 2>/dev/null; wait "$hbpid" 2>/dev/null
        line="$(grep -E '[0-9]+ failed' /tmp/l1_l15_mech.log | tail -1)"
        nfail="$(sed -n 's/^[^0-9]*\([0-9][0-9]*\) failed.*/\1/p' <<<"$line")"
        npass="$(sed -n 's/^[^0-9]*[0-9][0-9]* failed, \([0-9][0-9]*\) passed.*/\1/p' <<<"$line")"
        nfail="${nfail:-999}"; npass="${npass:-0}"
        if [ "$rc" -eq 124 ]; then
            echo "L1: l15_mech FAIL (timeout)"
            FAILS+=(l15_mech)
        elif [ "$nfail" -le 42 ] && [ "$npass" -ge 90 ]; then
            echo "L1: l15_mech PASS (within known-fail budget: $nfail failed / $npass passed; ledger docs/ci.md L1.5b)"
        else
            echo "L1: l15_mech FAIL (outside known-fail budget: $nfail failed / $npass passed vs baseline 38/99) -- tail:"
            tail -n 15 /tmp/l1_l15_mech.log | sed 's/^/    /'
            [ -s "$JPROG" ] && { echo "    jit journal tail:"; tail -n 3 "$JPROG" | sed 's/^/      /'; }
            FAILS+=(l15_mech)
        fi
    else
        echo "L1: l15_mech SKIP (no mechanism dirs in this tree)"
    fi

    fi

    # L1.5 v2 manifest shadow.  This collects explicit candidate sources and
    # exact nodeids without modifying the upstream tree.  During migration the
    # legacy directory/count gates above remain authoritative.  Observe mode
    # reports drift and infrastructure failures without adding FAILS; blocking
    # mode is opt-in.
    if [ "$L15_MANIFEST_MODE" != "off" ] && [ -f "$L15_MANIFEST" ]; then
        run_id="$(date +%Y%m%dT%H%M%S)-$$"
        manifest_out="$L15_RESULT_ROOT/$run_id"
        mkdir -p "$manifest_out"
        echo "L1: === l15_manifest ($L15_MANIFEST_MODE, batches=$L15_MANIFEST_BATCHES) ==="
        manifest_rc=0
        env HCU_DEEPGEMM_BRIDGE=1 PYTHONPATH="$UP_TREE/python:$ROOT" \
            python3 tools/l15_manifest.py collect \
                --up-tree "$UP_TREE" \
                --manifest "$L15_MANIFEST" \
                --out-dir "$manifest_out" \
                --batches "$L15_MANIFEST_BATCHES" \
                --timeout $(( 900 * TSCALE )) || manifest_rc=$?
        verify_rc=0
        if [ -f "$manifest_out/collect.json" ]; then
            python3 tools/l15_manifest.py verify \
                --manifest "$L15_MANIFEST" \
                --collect "$manifest_out/collect.json" \
                --mode "$L15_MANIFEST_MODE" \
                --out "$manifest_out/verify.json" || verify_rc=$?
        else
            verify_rc=2
        fi
        if [ "$manifest_rc" -eq 0 ] && [ "$verify_rc" -eq 0 ]; then
            echo "L1: l15_manifest PASS ($L15_MANIFEST_MODE; artifacts=$manifest_out)"
        elif [ "$L15_MANIFEST_MODE" = "blocking" ]; then
            echo "L1: l15_manifest FAIL (collect=$manifest_rc verify=$verify_rc; artifacts=$manifest_out)"
            FAILS+=(l15_manifest)
        else
            echo "L1: l15_manifest WARN (observe-only; collect=$manifest_rc verify=$verify_rc; artifacts=$manifest_out)"
        fi
    elif [ "$L15_MANIFEST_MODE" = "off" ]; then
        echo "L1: l15_manifest SKIP (mode=off)"
    elif [ "$L15_MANIFEST_MODE" = "blocking" ]; then
        echo "L1: l15_manifest FAIL (manifest missing: $L15_MANIFEST)"
        FAILS+=(l15_manifest)
    else
        echo "L1: l15_manifest WARN (observe-only; manifest missing: $L15_MANIFEST)"
    fi

    if [ "$L15_FAMILY_MATRIX_MODE" != "off" ]; then
        family_out="$L15_RESULT_ROOT/family-$(date +%Y%m%dT%H%M%S)-$$.json"
        echo "L1: === l15_family_matrix ($L15_FAMILY_MATRIX_MODE, families=$L15_FAMILY_MATRIX_FAMILIES) ==="
        family_rc=0
        env HCU_DEEPGEMM_BRIDGE=1 PYTHONPATH="$ROOT:$UP_TREE/python" \
            python3 tools/l15_family_matrix.py run \
                --up-tree "$UP_TREE" \
                --adapter-root "$ROOT" \
                --families "$L15_FAMILY_MATRIX_FAMILIES" \
                --out "$family_out" || family_rc=$?
        if [ "$family_rc" -eq 0 ]; then
            echo "L1: l15_family_matrix PASS (artifact=$family_out)"
        elif [ "$L15_FAMILY_MATRIX_MODE" = "blocking" ]; then
            echo "L1: l15_family_matrix FAIL (rc=$family_rc; artifact=$family_out)"
            FAILS+=(l15_family_matrix)
        else
            echo "L1: l15_family_matrix WARN (observe-only; rc=$family_rc; artifact=$family_out)"
        fi
    else
        echo "L1: l15_family_matrix SKIP (mode=off)"
    fi
elif [ -n "$UP_TREE" ]; then
    echo "L1: L1.5 upstream suites SKIP (L15_UPSTREAM_MODE=off; manual: L15_UPSTREAM_MODE=manifest|dirs)"
else
    echo "L1: l15_upstream SKIP (no upstream unittest tree found)"
fi

GATES_LOG=/tmp/l1_gates.log; : > "$GATES_LOG"
gates_fail=0
# The gates are independent processes with independent env contracts;
# each pays ~26-30 s of interpreter+torch import and a few GB of VRAM on a
# 96 GB card, so they run concurrently and the phase costs the slowest gate
# (~100 s) instead of the sum (~330 s; 2026-11-17 nmz16 cold-start timing).
# Per-gate log files are merged into GATES_LOG in fixed name order afterwards.
gate_names=(_test_hcu_config _test_fa3_emu_cp _test_rope_fallback_vs_fused \
           _test_aiter_moe_gate _test_contig_triton _test_fresh_process_class \
           _test_zero_degradation)
declare -A gate_pid gate_out
# review #152 P3: count derives from the array, never hardcoded again.
echo "L1: gates start (${#gate_names[@]} concurrent)"
for g in "${gate_names[@]}"; do
    f="tests/tools/$g.py"
    [ -f "$f" ] || { echo "    SKIP $g (missing)" >> "$GATES_LOG"; continue; }
    # The config gate only exercises config resolution + kernel tiers;
    # skipping the host-bridge install halves its interpreter startup
    # (52s -> 26s on the 2026-11-16 image) and it has no bridge surface.
    case "$g" in _test_hcu_config) gate_env=(env HCU_ADAPTER_HOST=none) ;;
        _test_fresh_process_class) gate_env=() ;;
        *) gate_env=() ;; esac
    # W4-2: the fresh-process lane pays ~26s interpreter+torch per spawn
    # (~370s across 9 tests) -- it needs a wider budget than the fixed 300s.
    case "$g" in _test_fresh_process_class) gate_tmo=900 ;; *) gate_tmo=300 ;; esac
    gate_out[$g]="/tmp/l1_gate_$g.log"
    timeout "$gate_tmo" "${gate_env[@]}" python3 "$f" > "${gate_out[$g]}" 2>&1 &
    gate_pid[$g]=$!
done
for g in "${gate_names[@]}"; do
    [ -n "${gate_pid[$g]:-}" ] || continue
    if wait "${gate_pid[$g]}"; then
        echo "L1: gate $g PASS"
    else
        echo "L1: gate $g FAIL -- tail:"; tail -n 8 "${gate_out[$g]}" | sed 's/^/    /'
        gates_fail=1; FAILS+=("gate:$g")
    fi
    { echo "=== $g ==="; cat "${gate_out[$g]}"; } >> "$GATES_LOG"
done
[ "$gates_fail" = 0 ] && echo "L1: gates PASS"

# vendor wheel sanity: WARN tier (bad wheel is an environment fact, not an
# MR regression) -- printed, never fails the gate
# Vendor probe drives the vendor wheels directly against a torch fp32
# reference; the sglang host bridge is not under test here, skip its
# install (~26 s cold interpreter).
env HCU_ADAPTER_HOST=none python3 -m pytest tests/vendor/test_flash_attn.py -q --no-header \
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
