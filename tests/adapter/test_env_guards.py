"""box16 bring-up env guards (2026-10-25): regression pins.

Three auto-adaptations born from the box16 stall-campaign bring-up, each
pinned at its pure-logic layer (no torch / no GPU / any machine):

  * jit_compat  — __nv_* token aliases in the cuda_fp8.h / cuda_bf16.h
                  forwarders (proven alias set: box16 dtk2604, 43/43
                  sgl-kernel JIT modules compiled; origin nmz26).
  * topo guard  — NCCL_TOPO_FILE <gpu>-count vs visible-count decision
                  table + env action (fatal XGMI "could not find GPU
                  5000" on 4-proc subsets).
  * watchdog    — cold-cache detection thresholds + the 300->1500s
                  floor rule (explicit --watchdog-timeout always wins).

REGISTRY <-> docs/env-vars.md reconciliation for the two new keys mirrors
the L1 gate of test_sitecustomize_sequence.py (多一漏一即红).

Run: python tests/adapter/test_env_guards.py  (或 pytest)
"""

import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hcu_adapter import hcu_env, jit_compat, nccl_topo_guard, watchdog_cold  # noqa: E402

# Real topology-XML shape (nmz26 /usr/local/built-in-508-topo-input-tj-default.xml).
_TOPO_2GPU = """<system version="2">
  <cpu numaid="0" arch="x86_64" vendor="HygonGenuine">
    <pci busid="0000:09:00.0" class="0x0b4000">
      <gpu dev="0" sm="80" gcn="gfx938" rank="0" gdr="1">
        <xgmi target="0000:9b:00.0" count="1" tclass="0x068000"/>
      </gpu>
    </pci>
    <pci busid="0000:55:00.0" class="0x0b4000">
      <gpu dev="2" sm="80" gcn="gfx938" rank="2" gdr="1"/>
    </pci>
    <pci busid="0000:03:00.0" class="0x020700">
      <nic><net name="shca_0" dev="0" speed="400000" port="1" gdr="1"/></nic>
    </pci>
  </cpu>
</system>
"""


# ---------------------------------------------------------------- jit_compat
def test_fp8_forwarder_carries_proven_nv_aliases():
    body = jit_compat._FORWARDS["cuda_fp8.h"]
    assert body[0] == "#pragma once"
    assert body[1] == "#include <hip/hip_fp8.h>"
    for needle in (
        "using __nv_fp8x2_storage_t = __hip_fp8x2_storage_t;",
        # _fnuz suffix on the HIP side — the first hand-shim on box16
        # used the unsuffixed name and the header did not exist.
        "using __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3_fnuz;",
        "#define __NV_E4M3 __HIP_E4M3",
        "#define __NV_SATFINITE __HIP_SATFINITE",
        "#define __nv_cvt_float2_to_fp8x2 __hip_cvt_float2_to_fp8x2",
        "#define __nv_cvt_fp8x2_to_halfraw2 __hip_cvt_fp8x2_to_halfraw2",
    ):
        assert needle in body, needle
        assert body.count(needle) == 1, needle


def test_bf16_forwarder_carries_nv_bfloat16():
    body = jit_compat._FORWARDS["cuda_bf16.h"]
    assert body[:2] == ["#pragma once", "#include <hip/hip_bf16.h>"]
    assert "using __nv_bfloat16 = __hip_bfloat16;" in body


def test_forwarder_set_unchanged():
    assert set(jit_compat._FORWARDS) == {
        "cuda_bf16.h", "cuda_fp16.h", "cuda_fp8.h",
        "cuda.h", "cuda_runtime.h", "vector_types.h",
    }


# ---------------------------------------------------------------- topo guard
def test_topo_gpu_count_real_xml_shape():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "topo.xml"
        p.write_text(_TOPO_2GPU, encoding="utf-8")
        assert nccl_topo_guard._topo_gpus(str(p)) == 2  # <nic> not counted
        assert nccl_topo_guard._topo_gpus(str(pathlib.Path(d) / "gone.xml")) == 0


def test_check_decision_table():
    with tempfile.TemporaryDirectory() as d:
        p = str(pathlib.Path(d) / "topo.xml")
        pathlib.Path(p).write_text(_TOPO_2GPU, encoding="utf-8")
        # the box16 fatality: 8-card preset vs 4 visible -> clear
        _patch_gpus(p, 8)
        a, why = nccl_topo_guard._check(p, 4)
        assert a == "clear" and "8" in why and "4" in why
        # full-card config: preset matches visibility -> keep
        a, _ = nccl_topo_guard._check(p, 8)
        assert a == "keep"
        # undecidable -> conservative keep
        a, _ = nccl_topo_guard._check(p, 0)
        assert a == "keep"


def _patch_gpus(path, n):
    """Rewrite the sample XML to carry n <gpu> nodes."""
    one = '<gpu dev="0" sm="80" gcn="gfx938" rank="0" gdr="1"/>'
    pathlib.Path(path).write_text(
        "<system>" + one * n + "</system>", encoding="utf-8"
    )
    return True


def test_install_clears_env_on_mismatch():
    key = "NCCL_TOPO_FILE"
    with tempfile.TemporaryDirectory() as d:
        p = str(pathlib.Path(d) / "topo.xml")
        _patch_gpus(p, 8)
        old = os.environ.get(key)
        os.environ[key] = p
        try:
            orig = nccl_topo_guard._visible_gpus
            nccl_topo_guard._visible_gpus = lambda: 4
            nccl_topo_guard._registered = False  # fresh registry attempt
            nccl_topo_guard.install()
            assert key not in os.environ, "mismatch must clear the preset"
            # match arm keeps it
            os.environ[key] = p
            nccl_topo_guard._visible_gpus = lambda: 8
            nccl_topo_guard.install()
            assert os.environ.get(key) == p
            # debug tier opt-out wins even on mismatch
            nccl_topo_guard._visible_gpus = lambda: 4
            os.environ["HCU_NCCL_TOPO_GUARD"] = "0"
            try:
                nccl_topo_guard.install()
                assert os.environ.get(key) == p
            finally:
                del os.environ["HCU_NCCL_TOPO_GUARD"]
        finally:
            nccl_topo_guard._visible_gpus = orig
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


# ---------------------------------------------------------------- watchdog
def test_maybe_bump_matrix():
    orig = watchdog_cold.caches_cold
    try:
        for cold, value, soft, want, bumped in (
            (True, 300.0, False, 1500.0, True),   # cold + default -> floor
            (True, 300, False, 1500.0, True),     # int-typed default too
            (False, 300.0, False, 300.0, False),  # warm -> stock
            (True, 900.0, False, 900.0, False),   # explicit wins
            (True, 1500.0, False, 1500.0, False), # explicit == floor, no bump flag
            (True, 300.0, True, 300.0, False),    # soft watchdog untouched
            (True, None, False, None, False),     # soft default None path
        ):
            watchdog_cold.caches_cold = lambda c=cold: c
            got, did = watchdog_cold.maybe_bump(value, soft=soft)
            assert (got, did) == (want, bumped), (cold, value, soft, got, did)
    finally:
        watchdog_cold.caches_cold = orig


def test_caches_cold_thresholds():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        # all roots thin/absent -> cold (the box16 cold-boot case)
        os.environ["TRITON_CACHE_DIR"] = str(root / "triton")
        os.environ["TORCH_EXTENSIONS_DIR"] = str(root / "ext")
        try:
            assert watchdog_cold.caches_cold() is True
            # seeded triton cache (layout root/<hash>/file) -> warm
            sub = root / "triton" / "deadbeef"
            sub.mkdir(parents=True)
            for i in range(watchdog_cold._COLD_FILES):
                (sub / f"k{i}.json").write_text("{}", encoding="utf-8")
            assert watchdog_cold.caches_cold() is False
        finally:
            os.environ.pop("TRITON_CACHE_DIR", None)
            os.environ.pop("TORCH_EXTENSIONS_DIR", None)


def test_count_files_capped():
    with tempfile.TemporaryDirectory() as d:
        sub = pathlib.Path(d) / "c"
        sub.mkdir()
        for i in range(500):
            (sub / f"f{i}").write_text("x", encoding="utf-8")
        cap = watchdog_cold._COLD_FILES * 2
        assert watchdog_cold._count_files(str(sub)) == cap  # bounded, not 500


# ------------------------------------------------- registry <-> manual (L1)
def test_registry_rows_in_manual():
    manual = (ROOT / "docs" / "env-vars.md").read_text(encoding="utf-8")
    for name in ("HCU_NCCL_TOPO_GUARD", "HCU_JIT_WATCHDOG"):
        assert name in hcu_env.REGISTRY
        assert f"`{name}`" in manual, f"{name} missing from docs/env-vars.md"


def test_bootstrap_bridges_wire_both_guards():
    common = (ROOT / "hcu_adapter" / "bridges" / "common.py").read_text(
        encoding="utf-8"
    )
    sglang = (ROOT / "hcu_adapter" / "bridges" / "sglang.py").read_text(
        encoding="utf-8"
    )
    # common.install runs before the host bridge (test_bootstrap.py); keep
    # topology hygiene in common and the scheduler wrapper in the host chain.
    assert "nccl_topo_guard.install()" in common
    # watchdog wraps the factory before scheduler.py imports it
    assert "watchdog_cold.install()" in sglang


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
