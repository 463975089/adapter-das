"""moe_c W4A16 vendor tier (2026-10-26, box16 dsv41f-INT4): regression pins.

Pinned here at the pure-logic layer (torch tensors, no aiter / no sglang /
no GPU):

  * pad law — gemm2 K rides (K/32)%16==0 (256 special-case): 2304 pads to
    2560, 512/1024/2048 stay, 384 pads to 512;
  * route predicate — dsv41f-INT4 per-rank face passes; wrong dtype,
    sub-32 groups, gemm1 K-law misses, N%256 misses, tiny I, missing
    scales each fall through;
  * activation formula — the exact dsv4 clamp (silu(min(g,lim))*clamp(u,
    ±lim)), NOT aiter's "situ" softcap;
  * buffer pad discipline — a2 pad columns stay zero across calls;
  * wrapper scoping — non-int4 faces pass straight through;
  * sitecustomize wiring (10.7 outermost) + REGISTRY/env-vars.md
    reconciliation (L1 gate).

Run: python tests/adapter/test_moe_c_w4a16.py  (或 pytest)
"""

import os
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sgl_kernel._adapters import hcu_env, moe_c_w4a16  # noqa: E402


def test_pad_law_matches_wheel_ks():
    # v7 box16 sweep: {256,512,1024,1536,2048} pass, {384,768,2304} fail;
    # law (K/32)%16==0 with 256 special-cased by the wheel
    assert moe_c_w4a16.pad_intermediate(256) == 256
    assert moe_c_w4a16.pad_intermediate(512) == 512
    assert moe_c_w4a16.pad_intermediate(1024) == 1024
    assert moe_c_w4a16.pad_intermediate(1536) == 1536
    assert moe_c_w4a16.pad_intermediate(2048) == 2048
    assert moe_c_w4a16.pad_intermediate(384) == 512
    assert moe_c_w4a16.pad_intermediate(768) == 1024
    assert moe_c_w4a16.pad_intermediate(2304) == 2560  # dsv41f-INT4


def _dsv41f_face():
    E, H, I = 48, 5120, 2304
    w1s = (E, 2 * I, H // 2)
    w2s = (E, H, I // 2)
    s1s = (E, 2 * I, 1)   # channel
    s2s = (E, H, 1)
    return w1s, w2s, s1s, s2s, torch.bfloat16


def test_route_ok_dsv41f_and_misses():
    ok, why = moe_c_w4a16.route_ok(*_dsv41f_face())
    assert ok, why
    w1s, w2s, s1s, s2s, _ = _dsv41f_face()
    # fp16 activations fall through
    assert not moe_c_w4a16.route_ok(w1s, w2s, s1s, s2s, torch.float16)[0]
    # missing scales (shape (0,))
    assert not moe_c_w4a16.route_ok(w1s, w2s, (0,), s2s, torch.bfloat16)[0]
    # gemm1 K-law: K=4224 (132 groups, %16!=0) misses
    assert not moe_c_w4a16.route_ok(
        (48, 4608, 4224 // 2), (48, 5120, 2304 // 2), (48, 4608, 1),
        (48, 5120, 1), torch.bfloat16)[0]
    # N%256: N1=4480 misses
    assert not moe_c_w4a16.route_ok(
        (48, 4480, 2560), (48, 5120, 1152), (48, 4480, 1), (48, 5120, 1),
        torch.bfloat16)[0]
    # tiny intermediate (I=128 < 256; N1=256 keeps the other guards green)
    assert not moe_c_w4a16.route_ok(
        (48, 256, 2560), (48, 5120, 64), (48, 256, 1), (48, 5120, 1),
        torch.bfloat16)[0]
    # group-16 (finer than 32) scales fall through
    assert not moe_c_w4a16.route_ok(
        (48, 4608, 2560), (48, 5120, 1152), (48, 4608, 5120 // 16),
        (48, 5120, 1), torch.bfloat16)[0]


def test_clamp_formula_is_exact_dsv4_not_aiter_situ():
    torch.manual_seed(0)
    I, limit = 64, 7.0
    x = torch.randn(8, 2 * I, dtype=torch.float32) * 12  # exercise caps
    out = torch.empty(8, I, dtype=torch.bfloat16)
    moe_c_w4a16._clamp_silu(x, out, limit)
    g = x[:, :I].clamp(max=limit)
    u = x[:, I:].clamp(min=-limit, max=limit)
    ref = torch.nn.functional.silu(g) * u
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)
    # not the softcap: beta1*tanh(g/b1)*sigmoid(g)*beta2*tanh(u/b2) differs
    soft = (4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)
            * 25.0 * torch.tanh(u / 25.0))
    assert not torch.allclose(out.float(), soft, atol=5e-2)
    # plain silu when no limit
    out2 = torch.empty(8, I, dtype=torch.bfloat16)
    moe_c_w4a16._clamp_silu(x, out2, None)
    ref2 = torch.nn.functional.silu(x[:, :I]) * x[:, I:]
    assert torch.allclose(out2.float(), ref2, atol=2e-2, rtol=2e-2)


def test_a2_pad_columns_stay_zero():
    # the persistent activation buffer's pad columns are written exactly
    # once (zero fill) and never touched afterwards -- graph replays keep
    # them zero. Emulate two "replays" incl. a garbage-preloaded c1.
    E, H, I = 4, 512, 384  # I=384 -> pads to 512 (real pad columns exist)
    IP = moe_c_w4a16.pad_intermediate(I)
    assert IP == 512
    c1, a2, c3, act = moe_c_w4a16._buffers(8, 2 * I, IP, H, torch.device("cpu"))
    assert a2.shape == (8, IP)
    c1.normal_()
    moe_c_w4a16._clamp_silu(c1, act, 7.0)
    a2[:, :I].copy_(act)
    assert (a2[:, I:] == 0).all()
    a2[:, :I].normal_()  # second call writes only the live region again
    assert (a2[:, I:] == 0).all()
    # buffers keyed by rows -> distinct batch sizes get distinct sets
    c1b, a2b, _, _ = moe_c_w4a16._buffers(16, 2 * I, IP, H, torch.device("cpu"))
    assert a2b is not a2


def test_wrapper_scopes_to_int4_face():
    src = pathlib.Path(moe_c_w4a16.__file__).read_text(encoding="utf-8")
    # engage gate checks the int4 kwarg; everything else falls through
    assert 'kwargs.get("use_int4_w4a16")' in src
    # zero-point / no_combine / apply-on-input guards present
    assert "w1_zp" in src and "no_combine" in src
    assert "apply_router_weight_on_input" in src
    # fallback to the next arm never raises outward
    assert "call failed -> next arm" in src


def test_registry_entry_and_manual_reconcile():
    entry = hcu_env.REGISTRY.get("HCU_W4A16_MOE_IMPL")
    assert entry is not None, "REGISTRY key missing"
    assert "**moe_c**=" in entry["about"] or "moe_c**=" in entry["about"]
    manual = (ROOT / "docs" / "env-vars.md").read_text(encoding="utf-8")
    row = [ln for ln in manual.splitlines()
           if ln.startswith("| `HCU_W4A16_MOE_IMPL`")]
    assert row and "moe_c" in row[0], "env-vars.md row missing moe_c tier"
    assert "moe_c_w4a16.py" in row[0]


def test_sitecustomize_wiring_outermost():
    src = (ROOT / "sitecustomize.py").read_text(encoding="utf-8")
    assert "moe_c_w4a16 as _mcw" in src
    assert "_mcw.install()" in src
    # chained after every other fused_experts arm (10 -> 10.5 -> 10.6 -> 10.7)
    assert src.index("aiter_moe as _amo") < src.index("w4a16_moe as _w16m")
    assert src.index("w4a16_moe as _w16m") < src.index(
        "wna16_channel_moe as _wnch")
    assert src.index("wna16_channel_moe as _wnch") < src.index(
        "moe_c_w4a16 as _mcw")


def test_mode_gate_default_off():
    os.environ.pop("HCU_W4A16_MOE_IMPL", None)
    v, _src = hcu_env.resolve("HCU_W4A16_MOE_IMPL")
    assert (v or "auto").strip().lower() not in moe_c_w4a16._MODE_MOEC


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
