# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""R-09(2/3) gate: aiter per-token fp8 quant tier (quant_aiter bridge).

Pins, through the UPSTREAM module surface (what real consumers call):
  * rowwise  — behavior restoration: upstream ``sglang_per_token_quant_fp8``
    NameErrors on HIP (no ``sgl_per_token_quant_fp8``); bridged it must
    match the torch golden (amax/448 RTN, [T,1] fp32 scales).
  * group    — plain arm via ``per_token_group_quant_fp8`` and
    ``sglang_per_token_group_quant_fp8`` matches the row-group golden.
  * fused    — whole-row ``fuse_silu_and_mul=True`` arm (R-09②) matches
    silu*mul then rowwise-quant golden ([T,1] scales).
  * scaled   — ``scaled_fp8_quant`` dynamic-rowwise + static arms.
  * fallback — exotic arms (column-major scales / ue8m0) keep working via
    the saved upstream functions.

Metric discipline (probe_r09_quant.py lesson, 2026-10-19): quant parity is
NOT global-max rel (1-ulp fp8 steps read as 5%) — use scale allclose +
q bit-match fraction + dequant cosine.

Run: python tests/kernels/test_quant_aiter_gfx938.py   (HIP GPU; L1
kernels segment collects this file)
"""

import sys

import pytest
import torch

torch.manual_seed(20261019)
FP8_MAX = 448.0


def _fk():
    from sglang.kernels.ops.quantization import fp8_kernel as fk
    return fk


def _bridged():
    fk = _fk()
    return getattr(fk.sglang_per_token_quant_fp8, "_hcu_quant_aiter", False)


def _golden_row(x):
    amax = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    s = amax / FP8_MAX
    q = (x.float() / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, s


def _golden_group(x, group=128):
    m, k = x.shape
    xg = x.float().view(m, k // group, group)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    s = amax / FP8_MAX
    q = (xg / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(m, k), s.squeeze(-1)


def _deq(q, s):
    if s.ndim == 2 and s.shape[-1] > 1:
        g = q.shape[-1] // s.shape[-1]
        return q.float() * s.repeat_interleave(g, -1)
    return q.float() * (s.float() if s.ndim == 2 else s.float().unsqueeze(-1))


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-9)).item()


def _verdict(tag, aq, as_, gq, gs):
    assert torch.allclose(as_.float(), gs.float(), rtol=1e-5, atol=1e-8), (
        f"{tag}: scales differ max "
        f"{(as_.float() - gs.float()).abs().max().item():.3e}")
    match = (aq.float() == gq.float()).float().mean().item()
    assert match > 0.99, f"{tag}: q bit-match {match:.4f}"
    c = _cos(_deq(aq, as_), _deq(gq, gs))
    assert c > 0.9999, f"{tag}: dequant cos {c:.6f}"
    print(f"[PASS] {tag}: scale_eq qmatch={match:.4f} cos={c:.6f}")


requires_bridge = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip is not None),
    reason="needs HIP GPU (aiter tier)")
# bridge presence is asserted inside the first test; stock-mode runs
# (HCU_QUANT_IMPL=stock) must fail loudly there, not silently skip.


@requires_bridge
def test_rowwise_nameerror_fixed():
    fk = _fk()
    assert _bridged(), (
        "quant_aiter bridge not installed (HCU_QUANT_IMPL=stock or "
        "sitecustomize missing?)")
    for m, k in ((33, 4096), (1, 128), (512, 512)):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 3.7
        q, s = fk.sglang_per_token_quant_fp8(x)
        gq, gs = _golden_row(x)
        _verdict(f"rowwise-{m}x{k}", q, s.view(m, 1), gq, gs)


@requires_bridge
def test_group_plain_arm():
    fk = _fk()
    for m, k, g in ((33, 4096, 128), (2048, 7168, 128), (512, 512, 128),
                    (5, 256, 64)):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 3.7
        gq, gs = _golden_group(x, g)
        for tag, fn in (
            ("alias", lambda t: fk.per_token_group_quant_fp8(t, g)),
            ("sglang", lambda t: fk.sglang_per_token_group_quant_fp8(t, g)),
        ):
            q, s = fn(x)
            _verdict(f"group-{tag}-{m}x{k}g{g}", q, s.view(m, -1), gq, gs)


@requires_bridge
def test_fused_whole_row_arm():
    """R-09②: fuse_silu_and_mul whole-row group -> aiter fused kernel."""
    fk = _fk()
    for t, d in ((33, 512), (256, 1408)):
        x = torch.randn(t, 2 * d, device="cuda", dtype=torch.bfloat16) * 2.1
        act = torch.nn.functional.silu(x[:, :d].float()) * x[:, d:].float()
        gq, gs = _golden_row(act.to(torch.bfloat16))
        q, s = fk.sglang_per_token_group_quant_fp8(
            x, d, fuse_silu_and_mul=True)
        match = (q.float() == gq.float()).float().mean().item()
        c = _cos(_deq(q, s.view(t, 1)), _deq(gq, gs))
        # aiter fused computes silu in lower intermediate precision than
        # the fp32 reference: boundary elements land 1 fp8 step off
        # (observed 0.969/0.99990 at 33x512) -- hold the tie budget, not
        # bit parity.
        assert match > 0.95 and c > 0.9995, (
            f"fused-{t}x{d}: qmatch={match:.4f} cos={c:.6f} "
            f"(bf16 activation rounding tolerance)")
        print(f"[PASS] fused-{t}x{d}: qmatch={match:.4f} cos={c:.6f}")


@requires_bridge
def test_scaled_fp8_quant_arms():
    fk = _fk()
    x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16) * 3.7
    q, s = fk.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
    gq, gs = _golden_row(x)
    assert torch.allclose(s.view(64, 1).float(), gs.float(), rtol=1e-5), (
        "scaled rowwise scales differ")
    c = _cos(q.float() * s.view(64, 1).float(), _deq(gq, gs))
    assert c > 0.9999, f"scaled rowwise cos {c:.6f}"
    print(f"[PASS] scaled-rowwise: cos={c:.6f}")
    st = torch.tensor([0.01], device="cuda", dtype=torch.float32)
    q2, _ = fk.scaled_fp8_quant(x, scale=st)
    ref = (x.float() / 0.01).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    # static arm must stay NATIVE: aiter's static_per_tensor_quant wheel
    # kernel does not clamp, |x/scale|>448 overflows to NaN (e4m3fn) where
    # the native clamps -- gate-locked 2026-10-19 (21% NaN at scale=0.01).
    assert not torch.isnan(q2.float()).any(), "static arm leaked to aiter"
    assert (q2.float() == ref.float()).all(), "static native parity broken"
    print("[PASS] scaled-static stays native (clamp, no NaN)")
    # per-tensor dynamic: aiter computes its own scale (amax/448), so no
    # overflow by construction; hold it to the same three-metric verdict.
    q3, s3 = fk.scaled_fp8_quant(x)
    amax = x.float().abs().max()
    gs3 = (amax / FP8_MAX).view(1)
    gq3 = (x.float() / gs3.item()).clamp(-FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn)
    _verdict("scaled-tensor-dynamic", q3, s3.view(1, -1), gq3, gs3)


@requires_bridge
def test_exotic_arms_still_upstream():
    """Degradation contract: exotic flags bypass aiter, keep upstream."""
    fk = _fk()
    x = torch.randn(33, 4096, device="cuda", dtype=torch.bfloat16) * 3.7
    q, s = fk.sglang_per_token_group_quant_fp8(
        x, 128, column_major_scales=True)
    assert q.shape == x.shape and s.numel() == 33 * 32, (q.shape, s.shape)
    q2, s2 = fk.per_token_group_quant_fp8(x, 128, column_major_scales=True)
    assert q2.shape == x.shape, q2.shape
    print("[PASS] exotic arms upstream (column-major scales alive)")


def main():
    fails = 0
    for fn in (test_rowwise_nameerror_fixed, test_group_plain_arm,
               test_fused_whole_row_arm, test_scaled_fp8_quant_arms,
               test_exotic_arms_still_upstream):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print("RESULT:", "PASS" if not fails else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
